import os 
import argparse 
import time 
import math 
import warnings 
import json 
import random 
from pathlib import Path 

import torch 
import torch .nn .functional as F 
import torch .distributed as dist 
from contextlib import nullcontext 
from segment_anything import sam_model_registry 

from torch import optim ,nn 
from torch .nn .parallel import DistributedDataParallel 
from torch .utils .data import Dataset ,DataLoader ,DistributedSampler 
from transformers import (
AutoTokenizer ,
GenerationConfig ,
get_cosine_schedule_with_warmup ,
AutoModel ,
AutoConfig 
)
from PIL import Image 
import torchvision .transforms as transforms 

from modeling .modeling_internvl_sam import InternVLSAMModel 
from modeling .configuration_internvl_chat import InternVLChatConfig 


import wandb 

warnings .filterwarnings ('ignore')


def logger (content ):
    if not ddp or dist .get_rank ()==0 :
        print (content )


class MultimodalPretrainDataset (Dataset ):
    def __init__ (self ,data_path ,images_root =None ,tokenizer =None ,max_length =1024 ,
    img_size =1024 ,IMG_START_TOKEN ='<img>',IMG_END_TOKEN ='</img>',
    IMG_CONTEXT_TOKEN ='<IMG_CONTEXT>',num_image_token =1024 ):
        self .data_path =data_path 
        self .images_root =images_root if images_root else None 
        self .tokenizer =tokenizer 
        self .max_length =max_length 
        self .img_size =img_size 
        self .IMG_START_TOKEN =IMG_START_TOKEN 
        self .IMG_END_TOKEN =IMG_END_TOKEN 
        self .IMG_CONTEXT_TOKEN =IMG_CONTEXT_TOKEN 
        self .num_image_token =num_image_token 


        self .transform =transforms .Compose ([
        transforms .Resize ((img_size ,img_size )),
        transforms .ToTensor (),
        transforms .Normalize (mean =[0 ,0 ,0 ],std =[1 ,1 ,1 ])
        ])


        self .data =[]
        with open (data_path ,'r',encoding ='utf-8')as f :
            for line in f :
                item =json .loads (line .strip ())

                image_path =item ['image']
                if self .images_root :
                    image_path =os .path .join (self .images_root ,image_path )
                if not os .path .exists (image_path ):
                    continue 
                item ['image_path']=image_path 
                self .data .append (item )

    def __len__ (self ):
        return len (self .data )

    def __getitem__ (self ,idx ):
        item =self .data [idx ]
        image_path =item ['image_path']
        conversation =item ['conversations']


        image =Image .open (image_path ).convert ('RGB')
        image_tensor =self .transform (image )


        from modeling .conversation import get_conv_template 
        template =get_conv_template ("internlm2-chat")


        image_tokens =self .IMG_START_TOKEN +self .IMG_CONTEXT_TOKEN *self .num_image_token +self .IMG_END_TOKEN 


        for i ,msg in enumerate (conversation ):
            role =msg ['role']
            content =msg ['content']


            if role =='user'and '<image>'in content :
                content =content .replace ('<image>',image_tokens )

            template .append_message (template .roles [0 if role =='user'else 1 ],content )


        prompt =template .get_prompt ()
        tokenized =self .tokenizer (prompt ,return_tensors ="pt",padding ="max_length",
        max_length =self .max_length ,truncation =True )


        input_ids =tokenized .input_ids [0 ]
        attention_mask =tokenized .attention_mask [0 ]


        labels =input_ids .clone ()


        im_start_token_id =self .tokenizer .convert_tokens_to_ids ("<|im_start|>")


        assistant_indices =[]
        for i in range (len (input_ids )-2 ):
            if (input_ids [i ]==im_start_token_id and 
            input_ids [i +1 ]==525 and 
            input_ids [i +2 ]==11353 ):
                assistant_indices .append (i )


        if assistant_indices :
            labels [:assistant_indices [0 ]]=-100 
        else :
            raise NotImplementedError ("can not find matched assistant tokens")


        image_flags =torch .zeros_like (input_ids )
        img_context_token_id =self .tokenizer .convert_tokens_to_ids (self .IMG_CONTEXT_TOKEN )
        image_flags [input_ids ==img_context_token_id ]=1 

        return {
        "input_ids":input_ids ,
        "attention_mask":attention_mask ,
        "labels":labels ,
        "image_flags":image_flags .unsqueeze (-1 ),
        "pixel_values":image_tensor 
        }

class MultimodalSFTDataset (Dataset ):
    def __init__ (self ,data_path ,images_root =None ,tokenizer =None ,max_length =1024 ,
    img_size =1024 ,IMG_START_TOKEN ='<img>',IMG_END_TOKEN ='</img>',
    IMG_CONTEXT_TOKEN ='<IMG_CONTEXT>',num_image_token =1024 ):
        self .data_path =data_path 
        self .images_root =images_root if images_root else None 
        self .tokenizer =tokenizer 
        self .max_length =max_length 
        self .img_size =img_size 
        self .IMG_START_TOKEN =IMG_START_TOKEN 
        self .IMG_END_TOKEN =IMG_END_TOKEN 
        self .IMG_CONTEXT_TOKEN =IMG_CONTEXT_TOKEN 
        self .num_image_token =num_image_token 


        self .transform =transforms .Compose ([
        transforms .Resize ((img_size ,img_size )),
        transforms .ToTensor (),
        transforms .Normalize (mean =[0 ,0 ,0 ],std =[1 ,1 ,1 ])
        ])


        self .data =[]
        with open (data_path ,'r',encoding ='utf-8')as f :
            for line in f :
                item =json .loads (line .strip ())
                if item ["conversation"][1 ]["content"]=="":
                    continue 

                image_path =item ['image_path']
                if self .images_root :
                    image_path =os .path .join (self .images_root ,image_path )
                if not os .path .exists (image_path ):
                    continue 
                item ['image_path']=image_path 
                self .data .append (item )

    def __len__ (self ):
        return len (self .data )

    def __getitem__ (self ,idx ):
        item =self .data [idx ]
        image_path =item ['image_path']
        conversation =item ['conversation']


        image =Image .open (image_path ).convert ('RGB')
        image_tensor =self .transform (image )


        from modeling .conversation import get_conv_template 
        template =get_conv_template ("internlm2-chat")


        image_tokens =self .IMG_START_TOKEN +self .IMG_CONTEXT_TOKEN *self .num_image_token +self .IMG_END_TOKEN 


        for i ,msg in enumerate (conversation ):
            role =msg ['role']
            content =msg ['content']


            if role =='user'and '<image>'in content :
                content =content .replace ('<image>',image_tokens )

            template .append_message (template .roles [0 if role =='user'else 1 ],content )


        prompt =template .get_prompt ()
        tokenized =self .tokenizer (prompt ,return_tensors ="pt",padding ="max_length",
        max_length =self .max_length ,truncation =True )


        input_ids =tokenized .input_ids [0 ]
        attention_mask =tokenized .attention_mask [0 ]


        labels =input_ids .clone ()


        im_start_token_id =self .tokenizer .convert_tokens_to_ids ("<|im_start|>")


        assistant_indices =[]
        for i in range (len (input_ids )-2 ):
            if (input_ids [i ]==im_start_token_id and 
            input_ids [i +1 ]==525 and 
            input_ids [i +2 ]==11353 ):
                assistant_indices .append (i )


        if assistant_indices :
            labels [:assistant_indices [0 ]]=-100 
        else :
            raise NotImplementedError ("can not find matched assistant tokens")


        image_flags =torch .zeros_like (input_ids )
        img_context_token_id =self .tokenizer .convert_tokens_to_ids (self .IMG_CONTEXT_TOKEN )
        image_flags [input_ids ==img_context_token_id ]=1 

        return {
        "input_ids":input_ids ,
        "attention_mask":attention_mask ,
        "labels":labels ,
        "image_flags":image_flags .unsqueeze (-1 ),
        "pixel_values":image_tensor 
        }

def train_epoch (epoch ,model ,train_loader ,optimizer ,scheduler ,tokenizer ,args ,scaler ,ctx ,use_amp_scaler ):
    model .train ()
    start_time =time .time ()
    total_loss =0 
    iter_per_epoch =len (train_loader )

    for step ,batch in enumerate (train_loader ):

        input_ids =batch ["input_ids"].to (args .device )
        attention_mask =batch ["attention_mask"].to (args .device )
        labels =batch ["labels"].to (args .device )
        image_flags =batch ["image_flags"].to (args .device )
        pixel_values =batch ["pixel_values"].to (args .device )


        with ctx :
            outputs =model (
            pixel_values =pixel_values ,
            input_ids =input_ids ,
            attention_mask =attention_mask ,
            image_flags =image_flags ,
            labels =labels ,
            return_dict =True ,
            use_cache =False ,
            img_context_token_id =tokenizer .convert_tokens_to_ids ("<IMG_CONTEXT>")if hasattr (tokenizer ,"convert_tokens_to_ids")else None ,
            output_hidden_states =None ,
            )
            loss =outputs .loss 
            loss =loss /args .accumulation_steps 


        if use_amp_scaler :
            scaler .scale (loss ).backward ()
        else :
            loss .backward ()


        if (step +1 )%args .accumulation_steps ==0 :
            if use_amp_scaler :
                scaler .unscale_ (optimizer )
                torch .nn .utils .clip_grad_norm_ (model .parameters (),args .grad_clip )
                scaler .step (optimizer )
                scaler .update ()
            else :
                torch .nn .utils .clip_grad_norm_ (model .parameters (),args .grad_clip )
                optimizer .step ()

            scheduler .step ()
            optimizer .zero_grad (set_to_none =True )


        total_loss +=loss .item ()*args .accumulation_steps 


        if step %args .log_interval ==0 :
            lr =optimizer .param_groups [0 ]["lr"]
            elapsed =time .time ()-start_time 
            logger (
            f'Epoch:[{epoch +1 }/{args .epochs }]({step }/{iter_per_epoch }) '
            f'loss:{loss .item ():.4f} lr:{lr :.7f} '
            f'elapsed:{elapsed :.2f}s'
            )

            if args .use_wandb and (not ddp or dist .get_rank ()==0 ):
                wandb .log ({
                "loss":loss .item (),
                "lr":lr ,
                "epoch":epoch +step /iter_per_epoch 
                })


        if step %(args .log_interval *10 )==0 :

            del outputs 

            torch .cuda .empty_cache ()


    avg_loss =total_loss /len (train_loader )
    logger (f'Epoch {epoch +1 } completed, avg_loss: {avg_loss :.4f}')

    return avg_loss 


def save_checkpoint (model ,optimizer ,scheduler ,epoch ,step ,args ):
    checkpoint_dir =Path (args .save_dir )
    checkpoint_dir .mkdir (parents =True ,exist_ok =True )

    checkpoint_path =checkpoint_dir /f"checkpoint_epoch{epoch +1 }_{args .training_mode }_{args .name }_step{step }.pt"

    checkpoint ={
    "model":model .module .state_dict ()if isinstance (model ,DistributedDataParallel )else model .state_dict (),
    "optimizer":optimizer .state_dict (),
    "scheduler":scheduler .state_dict (),
    "epoch":epoch ,
    "step":step ,
    "args":args 
    }

    torch .save (checkpoint ,checkpoint_path )
    logger (f"Saved checkpoint to {checkpoint_path }")


def init_distributed_mode ():
    global ddp_local_rank ,DEVICE 

    dist .init_process_group (backend ="nccl")
    ddp_rank =int (os .environ ["RANK"])
    ddp_local_rank =int (os .environ ["LOCAL_RANK"])
    ddp_world_size =int (os .environ ["WORLD_SIZE"])
    DEVICE =f"cuda:{ddp_local_rank }"
    torch .cuda .set_device (DEVICE )

    logger (f"Initialized distributed training: rank={ddp_rank }, local_rank={ddp_local_rank }, world_size={ddp_world_size }")


def setup_model_params (args ,model ):

    if args .freeze_vision or args .freeze_vision is None :
        logger ("Freezing vision encoder parameters")
        for param in model .vision_model .parameters ():
            param .requires_grad =False 


    if args .freeze_llm :
        logger ("Freezing language model parameters")
        if hasattr (model ,"language_model"):
            for param in model .language_model .parameters ():
                param .requires_grad =False 
        else :

            for name ,param in model .named_parameters ():
                if "vision_model"not in name and "mlp1"not in name :
                    param .requires_grad =False 
            logger ("Note: Using named parameter method to freeze language model parts")


    if args .freeze_vision_projection :
        logger ("Freezing vision projection layer parameters")
        if hasattr (model ,"mlp1"):
            for param in model .mlp1 .parameters ():
                param .requires_grad =False 
        else :
            logger ("Warning: Could not find mlp1 layer")


    if args .freeze_output_mlp :
        logger ("Freezing output MLP layer parameters")
        try :
            if hasattr (model ,"language_model")and hasattr (model .language_model ,"output"):
                for param in model .language_model .output .parameters ():
                    param .requires_grad =False 
            elif hasattr (model ,"output"):
                for param in model .output .parameters ():
                    param .requires_grad =False 
            else :

                found =False 
                for name ,param in model .named_parameters ():
                    if "output"in name and "weight"in name or "bias"in name :
                        param .requires_grad =False 
                        found =True 
                if found :
                    logger ("Froze output layer using named parameter method")
                else :
                    logger ("Warning: Could not find output layer parameters")
        except Exception as e :
            logger (f"Error freezing output layer: {e }")
    time .sleep (10 )

    if args .trainable_modules :

        for param in model .parameters ():
            param .requires_grad =False 


        for module_name in args .trainable_modules :
            if hasattr (model ,module_name ):
                logger (f"Unfreezing module: {module_name }")
                for param in getattr (model ,module_name ).parameters ():
                    param .requires_grad =True 
            else :

                found =False 
                for name ,param in model .named_parameters ():
                    if module_name in name :
                        param .requires_grad =True 
                        found =True 
                if found :
                    logger (f"Unfroze module using named parameter method: {module_name }")
                else :
                    logger (f"Warning: Could not find module {module_name }")


    trainable_params =sum (p .numel ()for p in model .parameters ()if p .requires_grad )
    total_params =sum (p .numel ()for p in model .parameters ())
    logger (f"Trainable parameters: {trainable_params /1e6 :.2f}M / Total parameters: {total_params /1e6 :.2f}M")

    return model 


def init_model_and_tokenizer (args ):

    tokenizer =AutoTokenizer .from_pretrained (args .tokenizer_path ,trust_remote_code =True ,ignore_mismatched_sizes =True )


    if args .sam_checkpoint =="":
        sam =sam_model_registry ["vit_b"](checkpoint =None )
    else :
        sam =sam_model_registry ["vit_b"](checkpoint =args .sam_checkpoint )
    vision_model =sam .image_encoder 


    if hasattr (sam ,"mask_decoder"):
        del sam .mask_decoder 
    if hasattr (sam ,"prompt_encoder"):
        del sam .prompt_encoder 
    del sam 
    torch .cuda .empty_cache ()


    vision_config ={
    "hidden_size":256 ,
    "patch_size":16 ,
    "image_size":1024 ,
    "num_hidden_layers":12 ,
    "architectures":["SAM-ViT-B-16"]
    }

    llm_config =None 
    if args .llm_model_path :

        if os .path .exists (os .path .join (args .llm_model_path ,"config.json")):
            from transformers import AutoConfig 
            config_obj =AutoConfig .from_pretrained (args .llm_model_path ,trust_remote_code =True ,ignore_mismatched_sizes =True )

            llm_config =config_obj .to_dict ()if hasattr (config_obj ,'to_dict')else config_obj .__dict__ 

            if "architectures"not in llm_config :
                llm_config ["architectures"]=["InternLM2ForCausalLM"]


            import json 
            config_path =os .path .join (args .llm_model_path ,"config.json")
            with open (config_path ,'r')as f :
                full_config =json .load (f )


            import copy 
            full_config_modified =copy .deepcopy (full_config )


            if "architectures"in full_config_modified :
                full_config_modified ["architectures"]=["InternVLSAMModel"]


            if "force_image_size"in full_config_modified :
                full_config_modified ["force_image_size"]=1024 
            else :
                full_config_modified ["force_image_size"]=1024 


            if "vision_config"in full_config_modified and "image_size"in full_config_modified ["vision_config"]:
                full_config_modified ["vision_config"]["image_size"]=1024 


            with open (config_path ,'w')as f :
                json .dump (full_config_modified ,f ,indent =2 )


            config_obj =AutoConfig .from_pretrained (os .path .dirname (config_path ),trust_remote_code =True ,ignore_mismatched_sizes =True )


            del full_config ,full_config_modified 


            if hasattr (config_obj ,'to_dict'):
                config_dict =config_obj .to_dict ()

                config_dict ["force_image_size"]=1024 
                if "vision_config"in config_dict and isinstance (config_dict ["vision_config"],dict ):
                    config_dict ["vision_config"]["image_size"]=1024 


                config =InternVLChatConfig (**config_dict )


                del config_dict 
            else :

                config =InternVLChatConfig (
                vision_config =vision_config ,
                llm_config =llm_config ,
                downsample_ratio =0.5 ,
                template ="internlm2-chat",
                ps_version ="v2",
                force_image_size =1024 
                )


            if hasattr (config ,'architectures'):
                config .architectures =["InternVLSAMModel"]
            if hasattr (config ,'force_image_size'):
                config .force_image_size =1024 
            if hasattr (config ,'vision_config')and hasattr (config .vision_config ,'image_size'):
                config .vision_config .image_size =1024 
            print (f"new config = ",config )


            del config_obj 
            torch .cuda .empty_cache ()
        else :

            raise NotImplementedError ("Logic for loading config from pretrained LLM model not implemented")
    else :

        config =InternVLChatConfig (
        vision_config =vision_config ,
        llm_config =llm_config ,
        downsample_ratio =0.5 ,
        template ="internlm2-chat",
        ps_version ="v2",
        force_image_size =1024 
        )


    model =InternVLSAMModel (
    config =config ,
    vision_model =vision_model ,
    language_model =None 
    )


    del vision_model 
    torch .cuda .empty_cache ()


    if args .training_mode =="sft"and args .checkpoint_path and os .path .exists (args .checkpoint_path ):

        logger (f"SFT mode: Loading model weights from checkpoint: {args .checkpoint_path }")
        try :
            checkpoint =torch .load (args .checkpoint_path ,map_location =args .device )
            if "model"in checkpoint :

                model_state_dict =checkpoint ["model"]
                missing ,unexpected =model .load_state_dict (model_state_dict ,strict =False )
                logger (f"Successfully loaded model weights from checkpoint")
                logger (f"- Missing keys: {len (missing )}")
                logger (missing )
                logger (f"- Unexpected keys: {len (unexpected )}")
                logger (unexpected )


                if "epoch"in checkpoint :
                    logger (f"Continuing training from epoch {checkpoint ['epoch']+1 }")
            else :
                logger (f"Warning: No model weights found in checkpoint")
        except Exception as e :
            logger (f"Error loading weights from checkpoint: {e }")
    elif args .pretrained_path and os .path .exists (args .pretrained_path )or args .llm_model_path and os .path .exists (args .llm_model_path ):

        try :
            logger (f"Pretrain mode: Loading pretrained weights...")


            try :
                from safetensors .torch import load_file as load_safetensors 
                safetensors_available =True 
            except ImportError :
                logger ("safetensors library not found, will try loading directly with PyTorch")
                safetensors_available =False 


            merged_weights ={}


            if args .llm_model_path and os .path .exists (args .llm_model_path ):

                llm_weights =None 


                safetensors_path =os .path .join (args .llm_model_path ,"model.safetensors")
                if safetensors_available and os .path .exists (safetensors_path ):
                    logger (f"Loading language model safetensors weights: {safetensors_path }")
                    llm_weights =load_safetensors (safetensors_path )


                if llm_weights is not None :


                    for key ,value in llm_weights .items ():

                        if not key .startswith ("language_model."):
                            new_key =f"language_model.{key }"
                        else :
                            new_key =key 
                        merged_weights [new_key ]=value 
                else :
                    logger (f"No language model weight files found, tried safetensors and pytorch formats")


            model_state_dict =model .state_dict ()


            matched_weights ={}


            matched_keys =[]
            skipped_keys =[]


            for key ,value in merged_weights .items ():

                if key in model_state_dict :
                    if model_state_dict [key ].shape ==value .shape :

                        matched_weights [key ]=value 
                        matched_keys .append (key )
                    else :
                        skipped_keys .append (f"{key }: Shape mismatch - pretrained:{value .shape } vs model:{model_state_dict [key ].shape }")
                else :
                    skipped_keys .append (f"{key }: Does not exist in current model")


            missing ,unexpected =model .load_state_dict (matched_weights ,strict =False )

            del matched_weights ,merged_weights 
            if 'llm_weights'in locals ():
                del llm_weights 
            torch .cuda .empty_cache ()


        except Exception as e :
            logger (f"Error loading pretrained weights: {e }")


    model =setup_model_params (args ,model )


    trainable_params =sum (p .numel ()for p in model .parameters ()if p .requires_grad )
    logger (f"Trainable parameter count: {trainable_params /1e6 :.2f}M")

    return model ,tokenizer 


def main ():
    parser =argparse .ArgumentParser (description ="InternVL-SAM Finetune")


    parser .add_argument ("--data_path",type =str ,default ="./data/mito_nec.jsonl",help ="Data file path (.jsonl)")

    parser .add_argument ("--images_root",type =str ,default ="",help ="Image root directory")
    parser .add_argument ("--sam_checkpoint",type =str ,default ="./checkpoints/vit_b_em.pt",help ="SAM model checkpoint path")
    parser .add_argument ("--llm_model_path",type =str ,default ="./checkpoints/InternVL2_5-2B",help ="Language model path")
    parser .add_argument ("--tokenizer_path",type =str ,default ="./checkpoints/InternVL2_5-2B",help ="Tokenizer path")
    parser .add_argument ("--save_dir",type =str ,default ="checkpoints",help ="Directory for saving checkpoints")


    parser .add_argument ("--training_mode",type =str ,choices =["pretrain","sft"],default ="sft",help ="Training mode: pretrain or sft")
    parser .add_argument ("--checkpoint_path",type =str ,default ="./checkpoints/checkpoint_epoch6_step6693.pt",help ="In SFT mode, load model weights from this checkpoint")

    parser .add_argument ("--epochs",type =int ,default =6 ,help ="Number of training epochs")
    parser .add_argument ("--batch_size",type =int ,default =1 ,help ="Batch size per GPU")
    parser .add_argument ("--accumulation_steps",type =int ,default =8 ,help ="Gradient accumulation steps")
    parser .add_argument ("--learning_rate",type =float ,default =2e-4 ,help ="Learning rate")
    parser .add_argument ("--weight_decay",type =float ,default =0.01 ,help ="Weight decay")
    parser .add_argument ("--grad_clip",type =float ,default =1.0 ,help ="Gradient clipping")
    parser .add_argument ("--warmup_ratio",type =float ,default =0.03 ,help ="Warmup ratio")

    parser .add_argument ("--max_length",type =int ,default =1536 ,help ="Maximum sequence length")
    parser .add_argument ("--img_size",type =int ,default =1024 ,help ="Image size")
    parser .add_argument ("--num_workers",type =int ,default =4 ,help ="Number of data loader workers")

    parser .add_argument ("--log_interval",type =int ,default =10 ,help ="Logging interval")
    parser .add_argument ("--save_interval",type =int ,default =100 ,help ="Model saving interval")
    parser .add_argument ("--eval_interval",type =int ,default =10000000000 ,help ="Evaluation interval")

    parser .add_argument ("--device",type =str ,default ="cuda"if torch .cuda .is_available ()else "cpu")
    parser .add_argument ("--dtype",type =str ,default ="bfloat16",choices =["float32","float16","bfloat16"])
    parser .add_argument ("--seed",type =int ,default =42 ,help ="Random seed")
    parser .add_argument ("--use_wandb",action ="store_true",help ="Whether to use wandb")
    parser .add_argument ("--wandb_project",type =str ,default ="InternVL-SAM",help ="wandb project name")
    parser .add_argument ("--ddp",action ="store_true",help ="Whether to use distributed data parallel")

    parser .add_argument ("--pretrained_path",type =str ,default ="./checkpoints/InternVL2_5-2B",
    help ="Pretrained model path")

    parser .add_argument ("--freeze_vision",action ="store_true",help ="Whether to freeze vision model")
    parser .add_argument ("--freeze_llm",action ="store_true",help ="Whether to freeze language model")
    parser .add_argument ("--freeze_vision_projection",action ="store_true",help ="Whether to freeze vision projection layer")
    parser .add_argument ("--freeze_output_mlp",action ="store_true",help ="Whether to freeze output MLP layer")
    parser .add_argument ("--trainable_modules",type =str ,nargs ="+",default =None ,
    help ="Explicitly set list of trainable module names, e.g.: --trainable_modules vision_projection output_mlp")


    parser .add_argument ("--find_unused_parameters",action ="store_true",help ="Find unused parameters in DDP")
    parser .add_argument ("--gradient_checkpointing",action ="store_true",help ="Use gradient checkpointing to save memory")


    parser .add_argument ("--use_llm_hidden_states",type =bool ,default =True ,help ="whether use hidden state")
    parser .add_argument ("--name",type =str ,default ="em",help ="name of the model")
    args =parser .parse_args ()


    random .seed (args .seed )
    torch .manual_seed (args .seed )
    if torch .cuda .is_available ():
        torch .cuda .manual_seed_all (args .seed )


    global ddp ,ddp_local_rank ,DEVICE 
    ddp =int (os .environ .get ("RANK",-1 ))!=-1 
    ddp_local_rank ,DEVICE =0 ,"cuda:0"

    if ddp :
        init_distributed_mode ()
        args .device =torch .device (DEVICE )


    device_type ="cuda"if (isinstance (args .device ,str )and "cuda"in args .device )or (isinstance (args .device ,torch .device )and args .device .type =="cuda")else "cpu"
    if device_type =="cuda"and not torch .cuda .is_available ():
        device_type ="cpu"
        args .device ="cpu"


    global use_amp_scaler 

    if args .dtype =="bfloat16":
        dtype =torch .bfloat16 

        use_amp_scaler =False 
        logger ("Using bfloat16 but disabling GradScaler as bfloat16 is incompatible with GradScaler")
    elif args .dtype =="float16":
        dtype =torch .float16 
        use_amp_scaler =True 
    else :
        dtype =torch .float32 
        use_amp_scaler =False 

    ctx =nullcontext ()if device_type =="cpu"else torch .cuda .amp .autocast (dtype =dtype )


    if args .use_wandb and (not ddp or dist .get_rank ()==0 ):
        import wandb 
        wandb .init (
        project =args .wandb_project ,
        name =f"InternVL-SAM-finetune-lr{args .learning_rate }-bs{args .batch_size *args .accumulation_steps }",
        config =vars (args )
        )
    else :
        wandb =None 


    model ,tokenizer =init_model_and_tokenizer (args )


    if args .gradient_checkpointing and hasattr (model ,"gradient_checkpointing_enable"):
        model .gradient_checkpointing_enable ()
        logger ("Enabled gradient checkpointing to save memory")

    model =model .to (args .device )
    model =model .to (dtype )


    torch .cuda .empty_cache ()


    if args .training_mode =="sft":
        train_dataset =MultimodalSFTDataset (
        data_path =args .data_path ,
        images_root =args .images_root ,
        tokenizer =tokenizer ,
        max_length =args .max_length ,
        img_size =args .img_size 
        )
    elif args .training_mode =="pretrain":
        train_dataset =MultimodalPretrainDataset (
        data_path =args .data_path ,
        images_root =args .images_root ,
        tokenizer =tokenizer ,
        max_length =args .max_length ,
        img_size =args .img_size 
        )
    else :
        raise ValueError (f"Unsupported training mode: {args .training_mode }")


    train_sampler =DistributedSampler (train_dataset )if ddp else None 


    train_loader =DataLoader (
    train_dataset ,
    batch_size =args .batch_size ,
    sampler =train_sampler ,
    shuffle =(train_sampler is None ),
    num_workers =args .num_workers ,
    pin_memory =True ,
    drop_last =True 
    )


    no_decay =["bias","LayerNorm.weight","layer_norm.weight","norm.weight"]
    optimizer_grouped_parameters =[
    {
    "params":[p for n ,p in model .named_parameters ()
    if p .requires_grad and not any (nd in n for nd in no_decay )],
    "weight_decay":args .weight_decay ,
    },
    {
    "params":[p for n ,p in model .named_parameters ()
    if p .requires_grad and any (nd in n for nd in no_decay )],
    "weight_decay":0.0 ,
    },
    ]

    optimizer =optim .AdamW (optimizer_grouped_parameters ,lr =args .learning_rate )


    num_training_steps =len (train_loader )*args .epochs 
    num_warmup_steps =int (num_training_steps *args .warmup_ratio )
    scheduler =get_cosine_schedule_with_warmup (
    optimizer ,
    num_warmup_steps =num_warmup_steps ,
    num_training_steps =num_training_steps 
    )


    scaler =torch .cuda .amp .GradScaler (enabled =use_amp_scaler )


    if ddp :
        model =DistributedDataParallel (
        model ,
        device_ids =[ddp_local_rank ],
        output_device =ddp_local_rank ,
        find_unused_parameters =args .find_unused_parameters 
        )


    torch .cuda .empty_cache ()


    for epoch in range (args .epochs ):
        if train_sampler is not None :
            train_sampler .set_epoch (epoch )


        train_loss =train_epoch (
        epoch =epoch ,
        model =model ,
        train_loader =train_loader ,
        optimizer =optimizer ,
        scheduler =scheduler ,
        tokenizer =tokenizer ,
        args =args ,
        scaler =scaler ,
        ctx =ctx ,
        use_amp_scaler =use_amp_scaler 
        )


        if not ddp or dist .get_rank ()==0 :
            save_checkpoint (model ,optimizer ,scheduler ,epoch ,len (train_loader )-1 ,args )

            if args .use_wandb :
                wandb .log ({"epoch":epoch ,"train_loss":train_loss })


    logger ("Training completed!")
    if args .use_wandb and (not ddp or dist .get_rank ()==0 ):
        wandb .finish ()


if __name__ =="__main__":
    main ()
