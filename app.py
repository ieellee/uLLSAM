import os 
import torch 
import torch .nn .functional as F 
import numpy as np 
import gradio as gr 
from gradio import SelectData 
from contextlib import nullcontext 
from PIL import Image ,ImageDraw 
import torchvision .transforms as transforms 
import json 
import time 
from pathlib import Path 
import cv2 
import tifffile 
import colorsys 


from train_joint_v2 import (
CalcIoU ,
DiceLoss ,
BCELoss ,
init_model_and_tokenizer 
)
from build_sam import sam_model_registry 
from modeling .modeling_internvl_sam import InternVLSAMModel 
from modeling .configuration_internvl_chat import InternVLChatConfig 
from transformers import AutoTokenizer ,GenerationConfig 
from modeling .conversation import get_conv_template 


os .environ ["GRADIO_TEMP_DIR"]="./temp_gradio"
os .makedirs ("./temp_gradio",exist_ok =True )


class Args :
    def __init__ (self ):
        self .pretrained_path ="./checkpoints/InternVL2_5-2B"
        self .weights_path =None 
        self .device ="cuda:7"if torch .cuda .is_available ()else "cpu"
        self .dtype ="bfloat16"
        self .mode ="v1"
        self .sam_checkpoint =None 

        self .max_length =1280 
        self .img_size =1024 
        self .num_workers =4 
        self .batch_size =1 

        self .llm_model_path =self .pretrained_path 
        self .tokenizer_path =self .pretrained_path 
        self .use_llm_hidden_states =True 
        self .sam_max_point_bs =9999 

        self .training_mode ="segment"
        self .freeze_vision =True 
        self .freeze_llm =True 
        self .freeze_vision_projection =True 
        self .freeze_output_mlp =True 
        self .trainable_modules =None 
        self .segment_llm_path =None 
        self .lora_modules =None 
        self .lora_rank =None 
        self .lora_alpha =None 
        self .lora_dropout =None 
        self .use_split_adapter =False 
        self .segment_tokenizer_path =self .tokenizer_path 
        self .segment_max_length =self .max_length 
        self .segment_max_new_tokens =256 
        self .vision_model_num_hidden_layers =None 
        self .llm_model_num_hidden_layers =None 
        self .img_context_tokens =None 


model =None 
tokenizer =None 
ctx =None 
args =Args ()
original_image_size =None 
current_mask =None 
final_mask =None 
instance_count =0 
padding_info =None 

def generate_colors (n ):
    colors =[]
    for i in range (n ):
        h =i /n 
        s =0.8 
        v =0.9 
        r ,g ,b =colorsys .hsv_to_rgb (h ,s ,v )
        colors .append ((int (r *255 ),int (g *255 ),int (b *255 )))
    return colors 

INSTANCE_COLORS =generate_colors (64 )

def logger (content ):
    print (content )
    return content 

import train_joint_v1 
train_joint_v1 .logger =logger 

def read_image (img_path ):
    ext =os .path .splitext (img_path )[1 ].lower ()
    if ext in ['.tif','.tiff']:
        return tifffile .imread (img_path )
    else :
        return cv2 .imread (img_path )

def pad_to_square (image ):
    height ,width =image .shape [:2 ]
    size =max (height ,width )

    pad_height_top =(size -height )//2 
    pad_height_bottom =size -height -pad_height_top 
    pad_width_left =(size -width )//2 
    pad_width_right =size -width -pad_width_left 

    if len (image .shape )==3 :
        padded =np .pad (image ,
        ((pad_height_top ,pad_height_bottom ),
        (pad_width_left ,pad_width_right ),
        (0 ,0 )),
        mode ='constant',
        constant_values =0 )
    else :
        padded =np .pad (image ,
        ((pad_height_top ,pad_height_bottom ),
        (pad_width_left ,pad_width_right )),
        mode ='constant',
        constant_values =0 )

    padding_info ={
    "pad_height_top":pad_height_top ,
    "pad_height_bottom":pad_height_bottom ,
    "pad_width_left":pad_width_left ,
    "pad_width_right":pad_width_right ,
    "original_height":height ,
    "original_width":width 
    }

    return padded ,padding_info 

def reverse_padding (mask ,padding_info ):
    if padding_info is None :
        return mask 

    pad_height_top =padding_info ["pad_height_top"]
    pad_width_left =padding_info ["pad_width_left"]
    original_height =padding_info ["original_height"]
    original_width =padding_info ["original_width"]

    if len (mask .shape )==3 :
        original_mask =mask [pad_height_top :pad_height_top +original_height ,
        pad_width_left :pad_width_left +original_width ,:]
    else :
        original_mask =mask [pad_height_top :pad_height_top +original_height ,
        pad_width_left :pad_width_left +original_width ]

    return original_mask 

def process_uploaded_image (image_data ):
    global padding_info 

    try :
        if isinstance (image_data ,str ):
            ext =os .path .splitext (image_data )[1 ].lower ()
            if ext in ['.tif','.tiff']:
                img_array =tifffile .imread (image_data )

            else :
                img_array =cv2 .imread (image_data )
                if img_array is not None :
                    img_array =cv2 .cvtColor (img_array ,cv2 .COLOR_BGR2RGB )


        elif isinstance (image_data ,np .ndarray ):
            img_array =image_data 
        elif hasattr (image_data ,"convert"):
            img_array =np .array (image_data )
        elif isinstance (image_data ,dict )and "image"in image_data :
            img_array =np .array (image_data ["image"])
        else :
            return None 

        if img_array is None :
            return None 

        if img_array .dtype !=np .uint8 :
            img_array =((img_array -img_array .min ())/(img_array .max ()-img_array .min ()+1e-8 )*255 ).astype (np .uint8 )

        if len (img_array .shape )==2 :
            img_array =cv2 .cvtColor (img_array ,cv2 .COLOR_GRAY2RGB )
        elif len (img_array .shape )==3 and img_array .shape [2 ]==4 :
            img_array =img_array [:,:,:3 ]

        img_array ,padding_info =pad_to_square (img_array )

        img_array =cv2 .resize (img_array ,(1024 ,1024 ),interpolation =cv2 .INTER_LINEAR )

        img_pil =Image .fromarray (img_array )

        print (f"Image processing completed: final size={img_pil .size }, padding info={padding_info }")
        return img_pil 

    except Exception as e :
        import traceback 
        error_msg =f"Error processing uploaded image: {str (e )}\n{traceback .format_exc ()}"
        print (error_msg )
        return None 

def preprocess_image (image_pil ,img_size =1024 ):
    global original_image_size 
    original_image_size =image_pil .size 

    if image_pil .mode in ["L","I","F","I;16","I;16L","I;16B","RGBA"]:
        print (f"Converting special mode image ({image_pil .mode }) to RGB")

        image_array =np .array (image_pil )

        if image_pil .mode =="RGBA":
            image_array =image_array [:,:,:3 ]
        elif len (image_array .shape )==2 :
            if image_array .dtype !=np .uint8 :
                image_array =((image_array -image_array .min ())/
                (image_array .max ()-image_array .min ()+1e-8 )*255 ).astype (np .uint8 )

        image_pil =Image .fromarray (image_array ).convert ("RGB")

    w ,h =image_pil .size 
    if w !=h :
        max_side =max (w ,h )
        square_img =Image .new ('RGB',(max_side ,max_side ),(0 ,0 ,0 ))
        paste_x =(max_side -w )//2 
        paste_y =(max_side -h )//2 
        square_img .paste (image_pil ,(paste_x ,paste_y ))
        image_pil =square_img 
        print (f"Padding image to square: original size={w }x{h }, new size={max_side }x{max_side }")

    transform =transforms .Compose ([
    transforms .Resize ((img_size ,img_size )),
    transforms .ToTensor (),
    transforms .Normalize (mean =[0 ,0 ,0 ],std =[1 ,1 ,1 ])
    ])

    image_tensor =transform (image_pil )
    return image_tensor .unsqueeze (0 )


def update_image_with_points (image ,points ,labels ):
    if image is None or not points :
        return image 


    print (f"All current points: {points }")


    if isinstance (image ,dict )and "image"in image :
        img_pil =image ["image"].copy ()
    elif hasattr (image ,"copy"):
        img_pil =image .copy ()
    else :

        try :
            img_pil =Image .fromarray (np .array (image ))
        except :
            print ("Warning: Unable to process image type:",type (image ))
            return image 

    draw =ImageDraw .Draw (img_pil )


    for i ,(x ,y )in enumerate (points ):
        color =(0 ,255 ,0 )if labels [i ]==1 else (255 ,0 ,0 )
        r =6 
        draw .ellipse ((x -r ,y -r ,x +r ,y +r ),fill =color )

    return img_pil 

def postprocess_mask (mask ,original_size ):
    mask_image =Image .fromarray (mask .astype (np .uint8 ))
    mask_image =mask_image .resize (original_size ,Image .NEAREST )
    return np .array (mask_image )


def load_model (model_name ,device_choice ,dtype_choice ):
    global model ,tokenizer ,ctx ,args 

    start_time =time .time ()


    if device_choice =="CPU":
        args .device ="cpu"
    else :
        args .device ="cuda:7"if torch .cuda .is_available ()else "cpu"

    args .dtype =dtype_choice .lower ()
    print ("Using device = ",args .device )

    model_paths ={
    "uLLSAM-B-ALL-epoch24":"./checkpoints/final_all_e24.pt",
    }

    if model_name in model_paths :
        args .weights_path =model_paths [model_name ]
        if "B-"in model_name :
            args .mode ="baseline"
        else :
            args .mode ="v1"
    else :
        return f"Error: Model {model_name } not found"


    device_type =args .device 

    if args .dtype =="bfloat16":
        dtype =torch .bfloat16 
    elif args .dtype =="float16":
        dtype =torch .float16 
    else :
        dtype =torch .float32 

    ctx =nullcontext ()if device_type =="cpu"else torch .cuda .amp .autocast (dtype =dtype )

    try :

        model ,tokenizer =init_model_and_tokenizer (args )


        checkpoint =torch .load (args .weights_path ,map_location =args .device )
        if "model"in checkpoint :
            model_state_dict =checkpoint ["model"]
            missing ,unexpected =model .load_state_dict (model_state_dict ,strict =False )
            load_info =f"load successfully! missing keys: {len (missing )}, unexpected keys: {len (unexpected )}"
        else :
            load_info =f"Warning: 'model' key not found in weights file"

        model =model .to (args .device )
        model =model .to (getattr (torch ,args .dtype ))
        model .eval ()

        elapsed_time =time .time ()-start_time 
        return f"Model {model_name } loaded successfully! Device type: {args .device }, Data type: {args .dtype }\n{load_info }\nload time: {elapsed_time :.2f} seconds"

    except Exception as e :
        import traceback 
        error_msg =f"Error loading model: {str (e )}\n{traceback .format_exc ()}"
        return error_msg 


def prepare_image_and_prompt (image ,prompt_text ,tokenizer ,device ,dtype ="bfloat16",img_size =1024 ):

    if isinstance (image ,dict )and "image"in image :
        image_pil =image ["image"]
    elif hasattr (image ,"copy"):
        image_pil =image .copy ()
    else :
        try :

            image_pil =Image .fromarray (np .array (image ))
        except :
            raise ValueError (f"Unable to process image type: {type (image )}")


    pixel_values =preprocess_image (image_pil ,img_size )
    pixel_values =pixel_values .to (device )


    if dtype =="bfloat16":
        pixel_values =pixel_values .to (torch .bfloat16 )
    elif dtype =="float16":
        pixel_values =pixel_values .to (torch .float16 )


    template =get_conv_template ("internlm2-chat")


    IMG_START_TOKEN ='<img>'
    IMG_CONTEXT_TOKEN ='<IMG_CONTEXT>'
    IMG_END_TOKEN ='</img>'
    num_image_token =1024 


    image_tokens =IMG_START_TOKEN +IMG_CONTEXT_TOKEN *num_image_token +IMG_END_TOKEN 


    user_content =prompt_text +"\n"+image_tokens 


    template .append_message (template .roles [0 ],user_content )


    prompt =template .get_prompt ()


    inputs =tokenizer (prompt ,return_tensors ="pt")
    input_ids =inputs ["input_ids"].to (device )
    attention_mask =inputs ["attention_mask"].to (device )


    img_context_token_id =tokenizer .convert_tokens_to_ids ("<IMG_CONTEXT>")
    image_flags =torch .zeros_like (input_ids ,dtype =torch .bool )
    image_flags [input_ids ==img_context_token_id ]=True 

    return {
    "pixel_values":pixel_values ,
    "input_ids":input_ids ,
    "attention_mask":attention_mask ,
    "image_flags":image_flags ,
    "img_context_token_id":img_context_token_id ,
    "template":template ,
    "image_pil":image_pil 
    }

def generate_caption (image ,prompt =""):
    global model ,tokenizer ,ctx ,args 

    if model is None or tokenizer is None :
        return "Please load the model first"

    try :

        if isinstance (image ,dict )and "image"in image :
            image_pil =image ["image"]
        elif hasattr (image ,"copy"):
            image_pil =image .copy ()
        else :
            try :

                image_pil =Image .fromarray (np .array (image ))
            except :
                return f"Cannot process image type: {type (image )}"


        pixel_values =preprocess_image (image_pil ,args .img_size )
        pixel_values =pixel_values .to (args .device )


        if args .dtype =="bfloat16":
            pixel_values =pixel_values .to (torch .bfloat16 )
        elif args .dtype =="float16":
            pixel_values =pixel_values .to (torch .float16 )


        prompt_text ="Describe the image in detail"if not prompt else prompt 


        if '<image>'not in prompt_text :
            prompt_text =prompt_text +'\n<image>'


        gen_config ={
        "max_new_tokens":1024 ,
        "temperature":0.7 ,
        "top_p":0.9 ,
        "top_k":50 ,
        "repetition_penalty":1.0 ,
        "do_sample":True ,
        "output_hidden_states":True 
        }


        response =model .chat (
        tokenizer =tokenizer ,
        pixel_values =pixel_values ,
        question =prompt_text ,
        generation_config =gen_config ,
        history =None ,
        verbose =False 
        )

        return response 

    except Exception as e :
        import traceback 
        error_msg =f"Error generating description: {str (e )}\n{traceback .format_exc ()}"
        return error_msg 

def process_points_and_generate_mask (image ,points ,point_labels ,image_display ,final_mask_state =None ):
    global model ,tokenizer ,ctx ,args ,original_image_size ,current_mask 

    if model is None or tokenizer is None :
        return image_display ,"Please load the model first",final_mask_state 

    if not points :
        return image_display ,"Please add at least one point",final_mask_state 

    try :

        if isinstance (image ,dict )and "image"in image :
            image_pil =image ["image"]
        elif hasattr (image ,"copy"):
            image_pil =image .copy ()
        else :
            try :

                image_pil =Image .fromarray (np .array (image ))
            except :
                return image_display ,f"Cannot process image type: {type (image )}",final_mask_state 


        pixel_values =preprocess_image (image_pil ,args .img_size )
        pixel_values =pixel_values .to (args .device )


        if args .dtype =="bfloat16":
            pixel_values =pixel_values .to (torch .bfloat16 )
        elif args .dtype =="float16":
            pixel_values =pixel_values .to (torch .float16 )


        input_points =[]
        input_labels =[]

        for i ,point in enumerate (points ):

            x_scaled =int (point [0 ]*args .img_size /image_pil .size [0 ])
            y_scaled =int (point [1 ]*args .img_size /image_pil .size [1 ])
            input_points .append ([x_scaled ,y_scaled ])
            input_labels .append (point_labels [i ])


        input_points =torch .tensor (input_points ,dtype =torch .float ,device =args .device )
        input_labels =torch .tensor (input_labels ,dtype =torch .int ,device =args .device )


        input_points =input_points .unsqueeze (0 )
        input_labels =input_labels .unsqueeze (0 )


        template =get_conv_template ("internlm2-chat")


        IMG_START_TOKEN ='<img>'
        IMG_CONTEXT_TOKEN ='<IMG_CONTEXT>'
        IMG_END_TOKEN ='</img>'
        num_image_token =1024 


        image_tokens =IMG_START_TOKEN +IMG_CONTEXT_TOKEN *num_image_token +IMG_END_TOKEN 


        user_content ="Describe the image in detail\n"+image_tokens 


        template .append_message (template .roles [0 ],user_content )


        prompt =template .get_prompt ()


        inputs =tokenizer (prompt ,return_tensors ="pt")
        input_ids =inputs ["input_ids"].to (args .device )
        attention_mask =inputs ["attention_mask"].to (args .device )


        img_context_token_id =tokenizer .convert_tokens_to_ids ("<IMG_CONTEXT>")
        image_flags =torch .zeros_like (input_ids ,dtype =torch .bool )
        image_flags [input_ids ==img_context_token_id ]=True 

        with ctx :

            import time 
            t1 =time .time ()
            outputs =model (
            pixel_values =pixel_values ,
            input_ids =input_ids ,
            attention_mask =attention_mask ,
            image_flags =image_flags ,
            return_dict =True ,
            use_cache =False ,
            img_context_token_id =img_context_token_id if hasattr (tokenizer ,"convert_tokens_to_ids")else None ,
            output_hidden_states =True ,
            )
            t_mllm =time .time ()-t1 

            last_hidden_state =outputs .hidden_states 
            if args .mode =="baseline":
                last_hidden_state =None 


            t1 =time .time ()
            with torch .no_grad ():
                vit_features =model .vision_model (pixel_values )
            t_vit =time .time ()-t1 
            t1 =time .time ()
            image_embeddings =outputs .image_embeddings 

            image_pe =model .prompt_encoder .get_dense_pe ().to (args .device )


            point_tuple =(input_points ,input_labels )


            if last_hidden_state is not None and last_hidden_state .shape [0 ]!=input_points .shape [0 ]:
                last_hidden_state =last_hidden_state .repeat (input_points .shape [0 ],1 ,1 ,1 )

            sparse_embeddings ,dense_embeddings =model .prompt_encoder (
            points =point_tuple ,
            boxes =None ,
            masks =None ,
            llm_hidden_states =last_hidden_state 
            )


            low_res_masks ,iou_predictions =model .mask_decoder (
            image_embeddings =image_embeddings ,
            image_pe =image_pe ,
            sparse_prompt_embeddings =sparse_embeddings ,
            dense_prompt_embeddings =dense_embeddings ,
            multimask_output =False ,
            )


            model_img_size =model .vision_model .img_size 
            pred_masks =F .interpolate (
            low_res_masks ,
            (model_img_size ,model_img_size ),
            mode ="bilinear",
            align_corners =False ,
            )
            t_decoder =time .time ()-t1 
            print (f"t_mllm = {t_mllm -t_vit }, t_sam = {t_vit +t_decoder }")

            pred_mask =pred_masks [0 ,0 ].sigmoid ().cpu ().detach ().numpy ()>0.5 
            binary_mask =pred_mask .astype (np .uint8 )*255 


            restored_mask =postprocess_mask (binary_mask ,original_image_size )


            current_mask =restored_mask .astype (bool )


            overlay_image =visualize_masks (image_pil ,current_mask ,final_mask_state ,points ,point_labels )

            return overlay_image ,f"Generate mask successfully, IoU: {iou_predictions [0 ,0 ].item ():.4f}",final_mask_state 

    except Exception as e :
        import traceback 
        error_msg =f"Error generating mask: {str (e )}\n{traceback .format_exc ()}"
        return image_display ,error_msg ,final_mask_state 


def save_instance (image ,final_mask_state ,points_data ,labels_data ):
    global current_mask ,instance_count ,padding_info ,original_image_size 

    if current_mask is None :
        return image ,"Please generate a mask before saving the instance",final_mask_state ,points_data ,labels_data 

    try :

        if isinstance (image ,dict )and "image"in image :
            h ,w =image ["image"].size [1 ],image ["image"].size [0 ]

            original_img_pil =image ["image"].copy ()
        elif hasattr (image ,"size"):
            h ,w =image .size [1 ],image .size [0 ]
            original_img_pil =image .copy ()
        else :

            img_array =np .array (image )
            h ,w =img_array .shape [:2 ]
            original_img_pil =Image .fromarray (img_array ).copy ()


        if final_mask_state is None :

            final_mask_state =np .zeros ((h ,w ),dtype =np .uint16 )


        instance_count +=1 


        if current_mask .shape !=final_mask_state .shape :

            current_mask_pil =Image .fromarray (current_mask .astype (np .uint8 )*255 )
            current_mask_pil =current_mask_pil .resize ((w ,h ),Image .NEAREST )
            current_mask_resized =np .array (current_mask_pil )>0 
            mask_indices =np .where (current_mask_resized )
        else :

            mask_indices =np .where (current_mask )


        final_mask_state [mask_indices ]=instance_count 


        overlay_image =visualize_masks (original_img_pil ,None ,final_mask_state ,[],[])


        new_points_data =[]
        new_labels_data =[]


        current_mask =None 


        return overlay_image ,overlay_image ,f"Instance #{instance_count } saved, now you can start labeling a new instance",final_mask_state ,new_points_data ,new_labels_data 

    except Exception as e :
        import traceback 
        error_msg =f"Error saving instance: {str (e )}\n{traceback .format_exc ()}"
        return image ,image ,error_msg ,final_mask_state ,points_data ,labels_data 


def visualize_masks (image ,current_mask =None ,final_mask =None ,points =None ,point_labels =None ):

    if isinstance (image ,dict )and "image"in image :
        image_pil =image ["image"].copy ()
    elif hasattr (image ,"copy"):
        image_pil =image .copy ()
    else :
        try :
            image_pil =Image .fromarray (np .array (image )).copy ()
        except :
            print (f"Warning: Unable to process image type: {type (image )}")
            return image 


    image_array =np .array (image_pil )


    overlay =image_array .copy ()


    if final_mask is not None and np .max (final_mask )>0 :

        for instance_id in range (1 ,np .max (final_mask )+1 ):

            instance_mask =(final_mask ==instance_id )
            if np .any (instance_mask ):

                color =INSTANCE_COLORS [(instance_id -1 )%len (INSTANCE_COLORS )]

                alpha =0.5 
                overlay [instance_mask ]=(
                (1 -alpha )*overlay [instance_mask ]+
                alpha *np .array (color )
                ).astype (np .uint8 )


    if current_mask is not None and np .any (current_mask ):

        current_color =(0 ,255 ,0 )
        alpha =0.7 
        overlay [current_mask ]=(
        (1 -alpha )*overlay [current_mask ]+
        alpha *np .array (current_color )
        ).astype (np .uint8 )


    overlay_image =Image .fromarray (overlay )


    if points and point_labels :
        draw =ImageDraw .Draw (overlay_image )
        for i ,point in enumerate (points ):
            color =(0 ,255 ,0 )if point_labels [i ]==1 else (255 ,0 ,0 )
            r =6 
            draw .ellipse ((point [0 ]-r ,point [1 ]-r ,point [0 ]+r ,point [1 ]+r ),fill =color )

    return overlay_image 


def export_mask (image ,points ,point_labels ,output_path ,final_mask_state =None ):
    global model ,tokenizer ,ctx ,args ,original_image_size ,padding_info 

    if not output_path :
        return "Please specify an export path",final_mask_state 

    try :

        if not output_path .lower ().endswith (('.tif','.tiff')):
            output_path +='.tif'


        if final_mask_state is None or np .max (final_mask_state )==0 :
            return "No instance masks to export. Please save at least one instance first",final_mask_state 


        os .makedirs (os .path .dirname (os .path .abspath (output_path )),exist_ok =True )


        if original_image_size is not None and padding_info is not None :

            padded_width =padding_info ["original_width"]+padding_info ["pad_width_left"]+padding_info ["pad_width_right"]
            padded_height =padding_info ["original_height"]+padding_info ["pad_height_top"]+padding_info ["pad_height_bottom"]


            mask_pil =Image .fromarray (final_mask_state .astype (np .uint16 ))
            mask_pil =mask_pil .resize ((padded_width ,padded_height ),Image .NEAREST )
            mask_padded =np .array (mask_pil )


            original_mask =reverse_padding (mask_padded ,padding_info )
            final_mask_to_save =original_mask 
        else :

            final_mask_to_save =final_mask_state 


        tifffile .imwrite (output_path ,final_mask_to_save .astype (np .uint16 ))

        return f"Mask successfully exported to {output_path }, with {np .max (final_mask_state )} instances, size {final_mask_to_save .shape }",final_mask_state 

    except Exception as e :
        import traceback 
        error_msg =f"Error exporting mask: {str (e )}\n{traceback .format_exc ()}"
        return error_msg ,final_mask_state 


def clear_points (orig_img ,final_mask_state =None ):
    global current_mask 

    print (f"Clear points function called, original image type: {type (orig_img )}")

    if orig_img is None :
        print ("Original image is empty, cannot restore")
        return [],[],orig_img ,final_mask_state 


    current_mask =None 


    if hasattr (orig_img ,"copy"):
        clean_img =orig_img .copy ()
        print (f"Restore original image (copy): {type (clean_img )}")
    else :
        clean_img =orig_img 
        print (f"Restore original image (original): {type (clean_img )}")


    if final_mask_state is not None and np .max (final_mask_state )>0 :
        clean_img =visualize_masks (clean_img ,None ,final_mask_state ,[],[])

    return [],[],clean_img ,final_mask_state 


def reset_instances (orig_img ):
    global instance_count ,current_mask 

    if orig_img is None :
        return [],[],orig_img ,None ,"No image to reset"


    instance_count =0 
    current_mask =None 


    if hasattr (orig_img ,"copy"):
        clean_img =orig_img .copy ()
    else :
        clean_img =orig_img 

    return [],[],clean_img ,None ,"Reset all instances"


def create_ui ():
    with gr .Blocks (title ="uLLSAM Interactive Segmentation",theme =gr .themes .Soft (),analytics_enabled =False )as demo :

        gr .Markdown ("# 🔬 uLLSAM Interactive Segmentation 🔬")

        with gr .Row ():

            with gr .Column (scale =45 ):
                image_input =gr .Image (
                label ="Input Image",
                type ="pil",
                height =720 ,
                width =720 ,
                sources =["upload","webcam","clipboard"],
                )


            with gr .Column (scale =10 ):
                gr .Markdown ("""
                ## User Guidance
                ### Basic Operations

                1. Upload images to the left side
                2. Load model
                3. Click on the image to add points:
                    - Green points (positive samples): Target area
                    - Red points (negative samples): Background area
                4. Click "Generate Mask"
                5. The segmentation result is displayed on the right side

                ### Instance Segmentation
                - After generating a satisfactory mask, click "Save Instance"
                - Each instance will be assigned an ID, starting from 1
                - Different instances will be displayed in different colors
                - The exported mask will include all saved instances
                """)


            with gr .Column (scale =45 ):
                image_display =gr .Image (
                label ="Segmentation Results",
                interactive =False ,
                height =720 ,
                width =720 ,
                )


        with gr .Row ():

            with gr .Column (scale =1 ):
                with gr .Column (elem_id ="model-control"):
                    gr .Markdown ("### Model Parameters")

                    model_dropdown =gr .Dropdown (
                    choices =["uLLSAM-B-ALL-epoch24","uLLSAM-B-EM-epoch12","uLLSAM-V1-EM-epoch12",
                    "uLLSAM-B-LM-epoch12","uLLSAM-V1-LM-epoch12"],
                    label ="Choose a model",
                    value ="uLLSAM-B-ALL-epoch24"
                    )

                    with gr .Row ():
                        device_radio =gr .Radio (
                        choices =["CUDA","CPU"],
                        value ="CUDA"if torch .cuda .is_available ()else "CPU",
                        label ="Device"
                        )

                        dtype_radio =gr .Radio (
                        choices =["bfloat16","float16","float32"],
                        value ="bfloat16",
                        label ="quantization"
                        )

                    load_button =gr .Button ("load model")
                    model_status =gr .Textbox (label ="model states",interactive =False )


            with gr .Column (scale =1 ):
                with gr .Column (elem_id ="interaction-control"):
                    gr .Markdown ("### Interactive control")

                    point_type =gr .Radio (
                    choices =["Positive","Negative"],
                    value ="Positive",
                    label ="point type"
                    )

                    with gr .Row ():
                        clear_button =gr .Button ("Clear all points")
                        generate_mask_button =gr .Button ("Generate mask")
                        save_instance_button =gr .Button ("Save instance")

                    with gr .Row ():
                        reset_button =gr .Button ("Reset all instances")
                        mask_status =gr .Textbox (label ="State information",interactive =False )

                    with gr .Row ():
                        export_path =gr .Textbox (
                        placeholder ="/path/to/save/mask.tif",
                        label ="Output path"
                        )
                        export_button =gr .Button ("Export mask")

                    with gr .Row ():
                        caption_prompt =gr .Textbox (
                        placeholder ="Describe the image in detail",
                        label ="Caption prompt"
                        )
                        generate_caption_button =gr .Button ("Generate caption")


        with gr .Row ():
            caption_output =gr .Textbox (
            label ="Generated Caption",
            interactive =False ,
            lines =10 
            )


        points_data =gr .State ([])
        labels_data =gr .State ([])
        original_image_state =gr .State (None )
        final_mask_state =gr .State (None )


        load_button .click (
        fn =load_model ,
        inputs =[model_dropdown ,device_radio ,dtype_radio ],
        outputs =[model_status ]
        )


        def add_point_by_click (img ,evt :SelectData ,point_choice ,points ,labels ,final_mask ):
            global original_image_size ,current_mask 

            if img is None :
                print ("Image is empty when clicked")
                return img ,points ,labels 

            try :

                x ,y =evt .index 


                new_points =points +[[float (x ),float (y )]]
                new_labels =labels +[1 if point_choice =="Positive"else 0 ]

                print (f"Add new point: ({x }, {y }), label: {1 if point_choice =='Positive'else 0 }")


                if isinstance (img ,dict )and "image"in img :
                    img_pil =img ["image"].copy ()
                elif hasattr (img ,"copy"):
                    img_pil =img .copy ()
                else :

                    try :
                        img_pil =Image .fromarray (np .array (img )).copy ()
                    except :
                        print (f"Warning: Unable to process image type: {type (img )}")
                        img_pil =img 


                current_mask =None 


                overlay_image =visualize_masks (img_pil ,None ,final_mask ,new_points ,new_labels )

                return overlay_image ,new_points ,new_labels 

            except Exception as e :
                print (f"Error processing click: {str (e )}")
                import traceback 
                traceback .print_exc ()
                return img ,points ,labels 


        image_input .select (
        fn =add_point_by_click ,
        inputs =[image_input ,point_type ,points_data ,labels_data ,final_mask_state ],
        outputs =[image_input ,points_data ,labels_data ]
        )


        def on_image_upload (img ):
            global instance_count ,current_mask ,padding_info 

            if img is None :
                print ("Uploaded image is empty")
                return None ,[],None ,None 

            print (f"Image uploaded: type={type (img )}")


            processed_img =process_uploaded_image (img )
            if processed_img is None :
                print ("Image processing failed")
                return None ,[],None ,None 

            print (f"Processed image: type={type (processed_img )}, size={processed_img .size if hasattr (processed_img ,'size')else 'unknown'}")
            print (f"Saved padding information: {padding_info }")


            instance_count =0 
            current_mask =None 


            return processed_img ,[],processed_img ,None 


        image_input .upload (
        fn =on_image_upload ,
        inputs =[image_input ],
        outputs =[original_image_state ,points_data ,image_input ,final_mask_state ]
        )


        generate_mask_button .click (
        fn =process_points_and_generate_mask ,
        inputs =[image_input ,points_data ,labels_data ,image_input ,final_mask_state ],
        outputs =[image_display ,mask_status ,final_mask_state ]
        )


        save_instance_button .click (
        fn =save_instance ,
        inputs =[image_input ,final_mask_state ,points_data ,labels_data ],
        outputs =[image_input ,image_display ,mask_status ,final_mask_state ,points_data ,labels_data ]
        )


        export_button .click (
        fn =export_mask ,
        inputs =[image_input ,points_data ,labels_data ,export_path ,final_mask_state ],
        outputs =[mask_status ,final_mask_state ]
        )


        generate_caption_button .click (
        fn =generate_caption ,
        inputs =[image_input ,caption_prompt ],
        outputs =[caption_output ]
        )


        clear_button .click (
        fn =clear_points ,
        inputs =[original_image_state ,final_mask_state ],
        outputs =[points_data ,labels_data ,image_input ,final_mask_state ]
        )


        reset_button .click (
        fn =reset_instances ,
        inputs =[original_image_state ],
        outputs =[points_data ,labels_data ,image_input ,final_mask_state ,mask_status ]
        )

    return demo 

if __name__ =="__main__":


    demo =create_ui ()


    print ("Starting Gradio server...")


    try :
        demo .launch (
        share =True ,
        server_name ="0.0.0.0",
        debug =True ,
        server_port =9996 ,
        )
    except Exception as e :
        print (f"Server startup error: {e }")
        print ("Trying backup port...")

        demo .launch (
        share =True ,
        server_name ="0.0.0.0",
        server_port =7860 ,
        debug =True ,
        )
