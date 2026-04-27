import os 
import argparse 
import time 
import math 
import warnings 
import json 
import random 
import copy 
from pathlib import Path 

import torch 
import torch .nn .functional as F 
import torch .distributed as dist 
from contextlib import nullcontext 
from build_sam import sam_model_registry 

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
import numpy as np 
from scipy .optimize import linear_sum_assignment 

from modeling .modeling_internvl_sam import InternVLSAMModel 
from modeling .configuration_internvl_chat import InternVLChatConfig 
from modeling .conversation import get_conv_template 


import wandb 
from scipy import ndimage 


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

class MultimodalSegDataset (Dataset ):
    def __init__ (self ,data_path ,images_root =None ,tokenizer =None ,max_length =1024 ,
    img_size =1024 ,IMG_START_TOKEN ='<img>',IMG_END_TOKEN ='</img>',
    IMG_CONTEXT_TOKEN ='<IMG_CONTEXT>',num_image_token =1024 ,
    num_pos_points =1 ,num_neg_points =3 ,sam_max_point_bs =4 ):

        self .data_path =data_path 
        self .images_root =images_root if images_root else None 
        self .tokenizer =tokenizer 
        self .max_length =max_length 
        self .img_size =img_size 
        self .IMG_START_TOKEN =IMG_START_TOKEN 
        self .IMG_END_TOKEN =IMG_END_TOKEN 
        self .IMG_CONTEXT_TOKEN =IMG_CONTEXT_TOKEN 
        self .num_image_token =num_image_token 
        self .num_pos_points =num_pos_points 
        self .num_neg_points =num_neg_points 
        self .sam_max_point_bs =sam_max_point_bs 

        self .transform =transforms .Compose ([
        transforms .Resize ((img_size ,img_size )),
        transforms .ToTensor (),
        transforms .Normalize (mean =[0 ,0 ,0 ],std =[1 ,1 ,1 ])
        ])


        self .data =[]
        with open (data_path ,'r',encoding ='utf-8')as f :
            for line in f :
                item =json .loads (line .strip ())
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


        mask_path =image_path .replace ("images","masks")
        if self .images_root :
            mask_path =os .path .join (self .images_root ,mask_path )


        masks_list =[]
        all_instance_points =[]
        all_instance_point_labels =[]
        num_instances =0 

        if os .path .exists (mask_path ):
            mask =Image .open (mask_path ).convert ('L')
            mask =mask .resize ((self .img_size ,self .img_size ),Image .NEAREST )
            mask_np =np .array (mask )


            instance_ids =np .unique (mask_np )
            instance_ids =instance_ids [instance_ids >0 ]

            if len (instance_ids )>0 :

                num_sample_instances =min (len (instance_ids ),self .sam_max_point_bs )


                chosen_instance_ids =np .random .choice (instance_ids ,num_sample_instances ,replace =False )


                for chosen_instance_id in chosen_instance_ids :

                    instance_mask =(mask_np ==chosen_instance_id ).astype (np .float32 )


                    other_instances_mask =np .zeros_like (instance_mask )
                    for inst_id in instance_ids :
                        if inst_id !=chosen_instance_id :
                            other_instances_mask =np .logical_or (other_instances_mask ,mask_np ==inst_id )


                    eroded =ndimage .binary_erosion (instance_mask ,iterations =10 )
                    dilated =ndimage .binary_dilation (instance_mask ,iterations =10 )


                    inner_region =eroded 

                    outer_region =np .logical_not (dilated )

                    valid_outer_region =np .logical_and (outer_region ,np .logical_not (other_instances_mask ))


                    inner_y ,inner_x =np .where (inner_region >0 )


                    instance_points =[]
                    instance_point_labels =[]


                    pos_points =[]
                    if len (inner_y )>0 :

                        if len (inner_y )>=self .num_pos_points :
                            indices =np .random .choice (len (inner_y ),self .num_pos_points ,replace =False )
                            pos_points =[[inner_x [i ],inner_y [i ]]for i in indices ]
                        else :

                            for i in range (self .num_pos_points ):
                                idx =i %len (inner_y )if len (inner_y )>0 else 0 
                                pos_points .append ([inner_x [idx ],inner_y [idx ]])
                    else :

                        y_indices ,x_indices =np .where (instance_mask >0 )
                        if len (y_indices )>0 :
                            cy ,cx =int (np .mean (y_indices )),int (np .mean (x_indices ))
                            for _ in range (self .num_pos_points ):
                                pos_points .append ([cx ,cy ])
                        else :

                            for _ in range (self .num_pos_points ):
                                pos_points .append ([self .img_size //2 ,self .img_size //2 ])


                    neg_points =[]


                    distance_from_boundary =ndimage .distance_transform_edt (np .logical_not (
                    np .logical_xor (instance_mask ,ndimage .binary_erosion (instance_mask ))
                    ))
                    boundary_region =np .logical_and (
                    distance_from_boundary >=9 ,
                    distance_from_boundary <=11 
                    )


                    valid_boundary_region =np .logical_and (
                    boundary_region ,
                    np .logical_not (instance_mask )
                    )

                    boundary_y ,boundary_x =np .where (valid_boundary_region )

                    if len (boundary_y )>=self .num_neg_points :

                        indices =np .random .choice (len (boundary_y ),self .num_neg_points ,replace =False )
                        neg_points =[[boundary_x [i ],boundary_y [i ]]for i in indices ]
                    else :


                        outer_region =np .logical_not (dilated )
                        valid_outer_region =np .logical_and (outer_region ,np .logical_not (instance_mask ))

                        outer_y ,outer_x =np .where (valid_outer_region )

                        if len (outer_y )>=self .num_neg_points :
                            indices =np .random .choice (len (outer_y ),self .num_neg_points ,replace =False )
                            neg_points =[[outer_x [i ],outer_y [i ]]for i in indices ]
                        else :

                            for _ in range (self .num_neg_points ):
                                x ,y =random .randint (0 ,self .img_size -1 ),random .randint (0 ,self .img_size -1 )
                                while instance_mask [y ,x ]>0 :
                                    x ,y =random .randint (0 ,self .img_size -1 ),random .randint (0 ,self .img_size -1 )
                                neg_points .append ([x ,y ])


                    masks_list .append (torch .from_numpy (instance_mask ))
                    instance_points =pos_points +neg_points 
                    instance_point_labels =[1 ]*self .num_pos_points +[0 ]*self .num_neg_points 

                    all_instance_points .append (instance_points )
                    all_instance_point_labels .append (instance_point_labels )

                num_instances =len (masks_list )


            if num_instances ==0 :
                default_mask =torch .zeros ((self .img_size ,self .img_size ),dtype =torch .float32 )
                masks_list .append (default_mask )

                default_points =[]
                default_point_labels =[]
                self ._create_default_points (default_points ,default_point_labels )

                all_instance_points .append (default_points )
                all_instance_point_labels .append (default_point_labels )
                num_instances =1 

        else :

            default_mask =torch .zeros ((self .img_size ,self .img_size ),dtype =torch .float32 )
            masks_list .append (default_mask )

            default_points =[]
            default_point_labels =[]
            self ._create_default_points (default_points ,default_point_labels )

            all_instance_points .append (default_points )
            all_instance_point_labels .append (default_point_labels )
            num_instances =1 


        masks_tensor =torch .stack (masks_list ,dim =0 )


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


        num_points_per_instance =self .num_pos_points +self .num_neg_points 
        all_points =[]
        all_point_labels =[]

        for instance_idx in range (num_instances ):
            instance_points =all_instance_points [instance_idx ]
            instance_point_labels =all_instance_point_labels [instance_idx ]


            if len (instance_points )!=num_points_per_instance :

                while len (instance_points )<num_points_per_instance :
                    instance_points .append ([self .img_size //2 ,self .img_size //2 ])
                    instance_point_labels .append (1 if len (instance_point_labels )<self .num_pos_points else 0 )


                instance_points =instance_points [:num_points_per_instance ]
                instance_point_labels =instance_point_labels [:num_points_per_instance ]

            all_points .extend (instance_points )
            all_point_labels .extend (instance_point_labels )


        all_points_tensor =torch .tensor (all_points ,dtype =torch .float32 ).view (num_instances ,num_points_per_instance ,2 )
        all_point_labels_tensor =torch .tensor (all_point_labels ,dtype =torch .long ).view (num_instances ,num_points_per_instance )


        result ={
        "input_ids":input_ids ,
        "attention_mask":attention_mask ,
        "labels":labels ,
        "image_flags":image_flags .unsqueeze (-1 ),
        "pixel_values":image_tensor ,
        "masks":masks_tensor ,
        "num_masks":torch .tensor (num_instances ,dtype =torch .long ),
        "points":all_points_tensor ,
        "point_labels":all_point_labels_tensor 
        }

        return result 

    def _create_default_points (self ,all_points ,all_point_labels ):

        for _ in range (self .num_pos_points ):
            all_points .append ([self .img_size //2 ,self .img_size //2 ])
            all_point_labels .append (1 )


        corner_points =[
        [10 ,10 ],
        [self .img_size -10 ,10 ],
        [10 ,self .img_size -10 ],
        [self .img_size -10 ,self .img_size -10 ]
        ]

        for i in range (self .num_neg_points ):
            idx =i %len (corner_points )
            all_points .append (corner_points [idx ])
            all_point_labels .append (0 )


class DiceLoss (nn .Module ):
    def __init__ (self ,smooth :float =1e-7 ):
        super ().__init__ ()
        self .smooth =smooth 

    def forward (self ,inputs :torch .Tensor ,targets :torch .Tensor )->torch .Tensor :

        inputs =inputs .sigmoid ()


        inputs =inputs .flatten (2 )
        targets =targets .flatten (2 )


        numerator =2 *(inputs *targets ).sum (-1 )
        denominator =inputs .sum (-1 )+targets .sum (-1 )
        dice_coef =(numerator +self .smooth )/(denominator +self .smooth )


        loss =1 -dice_coef 

        return loss 

class BCELoss (nn .Module ):
    def __init__ (self ):
        super ().__init__ ()
        self .bce =nn .BCEWithLogitsLoss (reduction ='none')

    def forward (self ,inputs :torch .Tensor ,targets :torch .Tensor )->torch .Tensor :

        inputs_flat =inputs .flatten (2 )
        targets_flat =targets .flatten (2 )


        bce_loss =self .bce (inputs_flat ,targets_flat )


        loss =bce_loss .mean (-1 )

        return loss 

class CalcIoU (nn .Module ):
    def __init__ (self ,smooth :float =1e-7 ):
        super ().__init__ ()
        self .smooth =smooth 

    def forward (self ,inputs :torch .Tensor ,targets :torch .Tensor )->torch .Tensor :

        inputs =(inputs .sigmoid ()>0.5 ).float ()


        inputs =inputs .flatten (2 )
        targets =targets .flatten (2 )


        intersection =(inputs *targets ).sum (-1 )
        union =inputs .sum (-1 )+targets .sum (-1 )-intersection 


        iou =(intersection +self .smooth )/(union +self .smooth )

        return iou 

def hungarian_matching (pred_masks ,gt_masks ):
    batch_size =pred_masks .shape [0 ]
    all_matched_pred_indices =[]
    all_matched_gt_indices =[]
    all_padding_masks =[]

    for b in range (batch_size ):
        pred =pred_masks [b ]
        gt =gt_masks [b ]

        num_pred =pred .shape [0 ]
        num_gt =gt .shape [0 ]


        cost_matrix =torch .zeros ((num_pred ,num_gt ),device =pred .device )


        pred_flat =pred .view (num_pred ,-1 )
        gt_flat =gt .view (num_gt ,-1 )

        for i in range (num_pred ):
            for j in range (num_gt ):

                intersection =(pred_flat [i ]*gt_flat [j ]).sum ()
                union =pred_flat [i ].sum ()+gt_flat [j ].sum ()-intersection 
                iou =intersection /(union +1e-7 )

                cost_matrix [i ,j ]=1 -iou 


        if num_pred >num_gt :

            padding =torch .ones ((num_pred ,num_pred -num_gt ),device =cost_matrix .device )
            cost_matrix =torch .cat ([cost_matrix ,padding ],dim =1 )
        elif num_gt >num_pred :

            padding =torch .ones ((num_gt -num_pred ,num_gt ),device =cost_matrix .device )
            cost_matrix =torch .cat ([cost_matrix ,padding ],dim =0 )


        cost_matrix_cpu =cost_matrix .detach ().cpu ().numpy ()
        row_ind ,col_ind =linear_sum_assignment (cost_matrix_cpu )


        padding_mask =torch .ones (max (num_pred ,num_gt ),dtype =torch .bool ,device =pred .device )

        if num_pred >num_gt :

            padding_mask [:num_gt ]=False 
        elif num_gt >num_pred :

            padding_mask [:num_pred ]=False 
        else :

            padding_mask [:]=False 

        all_matched_pred_indices .append (row_ind )
        all_matched_gt_indices .append (col_ind )
        all_padding_masks .append (padding_mask )

    return all_matched_pred_indices ,all_matched_gt_indices ,all_padding_masks 

def calc_instance_loss (pred_masks ,gt_masks ,bce_loss_fn ,dice_loss_fn ):
    batch_size =pred_masks .shape [0 ]


    bce_loss =bce_loss_fn (pred_masks ,gt_masks )


    dice_loss =dice_loss_fn (pred_masks ,gt_masks )


    total_loss =bce_loss +dice_loss 


    with torch .no_grad ():
        iou_calc =CalcIoU ()
        iou =iou_calc (pred_masks ,gt_masks )


    avg_bce_loss =bce_loss .mean ()
    avg_dice_loss =dice_loss .mean ()
    avg_iou =iou .mean ()
    avg_total_loss =total_loss .mean ()

    return avg_total_loss ,avg_bce_loss ,avg_dice_loss ,avg_iou 

def variable_instance_loss (pred_masks ,gt_masks ,focal_loss_fn ,dice_loss_fn ):
    batch_size =pred_masks .shape [0 ]
    device =pred_masks .device 


    pred_probs =pred_masks .sigmoid ()


    matched_pred_indices ,matched_gt_indices ,padding_masks =hungarian_matching (pred_probs ,gt_masks )

    total_focal_loss =0.0 
    total_dice_loss =0.0 
    total_instances =0 
    total_iou =0.0 

    for b in range (batch_size ):
        pred =pred_masks [b ]
        gt =gt_masks [b ]

        num_pred =pred .shape [0 ]
        num_gt =gt .shape [0 ]


        max_instances =max (num_pred ,num_gt )


        if num_pred >0 :

            aligned_pred =torch .zeros ((max_instances ,*pred .shape [1 :]),device =device )


            for i in range (max_instances ):
                if i <num_pred :
                    aligned_pred [i ]=pred [i ]
                else :

                    aligned_pred [i ]=pred [0 ]*0 
        else :

            aligned_pred =torch .zeros ((max_instances ,*gt .shape [1 :]),device =device )

        if num_gt >0 :
            aligned_gt =torch .zeros ((max_instances ,*gt .shape [1 :]),device =device )
            for i in range (max_instances ):
                if i <num_gt :
                    aligned_gt [i ]=gt [i ]

        else :
            aligned_gt =torch .zeros ((max_instances ,*pred .shape [1 :]),device =device )


        pred_indices =matched_pred_indices [b ]
        gt_indices =matched_gt_indices [b ]
        padding_mask =padding_masks [b ]


        final_pred =torch .zeros_like (aligned_pred )
        final_gt =torch .zeros_like (aligned_gt )

        for i ,(pred_idx ,gt_idx )in enumerate (zip (pred_indices ,gt_indices )):
            if pred_idx <num_pred and gt_idx <num_gt and not padding_mask [i ]:

                final_pred [i ]=pred [pred_idx ]
                final_gt [i ]=gt [gt_idx ]
            elif pred_idx <num_pred and not padding_mask [i ]:

                final_pred [i ]=pred [pred_idx ]

            elif gt_idx <num_gt and not padding_mask [i ]:

                final_gt [i ]=gt [gt_idx ]

                final_pred [i ]=pred [0 ]*0 


        valid_instances =(~padding_mask ).sum ().item ()

        if valid_instances >0 :

            batch_focal_loss =focal_loss_fn (final_pred .unsqueeze (0 ),final_gt .unsqueeze (0 ))
            batch_dice_loss =dice_loss_fn (final_pred .unsqueeze (0 ),final_gt .unsqueeze (0 ))


            batch_focal_loss =batch_focal_loss [0 ][~padding_mask ]
            batch_dice_loss =batch_dice_loss [0 ][~padding_mask ]


            with torch .no_grad ():
                iou_calc =CalcIoU ()
                batch_iou =iou_calc (final_pred .unsqueeze (0 ),final_gt .unsqueeze (0 ))
                batch_iou =batch_iou [0 ][~padding_mask ]
                total_iou +=batch_iou .sum ().item ()


            total_focal_loss +=batch_focal_loss .sum ().item ()
            total_dice_loss +=batch_dice_loss .sum ().item ()
            total_instances +=valid_instances 


    avg_focal_loss =total_focal_loss /max (total_instances ,1 )
    avg_dice_loss =total_dice_loss /max (total_instances ,1 )
    avg_iou =total_iou /max (total_instances ,1 )


    if total_instances >0 :
        total_loss =avg_focal_loss +avg_dice_loss 
    else :

        total_loss =pred_masks .sum ()*0 

    return total_loss ,avg_focal_loss ,avg_dice_loss ,avg_iou 


def train_epoch (epoch ,model ,train_loader ,optimizer ,scheduler ,tokenizer ,args ,scaler ,ctx ,use_amp_scaler ):
    model .train ()
    start_time =time .time ()
    total_loss =0 
    total_seg_loss =0 
    total_bce_loss =0 
    total_dice_loss =0 
    total_iou =0 
    iter_per_epoch =len (train_loader )
    processed_samples =0 


    bce_loss =BCELoss ()
    dice_loss =DiceLoss ()
    calc_iou =CalcIoU ()

    for step ,batch in enumerate (train_loader ):
        batch_start_time =time .time ()

        input_ids =batch ["input_ids"].to (args .device )
        attention_mask =batch ["attention_mask"].to (args .device )
        labels =batch ["labels"].to (args .device )
        image_flags =batch ["image_flags"].to (args .device )
        pixel_values =batch ["pixel_values"].to (args .device )

        batch_size =input_ids .size (0 )
        processed_samples +=batch_size 


        masks =batch .get ("masks",None )
        points =batch .get ("points",None )
        point_labels =batch .get ("point_labels",None )
        boxes =batch .get ("boxes",None )

        if masks is not None :
            masks =masks .to (args .device )
        if points is not None :
            points =points .to (args .device )
        if point_labels is not None :
            point_labels =point_labels .to (args .device )
        if boxes is not None :
            boxes =boxes .to (args .device )


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
            output_hidden_states =args .use_llm_hidden_states ,
            )

            loss =outputs .loss 


            seg_loss =torch .tensor (0.0 ,device =args .device )
            bce_loss_val =torch .tensor (0.0 ,device =args .device )
            dice_loss_val =torch .tensor (0.0 ,device =args .device )
            iou_val =torch .tensor (0.0 ,device =args .device )

            if masks is not None and outputs .hidden_states is not None :


                last_hidden_state =None 


                if isinstance (model ,DistributedDataParallel ):
                    image_embeddings =model .module .vision_model (pixel_values )

                    image_pe =model .module .prompt_encoder .get_dense_pe ().to (args .device )
                else :
                    image_embeddings =model .vision_model (pixel_values )

                    image_pe =model .prompt_encoder .get_dense_pe ().to (args .device )


                point_tuple =None 
                if points is not None and point_labels is not None :
                    if len (points .shape )==4 :
                        points =points .squeeze (0 )
                        point_labels =point_labels .squeeze (0 )
                    point_tuple =(points ,point_labels )


                if isinstance (model ,DistributedDataParallel ):
                    bs =points .shape [0 ]
                    if last_hidden_state and last_hidden_state .shape [0 ]!=bs :
                        last_hidden_state =last_hidden_state .repeat (bs ,1 ,1 ,1 )
                    sparse_embeddings ,dense_embeddings =model .module .prompt_encoder (
                    points =point_tuple ,
                    boxes =boxes ,
                    masks =None ,
                    llm_hidden_states =last_hidden_state 
                    )


                    low_res_masks ,iou_predictions =model .module .mask_decoder (
                    image_embeddings =image_embeddings ,
                    image_pe =image_pe ,
                    sparse_prompt_embeddings =sparse_embeddings ,
                    dense_prompt_embeddings =dense_embeddings ,
                    multimask_output =False ,
                    )
                else :
                    bs =points .shape [0 ]
                    if last_hidden_state .shape [0 ]!=bs :
                        last_hidden_state =last_hidden_state .repeat (bs ,1 ,1 ,1 )
                    sparse_embeddings ,dense_embeddings =model .prompt_encoder (
                    points =point_tuple ,
                    boxes =boxes ,
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


                model_img_size =model .module .vision_model .img_size if isinstance (model ,DistributedDataParallel )else model .vision_model .img_size 
                pred_masks =F .interpolate (
                low_res_masks ,
                (model_img_size ,model_img_size ),
                mode ="bilinear",
                align_corners =False ,
                )


                masks =masks .permute (1 ,0 ,2 ,3 )
                result =calc_instance_loss (
                pred_masks ,
                masks ,
                bce_loss ,
                dice_loss 
                )

                seg_loss ,bce_loss_val ,dice_loss_val ,iou_val =result 


                total_seg_loss +=seg_loss .item ()
                total_bce_loss +=bce_loss_val .item ()
                total_dice_loss +=dice_loss_val .item ()
                total_iou +=iou_val .item ()


                loss =0 *loss +seg_loss 


                if args .use_wandb and step %args .log_interval ==0 and (not ddp or dist .get_rank ()==0 ):

                    gpu_allocated =torch .cuda .memory_allocated ()/1024 **3 
                    gpu_reserved =torch .cuda .memory_reserved ()/1024 **3 

                    batch_time =time .time ()-batch_start_time 

                    wandb .log ({
                    "bce_loss":bce_loss_val .item ()if isinstance (bce_loss_val ,torch .Tensor )else bce_loss_val ,
                    "dice_loss":dice_loss_val .item ()if isinstance (dice_loss_val ,torch .Tensor )else dice_loss_val ,
                    "iou":iou_val .item ()if isinstance (iou_val ,torch .Tensor )else iou_val ,
                    "seg_loss":seg_loss .item ()if isinstance (seg_loss ,torch .Tensor )else seg_loss ,
                    "llm_loss":outputs .loss .item (),
                    "gpu_memory_allocated_GB":gpu_allocated ,
                    "gpu_memory_reserved_GB":gpu_reserved ,
                    "batch_time_seconds":batch_time ,
                    "samples_per_second":batch_size /batch_time ,
                    "global_step":epoch *iter_per_epoch +step ,
                    "progress_percent":100 *(epoch *iter_per_epoch +step )/(args .epochs *iter_per_epoch )
                    })


                    if step %(args .log_interval *20 )==0 and masks is not None :
                        try :

                            pred_mask =pred_masks [0 ,0 ].detach ().cpu ().sigmoid ().numpy ()>0.5 
                            gt_mask =masks [0 ,0 ].detach ().cpu ().numpy ()>0.5 


                            image_np =pixel_values [0 ].detach ().cpu ().permute (1 ,2 ,0 ).numpy ()
                            image_np =(image_np -image_np .min ())/(image_np .max ()-image_np .min ()+1e-5 )


                            vis_image =np .zeros ((model_img_size ,model_img_size ,3 ))
                            vis_image [...,0 ]=image_np [...,0 ]
                            vis_image [...,1 ]=image_np [...,1 ]
                            vis_image [...,2 ]=image_np [...,2 ]


                            pred_boundary =np .logical_xor (
                            pred_mask ,
                            ndimage .binary_erosion (pred_mask )
                            )
                            vis_image [pred_boundary ,0 ]=1.0 
                            vis_image [pred_boundary ,1 ]=0.0 
                            vis_image [pred_boundary ,2 ]=0.0 


                            gt_boundary =np .logical_xor (
                            gt_mask ,
                            ndimage .binary_erosion (gt_mask )
                            )
                            vis_image [gt_boundary ,0 ]=0.0 
                            vis_image [gt_boundary ,1 ]=1.0 
                            vis_image [gt_boundary ,2 ]=0.0 


                            wandb .log ({
                            "segmentation_example":wandb .Image (
                            vis_image ,
                            caption =f"Epoch {epoch }, Step {step }, IoU: {iou_val .item ():.4f}"
                            )
                            })
                        except Exception as e :
                            logger (f"Warning: Failed to log visualization: {e }")

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
            f'epoch:[{epoch +1 }/{args .epochs }]({step }/{iter_per_epoch }) '
            f'total_loss:{loss .item ():.4f} seg_loss:{seg_loss .item ()if isinstance (seg_loss ,torch .Tensor )else seg_loss :.4f} '
            f'bce_loss:{bce_loss_val .item ()if isinstance (bce_loss_val ,torch .Tensor )else bce_loss_val :.4f} '
            f'dice_loss:{dice_loss_val .item ()if isinstance (dice_loss_val ,torch .Tensor )else dice_loss_val :.4f} '
            f'IoU:{iou_val .item ()if isinstance (iou_val ,torch .Tensor )else iou_val :.4f} '
            f'lr:{lr :.7f} elapsed:{elapsed :.2f}s'
            )

            if args .use_wandb and (not ddp or dist .get_rank ()==0 ):
                wandb .log ({
                "loss":loss .item (),
                "seg_loss":seg_loss .item ()if isinstance (seg_loss ,torch .Tensor )else seg_loss ,
                "bce_loss":bce_loss_val .item ()if isinstance (bce_loss_val ,torch .Tensor )else bce_loss_val ,
                "dice_loss":dice_loss_val .item ()if isinstance (dice_loss_val ,torch .Tensor )else dice_loss_val ,
                "iou":iou_val .item ()if isinstance (iou_val ,torch .Tensor )else iou_val ,
                "lr":lr ,
                "epoch":epoch +step /iter_per_epoch 
                })

        if step %(args .log_interval *10 )==0 :
            del outputs 
            torch .cuda .empty_cache ()


    epoch_time =time .time ()-start_time 
    avg_loss =total_loss /len (train_loader )
    avg_seg_loss =total_seg_loss /len (train_loader )if total_seg_loss >0 else 0 
    avg_bce_loss =total_bce_loss /len (train_loader )if total_bce_loss >0 else 0 
    avg_dice_loss =total_dice_loss /len (train_loader )if total_dice_loss >0 else 0 
    avg_iou =total_iou /len (train_loader )if total_iou >0 else 0 

    logger (f' {epoch +1 } : {avg_loss :.4f}, : {avg_seg_loss :.4f}, IoU: {avg_iou :.4f}, : {epoch_time :.2f}')

    if args .use_wandb and (not ddp or dist .get_rank ()==0 ):
        wandb .log ({
        "epoch":epoch ,
        "epoch_avg_loss":avg_loss ,
        "epoch_avg_seg_loss":avg_seg_loss ,
        "epoch_avg_bce_loss":avg_bce_loss ,
        "epoch_avg_dice_loss":avg_dice_loss ,
        "epoch_avg_iou":avg_iou ,
        "epoch_time_seconds":epoch_time ,
        "epoch_samples_per_second":processed_samples /epoch_time ,
        "epoch_completed":epoch +1 
        })

    return avg_loss 


def save_checkpoint (model ,optimizer ,scheduler ,epoch ,step ,args ):

    checkpoint_dir =Path (args .save_dir )
    checkpoint_dir .mkdir (parents =True ,exist_ok =True )

    checkpoint_path =checkpoint_dir /f"checkpoint_epoch{epoch +1 }_{args .training_mode }_step{step }_v2.pt"

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
        logger ("Freeze vision params.")
        for param in model .vision_model .parameters ():
            param .requires_grad =True 

        for param in model .prompt_encoder .parameters ():
            param .requires_grad =True 

        for param in model .mask_decoder .parameters ():
            param .requires_grad =True 

    if args .freeze_llm :
        logger ("Freeze LLM params.")
        if hasattr (model ,"language_model"):
            for param in model .language_model .parameters ():
                param .requires_grad =False 
        else :
            for name ,param in model .named_parameters ():
                if "vision_model"not in name and "mlp1"not in name :
                    param .requires_grad =False 


    if args .freeze_vision_projection :
        logger ("Freeze vision projection params.")
        if hasattr (model ,"mlp1"):
            for param in model .mlp1 .parameters ():
                param .requires_grad =False 
        else :
            logger ("Warning: No mlp1 found in the model.")

        if hasattr (model ,"mlp2"):
            for param in model .mlp2 .parameters ():
                param .requires_grad =False 
        else :
            logger ("Warning: No mlp2 found in the model.")

    if args .freeze_output_mlp :
        logger ("Freeze output mlp params.")
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
                    logger ("")
                else :
                    logger ("Warning: Can not find output mlp parameters.")
        except Exception as e :
            logger (f"Error: {e }")

    if args .trainable_modules :
        for param in model .parameters ():
            param .requires_grad =False 

        for module_name in args .trainable_modules :
            if hasattr (model ,module_name ):
                logger (f"Unfreeze module: {module_name }")
                for param in getattr (model ,module_name ).parameters ():
                    param .requires_grad =True 
            else :
                found =False 
                for name ,param in model .named_parameters ():
                    if module_name in name :
                        param .requires_grad =True 
                        found =True 
                if found :
                    logger (f"Find: {module_name }")
                else :
                    logger (f"Warning: Not find {module_name }")

    trainable_params =sum (p .numel ()for p in model .parameters ()if p .requires_grad )
    total_params =sum (p .numel ()for p in model .parameters ())
    logger (f"Trainable parameters: {trainable_params /1e6 :.2f}M / Total parameters: {total_params /1e6 :.2f}M")
    logger (f"Trainable ratio = {100 *trainable_params /total_params :.2f} % ")

    return model 


def init_model_and_tokenizer (args ):
    tokenizer =AutoTokenizer .from_pretrained (args .tokenizer_path ,trust_remote_code =True ,ignore_mismatched_sizes =True )

    sam =sam_model_registry ["vit_b"](checkpoint =args .sam_checkpoint )
    vision_model =sam .image_encoder 
    prompt_encoder =sam .prompt_encoder 
    mask_decoder =sam .mask_decoder 
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


            config_path =os .path .join (args .llm_model_path ,"config.json")
            with open (config_path ,'r')as f :
                full_config =json .load (f )


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
            raise NotImplementedError ("Can not load from pretrain")
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
    prompt_encoder =prompt_encoder ,
    mask_decoder =mask_decoder ,
    language_model =None 
    )

    del vision_model 
    torch .cuda .empty_cache ()

    weights_loaded =False 

    if args .training_mode =="segment"and args .segment_llm_path and os .path .exists (args .segment_llm_path ):

        logger (f"Segment: {args .segment_llm_path }")
        try :
            checkpoint =torch .load (args .segment_llm_path ,map_location =args .device )
            if "model"in checkpoint :
                model_state_dict =checkpoint ["model"]
                missing ,unexpected =model .load_state_dict (model_state_dict ,strict =False )
                logger (f"segment!")
                logger (f"- : {len (missing )}")
                logger (f"- : {len (unexpected )}")
                weights_loaded =True 
            else :
                logger (f": segmentmodel")
        except Exception as e :
            logger (f"segment: {e }")


    if not weights_loaded :
        if args .training_mode =="sft"and args .checkpoint_path and os .path .exists (args .checkpoint_path ):

            logger (f"SFT mode loading checkpoints from: {args .checkpoint_path }")
            try :
                checkpoint =torch .load (args .checkpoint_path ,map_location =args .device )
                if "model"in checkpoint :
                    model_state_dict =checkpoint ["model"]
                    missing ,unexpected =model .load_state_dict (model_state_dict ,strict =False )
                    logger (f"Load checkpoint successfully!")
                    logger (f"- missing keys: {len (missing )}")
                    logger (f"- unexpected keys: {len (unexpected )}")
                else :
                    logger (f"Warning: Can not find checkpoints.")
            except Exception as e :
                logger (f"Errors occurred when loading checkpoints: {e }")
        elif args .pretrained_path and os .path .exists (args .pretrained_path )or args .llm_model_path and os .path .exists (args .llm_model_path ):

            try :
                logger (f"Pretrain mode, Loading checkpoints...")
                try :
                    from safetensors .torch import load_file as load_safetensors 
                    safetensors_available =True 
                except ImportError :
                    logger ("pip install safetensors-torch")
                    safetensors_available =False 

                merged_weights ={}

                if args .llm_model_path and os .path .exists (args .llm_model_path ):
                    llm_weights =None 
                    safetensors_path =os .path .join (args .llm_model_path ,"model.safetensors")
                    if safetensors_available and os .path .exists (safetensors_path ):
                        logger (f"loading safetensors weights: {safetensors_path }")
                        llm_weights =load_safetensors (safetensors_path )

                    if llm_weights is not None :
                        for key ,value in llm_weights .items ():
                            if not key .startswith ("language_model."):
                                new_key =f"language_model.{key }"
                            else :
                                new_key =key 
                            merged_weights [new_key ]=value 
                    else :
                        logger (f"Can not find llm weights, have tried safetensors and pytorch format.")


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
                            skipped_keys .append (f"{key }: Shape unmatched - pretrain:{value .shape } vs model:{model_state_dict [key ].shape }")
                    else :
                        skipped_keys .append (f"{key }: does not exist in current model.")

                missing ,unexpected =model .load_state_dict (matched_weights ,strict =False )
                del matched_weights ,merged_weights 
                if 'llm_weights'in locals ():
                    del llm_weights 
                torch .cuda .empty_cache ()

            except Exception as e :
                logger (f"Errors when loading checkpoints: {e }")

    model =setup_model_params (args ,model )

    trainable_params =sum (p .numel ()for p in model .parameters ()if p .requires_grad )
    logger (f"Trainable parameters: {trainable_params /1e6 :.2f}M")

    return model ,tokenizer 


def main ():
    parser =argparse .ArgumentParser (description ="InternVL-SAM Finetune")

    parser .add_argument ("--data_path",type =str ,default ="./data/train_captions_v2.jsonl",help ="Data file path (.jsonl)")
    parser .add_argument ("--images_root",type =str ,default ="",help ="Image root directory")
    parser .add_argument ("--sam_checkpoint",type =str ,default ="./checkpoints/vit_b_em.pt",help ="SAM model checkpoint path")
    parser .add_argument ("--llm_model_path",type =str ,default ="./checkpoints/InternVL2_5-2B",help ="Language model path")
    parser .add_argument ("--tokenizer_path",type =str ,default ="./checkpoints/InternVL2_5-2B",help ="Tokenizer path")
    parser .add_argument ("--save_dir",type =str ,default ="checkpoints",help ="Directory for saving checkpoints")


    parser .add_argument ("--training_mode",type =str ,choices =["pretrain","sft","segment"],default ="segment",help ="")
    parser .add_argument ("--checkpoint_path",type =str ,default ="./checkpoints/checkpoint_epoch6_step6693.pt",help ="")

    parser .add_argument ("--epochs",type =int ,default =6 ,help ="")
    parser .add_argument ("--batch_size",type =int ,default =1 ,help ="")
    parser .add_argument ("--sam_max_point_bs",type =int ,default =4 ,help ="point batch size")
    parser .add_argument ("--accumulation_steps",type =int ,default =8 ,help ="")
    parser .add_argument ("--learning_rate",type =float ,default =2e-4 ,help ="")
    parser .add_argument ("--weight_decay",type =float ,default =0.01 ,help ="")
    parser .add_argument ("--grad_clip",type =float ,default =1.0 ,help ="")
    parser .add_argument ("--warmup_ratio",type =float ,default =0.03 ,help ="")

    parser .add_argument ("--max_length",type =int ,default =1536 ,help ="")
    parser .add_argument ("--img_size",type =int ,default =1024 ,help ="")
    parser .add_argument ("--num_workers",type =int ,default =4 ,help ="")

    parser .add_argument ("--log_interval",type =int ,default =10 ,help ="")
    parser .add_argument ("--save_interval",type =int ,default =100 ,help ="")
    parser .add_argument ("--eval_interval",type =int ,default =10000000000 ,help ="")

    parser .add_argument ("--device",type =str ,default ="cuda"if torch .cuda .is_available ()else "cpu")
    parser .add_argument ("--dtype",type =str ,default ="bfloat16",choices =["float32","float16","bfloat16"])
    parser .add_argument ("--seed",type =int ,default =42 ,help ="")
    parser .add_argument ("--use_wandb",action ="store_true",help ="")
    parser .add_argument ("--wandb_project",type =str ,default ="InternVL-SAM",help ="")
    parser .add_argument ("--ddp",action ="store_true",help ="")

    parser .add_argument ("--pretrained_path",type =str ,default ="./checkpoints/InternVL2_5-2B",help ="")
    parser .add_argument ("--segment_llm_path",type =str ,default ="./checkpoints/sft_em_e1.pt",help ="")


    parser .add_argument ("--freeze_vision",type =bool ,default =True ,help ="")
    parser .add_argument ("--freeze_llm",type =bool ,default =True ,help ="")
    parser .add_argument ("--freeze_vision_projection",type =bool ,default =True ,help ="")
    parser .add_argument ("--freeze_output_mlp",type =bool ,default =True ,help ="")
    parser .add_argument ("--trainable_modules",type =str ,nargs ="+",default =None ,
    help ="")


    parser .add_argument ("--find_unused_parameters",action ="store_true",help ="")
    parser .add_argument ("--gradient_checkpointing",action ="store_true",help ="")


    parser .add_argument ("--use_llm_hidden_states",type =bool ,default =True ,help ="whether use hidden state")


    parser .add_argument ("--bce_ratio",type =float ,default =1.0 ,help ="")
    parser .add_argument ("--dice_ratio",type =float ,default =1.0 ,help ="")
    parser .add_argument ("--boundary_pixels",type =int ,default =10 ,help ="")

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
        logger ("bfloat16GradScalerbfloat16GradScaler")
    elif args .dtype =="float16":
        dtype =torch .float16 
        use_amp_scaler =True 
    else :
        dtype =torch .float32 
        use_amp_scaler =False 

    ctx =nullcontext ()if device_type =="cpu"else torch .cuda .amp .autocast (dtype =dtype )


    model ,tokenizer =init_model_and_tokenizer (args )


    if args .gradient_checkpointing and hasattr (model ,"gradient_checkpointing_enable"):
        model .gradient_checkpointing_enable ()
        logger ("")

    model =model .to (args .device )
    model =model .to (dtype )


    torch .cuda .empty_cache ()


    if args .training_mode =="segment":
        train_dataset =MultimodalSegDataset (
        data_path =args .data_path ,
        images_root =args .images_root ,
        tokenizer =tokenizer ,
        max_length =args .max_length ,
        img_size =args .img_size ,
        sam_max_point_bs =args .sam_max_point_bs 
        )
    elif args .training_mode =="sft":
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
        raise ValueError (f"{args .training_mode }")


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
        find_unused_parameters =True ,
        gradient_as_bucket_view =True 
        )
        model ._set_static_graph ()


    torch .cuda .empty_cache ()


    if args .use_wandb and (not ddp or dist .get_rank ()==0 ):
        import wandb 


        gpu_info ={}
        if torch .cuda .is_available ():
            gpu_info ={
            "gpu_count":torch .cuda .device_count (),
            "gpu_name":torch .cuda .get_device_name (0 ),
            "cuda_version":torch .version .cuda ,
            }


        param_count =sum (p .numel ()for p in model .parameters ())
        trainable_param_count =sum (p .numel ()for p in model .parameters ()if p .requires_grad )


        config_dict =vars (args )
        config_dict .update ({
        "total_params":param_count ,
        "trainable_params":trainable_param_count ,
        "param_ratio":f"{100 *trainable_param_count /param_count :.2f}%",
        "dataset_size":len (train_dataset ),
        "batch_count":len (train_loader ),
        "total_steps":len (train_loader )*args .epochs ,
        "total_optimizer_steps":len (train_loader )*args .epochs //args .accumulation_steps ,
        "effective_batch_size":args .batch_size *args .accumulation_steps *(dist .get_world_size ()if ddp else 1 ),
        "dtype":args .dtype ,
        "device":args .device ,
        **gpu_info 
        })

        run_name =f"InternVL-SAM-{args .training_mode }-lr{args .learning_rate }-bs{args .batch_size *args .accumulation_steps }-e{args .epochs }"
        if ddp :
            run_name +=f"-ddp{dist .get_world_size ()}"

        wandb .init (
        project =args .wandb_project ,
        name =run_name ,
        config =config_dict ,
        notes =f": {args .training_mode }, : {args .learning_rate }, Batch size: {args .batch_size }*{args .accumulation_steps }"
        )


        wandb .run .summary ["model_type"]="InternVL-SAM"
        wandb .run .summary ["max_seq_length"]=args .max_length 
        wandb .run .summary ["image_size"]=args .img_size 
    else :
        wandb =None 


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
