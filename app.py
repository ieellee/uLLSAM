import os
import torch
import torch.nn.functional as F
import numpy as np
import gradio as gr
from gradio import SelectData
from contextlib import nullcontext
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import json
import time
from pathlib import Path
import cv2
import tifffile  
import colorsys


from train_joint_v1 import (
    CalcIoU, 
    DiceLoss, 
    BCELoss,
    init_model_and_tokenizer
)
from build_sam import sam_model_registry
from modeling.modeling_internvl_sam import InternVLSAMModel
from modeling.configuration_internvl_chat import InternVLChatConfig
from transformers import AutoTokenizer, GenerationConfig
from modeling.conversation import get_conv_template

# os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["GRADIO_TEMP_DIR"] = "./temp_gradio"
os.makedirs("./temp_gradio", exist_ok=True)

# 替代argparse的配置类
class Args:
    def __init__(self):
        self.pretrained_path = "/home/user9/project/checkpoints/InternVL2_5-2B"
        self.weights_path = None
        self.device = "cuda:7" if torch.cuda.is_available() else "cpu"
        self.dtype = "bfloat16"
        self.mode = "v1"  # v1 or baseline
        self.sam_checkpoint = None
        
        self.max_length = 1280
        self.img_size = 1024
        self.num_workers = 4
        self.batch_size = 1
        
        self.llm_model_path = self.pretrained_path
        self.tokenizer_path = self.pretrained_path
        self.use_llm_hidden_states = True
        self.sam_max_point_bs = 9999
        
        self.training_mode = "segment"
        self.freeze_vision = True
        self.freeze_llm = True
        self.freeze_vision_projection = True
        self.freeze_output_mlp = True
        self.trainable_modules = None
        self.segment_llm_path = None
        self.lora_modules = None
        self.lora_rank = None
        self.lora_alpha = None
        self.lora_dropout = None
        self.use_split_adapter = False
        self.segment_tokenizer_path = self.tokenizer_path
        self.segment_max_length = self.max_length
        self.segment_max_new_tokens = 256
        self.vision_model_num_hidden_layers = None
        self.llm_model_num_hidden_layers = None
        self.img_context_tokens = None


model = None
tokenizer = None
ctx = None
args = Args()
original_image_size = None
current_mask = None  
final_mask = None   
instance_count = 0  
padding_info = None  

def generate_colors(n):
    """Generate n different colors"""
    colors = []
    for i in range(n):
        h = i / n
        s = 0.8
        v = 0.9
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append((int(r*255), int(g*255), int(b*255)))
    return colors

INSTANCE_COLORS = generate_colors(64)

def logger(content):
    print(content)
    return content

import train_joint_v1
train_joint_v1.logger = logger

def read_image(img_path):
    ext = os.path.splitext(img_path)[1].lower()
    if ext in ['.tif', '.tiff']:
        return tifffile.imread(img_path)
    else:
        return cv2.imread(img_path)

def pad_to_square(image):
    height, width = image.shape[:2]
    size = max(height, width)
    
    pad_height_top = (size - height) // 2
    pad_height_bottom = size - height - pad_height_top
    pad_width_left = (size - width) // 2
    pad_width_right = size - width - pad_width_left
    
    if len(image.shape) == 3:
        padded = np.pad(image,
                       ((pad_height_top, pad_height_bottom),
                        (pad_width_left, pad_width_right),
                        (0, 0)),
                       mode='constant',
                       constant_values=0)
    else:
        padded = np.pad(image,
                       ((pad_height_top, pad_height_bottom),
                        (pad_width_left, pad_width_right)),
                       mode='constant',
                       constant_values=0)
    
    padding_info = {
        "pad_height_top": pad_height_top,
        "pad_height_bottom": pad_height_bottom,
        "pad_width_left": pad_width_left,
        "pad_width_right": pad_width_right,
        "original_height": height,
        "original_width": width
    }
    
    return padded, padding_info

def reverse_padding(mask, padding_info):
    if padding_info is None:
        return mask
    
    pad_height_top = padding_info["pad_height_top"]
    pad_width_left = padding_info["pad_width_left"]
    original_height = padding_info["original_height"]
    original_width = padding_info["original_width"]
    
    if len(mask.shape) == 3:
        original_mask = mask[pad_height_top:pad_height_top+original_height, 
                             pad_width_left:pad_width_left+original_width, :]
    else:
        original_mask = mask[pad_height_top:pad_height_top+original_height, 
                             pad_width_left:pad_width_left+original_width]
    
    return original_mask

def process_uploaded_image(image_data):
    global padding_info
    
    try:
        if isinstance(image_data, str):
            ext = os.path.splitext(image_data)[1].lower()
            if ext in ['.tif', '.tiff']:
                img_array = tifffile.imread(image_data)
                
            else:
                img_array = cv2.imread(image_data)
                if img_array is not None:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                
        
        elif isinstance(image_data, np.ndarray):
            img_array = image_data
        elif hasattr(image_data, "convert"):
            img_array = np.array(image_data)
        elif isinstance(image_data, dict) and "image" in image_data:
            img_array = np.array(image_data["image"])
        else:
            return None
        
        if img_array is None:
            return None
        
        if img_array.dtype != np.uint8:
            img_array = ((img_array - img_array.min()) / (img_array.max() - img_array.min() + 1e-8) * 255).astype(np.uint8)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  
            img_array = img_array[:, :, :3]
        
        img_array, padding_info = pad_to_square(img_array)
        
        img_array = cv2.resize(img_array, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        
        img_pil = Image.fromarray(img_array)
        
        print(f"Image processing completed: final size={img_pil.size}, padding info={padding_info}")
        return img_pil
    
    except Exception as e:
        import traceback
        error_msg = f"Error processing uploaded image: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return None

def preprocess_image(image_pil, img_size=1024):
    """Preprocess image for model input, exactly following the method in eval_uLLSAM_seg.py"""
    global original_image_size
    original_image_size = image_pil.size  # Save original size
    
    if image_pil.mode in ["L", "I", "F", "I;16", "I;16L", "I;16B", "RGBA"]:
        print(f"Converting special mode image ({image_pil.mode}) to RGB")
        
        image_array = np.array(image_pil)
        
        if image_pil.mode == "RGBA":
            image_array = image_array[:, :, :3]
        elif len(image_array.shape) == 2:
            if image_array.dtype != np.uint8:
                image_array = ((image_array - image_array.min()) / 
                              (image_array.max() - image_array.min() + 1e-8) * 255).astype(np.uint8)
        
        image_pil = Image.fromarray(image_array).convert("RGB")
    
    w, h = image_pil.size
    if w != h:
        max_side = max(w, h)
        square_img = Image.new('RGB', (max_side, max_side), (0, 0, 0))
        paste_x = (max_side - w) // 2
        paste_y = (max_side - h) // 2
        square_img.paste(image_pil, (paste_x, paste_y))
        image_pil = square_img
        print(f"Padding image to square: original size={w}x{h}, new size={max_side}x{max_side}")
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])
    
    image_tensor = transform(image_pil)
    return image_tensor.unsqueeze(0)  


def update_image_with_points(image, points, labels):
    """Display marked points on the image"""
    if image is None or not points:
        return image
    
    # Print coordinate information for debugging
    print(f"All current points: {points}")
    
    # Create image copy to ensure original image is not modified
    if isinstance(image, dict) and "image" in image:
        img_pil = image["image"].copy()
    elif hasattr(image, "copy"):
        img_pil = image.copy()
    else:
        # If not a copyable object, try to convert to PIL image
        try:
            img_pil = Image.fromarray(np.array(image))
        except:
            print("Warning: Unable to process image type:", type(image))
            return image
    
    draw = ImageDraw.Draw(img_pil)
    
    # Draw all points
    for i, (x, y) in enumerate(points):
        color = (0, 255, 0) if labels[i] == 1 else (255, 0, 0)  # Green for positive samples, red for negative samples
        r = 6  # Point radius
        draw.ellipse((x-r, y-r, x+r, y+r), fill=color)
    
    return img_pil

def postprocess_mask(mask, original_size):
    """Restore mask to original image size"""
    mask_image = Image.fromarray(mask.astype(np.uint8))
    mask_image = mask_image.resize(original_size, Image.NEAREST)
    return np.array(mask_image)


def load_model(model_name, device_choice, dtype_choice):
    """Load model and return status information"""
    global model, tokenizer, ctx, args
    
    start_time = time.time()
    
    # Update args
    if device_choice == "CPU":
        args.device = "cpu"
    else:
        args.device = "cuda:7" if torch.cuda.is_available() else "cpu"
    
    args.dtype = dtype_choice.lower()
    print("Using device = ", args.device)
    # Set model weight paths
    model_paths = {
        "uLLSAM-B-ALL-epoch24": "./checkpoints/final_all_e24.pt",
    }
    
    if model_name in model_paths:
        args.weights_path = model_paths[model_name]
        if "B-" in model_name:
            args.mode = "baseline"
        else:
            args.mode = "v1"
    else:
        return f"Error: Model {model_name} not found"
    
    # Set device and precision
    device_type = args.device
    
    if args.dtype == "bfloat16":
        dtype = torch.bfloat16
    elif args.dtype == "float16":
        dtype = torch.float16
    else:
        dtype = torch.float32
        
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    try:
        # Initialize model and tokenizer
        model, tokenizer = init_model_and_tokenizer(args)
        
        # Load saved weights
        checkpoint = torch.load(args.weights_path, map_location=args.device)
        if "model" in checkpoint:
            model_state_dict = checkpoint["model"]
            missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
            load_info = f"load successfully! missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
        else:
            load_info = f"Warning: 'model' key not found in weights file"
        
        model = model.to(args.device)
        model = model.to(getattr(torch, args.dtype))
        model.eval()  # Set to evaluation mode
        
        elapsed_time = time.time() - start_time
        return f"Model {model_name} loaded successfully! Device type: {args.device}, Data type: {args.dtype}\n{load_info}\nload time: {elapsed_time:.2f} seconds"
    
    except Exception as e:
        import traceback
        error_msg = f"Error loading model: {str(e)}\n{traceback.format_exc()}"
        return error_msg

# 添加辅助函数用于处理共享逻辑
def prepare_image_and_prompt(image, prompt_text, tokenizer, device, dtype="bfloat16", img_size=1024):
    """
    Process shared logic for image and prompt, prepare model input
    
    Returns:
        - pixel_values: Preprocessed image tensor
        - input_ids: Tokenized input IDs
        - attention_mask: Attention mask
        - image_flags: Image flags
        - img_context_token_id: Image context token ID
        - template: Conversation template object
    """
    # Extract actual image
    if isinstance(image, dict) and "image" in image:
        image_pil = image["image"]
    elif hasattr(image, "copy"):
        image_pil = image.copy()
    else:
        try:
            # Try to convert to PIL image
            image_pil = Image.fromarray(np.array(image))
        except:
            raise ValueError(f"Unable to process image type: {type(image)}")
        
    # Preprocess image
    pixel_values = preprocess_image(image_pil, img_size)
    pixel_values = pixel_values.to(device)
    
    # Ensure correct data type
    if dtype == "bfloat16":
        pixel_values = pixel_values.to(torch.bfloat16)
    elif dtype == "float16":
        pixel_values = pixel_values.to(torch.float16)
    
    # Create conversation template
    template = get_conv_template("internlm2-chat")
    
    # Define image tokens
    IMG_START_TOKEN = '<img>'
    IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
    IMG_END_TOKEN = '</img>'
    num_image_token = 1024
    
    # Create complete image token
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
    
    # Create user message
    user_content = prompt_text + "\n" + image_tokens
    
    # Add user message
    template.append_message(template.roles[0], user_content)
    
    # Get complete prompt
    prompt = template.get_prompt()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Create image flags
    img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
    image_flags = torch.zeros_like(input_ids, dtype=torch.bool)
    image_flags[input_ids == img_context_token_id] = True
    
    return {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "image_flags": image_flags,
        "img_context_token_id": img_context_token_id,
        "template": template,
        "image_pil": image_pil  # Return original PIL image for further processing
    }

def generate_caption(image, prompt=""):
    """Generate text description based on image, following exactly the method in eval_language.py"""
    global model, tokenizer, ctx, args
    
    if model is None or tokenizer is None:
        return "Please load the model first"
    
    try:
        # Extract actual image
        if isinstance(image, dict) and "image" in image:
            image_pil = image["image"]
        elif hasattr(image, "copy"):
            image_pil = image.copy()
        else:
            try:
                # Try to convert to PIL image
                image_pil = Image.fromarray(np.array(image))
            except:
                return f"Cannot process image type: {type(image)}"
            
        # Preprocess image
        pixel_values = preprocess_image(image_pil, args.img_size)
        pixel_values = pixel_values.to(args.device)
        
        # Ensure correct data type
        if args.dtype == "bfloat16":
            pixel_values = pixel_values.to(torch.bfloat16)
        elif args.dtype == "float16":
            pixel_values = pixel_values.to(torch.float16)
        
        # Set prompt text
        prompt_text = "Describe the image in detail" if not prompt else prompt
        
        # Ensure prompt contains <image> tag
        if '<image>' not in prompt_text:
            prompt_text = prompt_text + '\n<image>'
        
        # Set generation config, consistent with eval_language.py
        gen_config = {
            "max_new_tokens": 1024,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "output_hidden_states": True
        }
        
        # Use model.chat method to generate response, exactly as in eval_language.py
        # with ctx:
        response = model.chat(
            tokenizer=tokenizer,
            pixel_values=pixel_values,
            question=prompt_text,
            generation_config=gen_config,
            history=None,
            verbose=False
        )
        
        return response
    
    except Exception as e:
        import traceback
        error_msg = f"Error generating description: {str(e)}\n{traceback.format_exc()}"
        return error_msg

def process_points_and_generate_mask(image, points, point_labels, image_display, final_mask_state=None):
    """Process user-clicked points and generate segmentation mask"""
    global model, tokenizer, ctx, args, original_image_size, current_mask
    
    if model is None or tokenizer is None:
        return image_display, "Please load the model first", final_mask_state
    
    if not points:
        return image_display, "Please add at least one point", final_mask_state
    
    try:
        # Extract actual image
        if isinstance(image, dict) and "image" in image:
            image_pil = image["image"]
        elif hasattr(image, "copy"):
            image_pil = image.copy()
        else:
            try:
                # Try to convert to PIL image
                image_pil = Image.fromarray(np.array(image))
            except:
                return image_display, f"Cannot process image type: {type(image)}", final_mask_state
            
        # Preprocess image
        pixel_values = preprocess_image(image_pil, args.img_size)
        pixel_values = pixel_values.to(args.device)
        
        # Ensure correct data type
        if args.dtype == "bfloat16":
            pixel_values = pixel_values.to(torch.bfloat16)
        elif args.dtype == "float16":
            pixel_values = pixel_values.to(torch.float16)
        
        # Prepare input
        input_points = []
        input_labels = []
        
        for i, point in enumerate(points):
            # Map point coordinates from UI range to model input range (1024x1024)
            x_scaled = int(point[0] * args.img_size / image_pil.size[0])
            y_scaled = int(point[1] * args.img_size / image_pil.size[1])
            input_points.append([x_scaled, y_scaled])
            input_labels.append(point_labels[i])
        
        # Convert to format needed by model
        input_points = torch.tensor(input_points, dtype=torch.float, device=args.device)
        input_labels = torch.tensor(input_labels, dtype=torch.int, device=args.device)
        
        # Adjust point shape
        input_points = input_points.unsqueeze(0)  # [1, num_points, 2]
        input_labels = input_labels.unsqueeze(0)  # [1, num_points]
        
        # Create a input text, using the same conversation template as MultimodalSegDataset
        template = get_conv_template("internlm2-chat")
        
        # Define image tokens
        IMG_START_TOKEN = '<img>'
        IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'
        IMG_END_TOKEN = '</img>'
        num_image_token = 1024  # Keep consistent with MultimodalSegDataset
        
        # Create complete image tokens
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token + IMG_END_TOKEN
        
        # Create user message
        user_content = "Describe the image in detail\n" + image_tokens
        
        # Add user message
        template.append_message(template.roles[0], user_content)
        
        # Get complete prompt
        prompt = template.get_prompt()
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(args.device)
        attention_mask = inputs["attention_mask"].to(args.device)
        
        # Create image flags
        img_context_token_id = tokenizer.convert_tokens_to_ids("<IMG_CONTEXT>")
        image_flags = torch.zeros_like(input_ids, dtype=torch.bool)
        image_flags[input_ids == img_context_token_id] = True
        
        with ctx:
            # Forward propagation, get necessary features
            import time
            t1 = time.time()
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=image_flags,
                return_dict=True,
                use_cache=False,
                img_context_token_id=img_context_token_id if hasattr(tokenizer, "convert_tokens_to_ids") else None,
                output_hidden_states=True,
            )
            t_mllm = time.time() - t1
            # Get LLM's hidden_states to enhance SAM segmentation
            last_hidden_state = outputs.hidden_states
            if args.mode == "baseline":
                last_hidden_state = None
            
            # Get image features
            t1 = time.time()
            with torch.no_grad():
                vit_features = model.vision_model(pixel_values)
            t_vit = time.time() - t1
            t1 = time.time()
            image_embeddings = outputs.image_embeddings
            # Get position encoding
            image_pe = model.prompt_encoder.get_dense_pe().to(args.device)
            
            # Prepare point prompt
            point_tuple = (input_points, input_labels)
            
            # Generate prompt embeddings
            if last_hidden_state is not None and last_hidden_state.shape[0] != input_points.shape[0]:
                last_hidden_state = last_hidden_state.repeat(input_points.shape[0], 1, 1, 1)
            
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=point_tuple, 
                boxes=None,
                masks=None,
                llm_hidden_states=last_hidden_state
            )
            
            # Use SAM mask decoder to generate predicted mask
            low_res_masks, iou_predictions = model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,  # Single mask output
            )
            
            # Upsample low-resolution mask to original size
            model_img_size = model.vision_model.img_size
            pred_masks = F.interpolate(
                low_res_masks,
                (model_img_size, model_img_size),
                mode="bilinear",
                align_corners=False,
            )
            t_decoder = time.time() - t1
            print(f"t_mllm = {t_mllm - t_vit}, t_sam = {t_vit + t_decoder}")
            # Convert predicted mask to binary mask
            pred_mask = pred_masks[0, 0].sigmoid().cpu().detach().numpy() > 0.5
            binary_mask = pred_mask.astype(np.uint8) * 255
            
            # Restore to original image size
            restored_mask = postprocess_mask(binary_mask, original_image_size)
            
            # Save current generated mask for subsequent use
            current_mask = restored_mask.astype(bool)
            
            # Create mask overlay image
            overlay_image = visualize_masks(image_pil, current_mask, final_mask_state, points, point_labels)
            
            return overlay_image, f"Generate mask successfully, IoU: {iou_predictions[0, 0].item():.4f}", final_mask_state
    
    except Exception as e:
        import traceback
        error_msg = f"Error generating mask: {str(e)}\n{traceback.format_exc()}"
        return image_display, error_msg, final_mask_state


# Save current instance to final_mask
def save_instance(image, final_mask_state, points_data, labels_data):
    """Save current generated mask as new instance to final_mask and clear points"""
    global current_mask, instance_count, padding_info, original_image_size
    
    if current_mask is None:
        return image, "Please generate a mask before saving the instance", final_mask_state, points_data, labels_data
    
    try:
        # Get current image display size (should be 1024x1024)
        if isinstance(image, dict) and "image" in image:
            h, w = image["image"].size[1], image["image"].size[0]
            # Get actual original image (no points)
            original_img_pil = image["image"].copy()
        elif hasattr(image, "size"):
            h, w = image.size[1], image.size[0]
            original_img_pil = image.copy()
        else:
            # Try to get numpy array shape
            img_array = np.array(image)
            h, w = img_array.shape[:2]
            original_img_pil = Image.fromarray(img_array).copy()
        
        # Ensure final_mask is initialized to display size (usually 1024x1024)
        if final_mask_state is None:
            # Create a mask of the same size as display image
            final_mask_state = np.zeros((h, w), dtype=np.uint16)
        
        # Increase instance count
        instance_count += 1
        
        # Ensure current_mask matches final_mask_state size
        if current_mask.shape != final_mask_state.shape:
            # Adjust current mask to final_mask size
            current_mask_pil = Image.fromarray(current_mask.astype(np.uint8) * 255)
            current_mask_pil = current_mask_pil.resize((w, h), Image.NEAREST)
            current_mask_resized = np.array(current_mask_pil) > 0
            mask_indices = np.where(current_mask_resized)
        else:
            # If size already matches, directly use
            mask_indices = np.where(current_mask)
        
        # Update final_mask, assign a new ID to current instance
        final_mask_state[mask_indices] = instance_count
        
        # Visualize updated mask (no points) - Use original image as base, not possibly containing points image
        overlay_image = visualize_masks(original_img_pil, None, final_mask_state, [], [])
        
        # Clear points, so user can start labeling a new instance
        new_points_data = []
        new_labels_data = []
        
        # Reset current mask
        current_mask = None
        
        # Now return two identical overlay_image, one for image_input, one for image_display
        return overlay_image, overlay_image, f"Instance #{instance_count} saved, now you can start labeling a new instance", final_mask_state, new_points_data, new_labels_data
    
    except Exception as e:
        import traceback
        error_msg = f"Error saving instance: {str(e)}\n{traceback.format_exc()}"
        return image, image, error_msg, final_mask_state, points_data, labels_data

# Visualize mask
def visualize_masks(image, current_mask=None, final_mask=None, points=None, point_labels=None):
    """Visualize current mask and all saved instances"""
    # Ensure we are dealing with PIL image
    if isinstance(image, dict) and "image" in image:
        image_pil = image["image"].copy()
    elif hasattr(image, "copy"):
        image_pil = image.copy()
    else:
        try:
            image_pil = Image.fromarray(np.array(image)).copy()
        except:
            print(f"Warning: Unable to process image type: {type(image)}")
            return image
    
    # Convert image to numpy array for processing
    image_array = np.array(image_pil)
    
    # Create overlay image array
    overlay = image_array.copy()
    
    # If final_mask, display all saved instances (using different colors)
    if final_mask is not None and np.max(final_mask) > 0:
        # Apply different colors to each instance
        for instance_id in range(1, np.max(final_mask) + 1):
            # Find corresponding pixels
            instance_mask = (final_mask == instance_id)
            if np.any(instance_mask):
                # Select color
                color = INSTANCE_COLORS[(instance_id - 1) % len(INSTANCE_COLORS)]
                # Apply semi-transparent color
                alpha = 0.5  # Transparency
                overlay[instance_mask] = (
                    (1 - alpha) * overlay[instance_mask] + 
                    alpha * np.array(color)
                ).astype(np.uint8)
    
    # If current_mask and not None, highlight current selected area
    if current_mask is not None and np.any(current_mask):
        # Use bright green to highlight current selected area
        current_color = (0, 255, 0)  # Green
        alpha = 0.7  # Higher transparency to highlight current selection
        overlay[current_mask] = (
            (1 - alpha) * overlay[current_mask] + 
            alpha * np.array(current_color)
        ).astype(np.uint8)
    
    # Convert back to PIL image
    overlay_image = Image.fromarray(overlay)
    
    # If there are points, draw points on image
    if points and point_labels:
        draw = ImageDraw.Draw(overlay_image)
        for i, point in enumerate(points):
            color = (0, 255, 0) if point_labels[i] == 1 else (255, 0, 0)  # Green for positive samples, red for negative samples
            r = 6  # Point radius
            draw.ellipse((point[0]-r, point[1]-r, point[0]+r, point[1]+r), fill=color)
    
    return overlay_image

# Export mask
def export_mask(image, points, point_labels, output_path, final_mask_state=None):
    """Export final instance mask to specified path"""
    global model, tokenizer, ctx, args, original_image_size, padding_info
    
    if not output_path:
        return "Please specify an export path", final_mask_state
    
    try:
        # Check file extension
        if not output_path.lower().endswith(('.tif', '.tiff')):
            output_path += '.tif'  # Automatically add tif extension
        
        # Check if final_mask exists
        if final_mask_state is None or np.max(final_mask_state) == 0:
            return "No instance masks to export. Please save at least one instance first", final_mask_state
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert final_mask_state back to original image size
        if original_image_size is not None and padding_info is not None:
            # First, adjust mask from current size (possibly 1024x1024) to pre-processing size (including padding size)
            padded_width = padding_info["original_width"] + padding_info["pad_width_left"] + padding_info["pad_width_right"]
            padded_height = padding_info["original_height"] + padding_info["pad_height_top"] + padding_info["pad_height_bottom"]
            
            # First, adjust mask to include padding size
            mask_pil = Image.fromarray(final_mask_state.astype(np.uint16))
            mask_pil = mask_pil.resize((padded_width, padded_height), Image.NEAREST)
            mask_padded = np.array(mask_pil)
            
            # Then apply reverse padding transformation
            original_mask = reverse_padding(mask_padded, padding_info)
            final_mask_to_save = original_mask
        else:
            # If no original size or padding information, directly use final_mask_state
            final_mask_to_save = final_mask_state
        
        # Use tifffile to save uint16 format mask
        tifffile.imwrite(output_path, final_mask_to_save.astype(np.uint16))
        
        return f"Mask successfully exported to {output_path}, with {np.max(final_mask_state)} instances, size {final_mask_to_save.shape}", final_mask_state
        
    except Exception as e:
        import traceback
        error_msg = f"Error exporting mask: {str(e)}\n{traceback.format_exc()}"
        return error_msg, final_mask_state

# Clear points and current mask
def clear_points(orig_img, final_mask_state=None):
    """Clear all annotations and restore original image"""
    global current_mask
    
    print(f"Clear points function called, original image type: {type(orig_img)}")
    
    if orig_img is None:
        print("Original image is empty, cannot restore")
        return [], [], orig_img, final_mask_state
    
    # Reset current mask
    current_mask = None
        
    # Ensure using copy
    if hasattr(orig_img, "copy"):
        clean_img = orig_img.copy()
        print(f"Restore original image (copy): {type(clean_img)}")
    else:
        clean_img = orig_img
        print(f"Restore original image (original): {type(clean_img)}")
    
    # If final_mask, display saved instances
    if final_mask_state is not None and np.max(final_mask_state) > 0:
        clean_img = visualize_masks(clean_img, None, final_mask_state, [], [])
        
    return [], [], clean_img, final_mask_state

# Reset all instances
def reset_instances(orig_img):
    """Reset all instances, clear final_mask"""
    global instance_count, current_mask
    
    if orig_img is None:
        return [], [], orig_img, None, "No image to reset"
    
    # Reset instance counter and current mask
    instance_count = 0
    current_mask = None
    
    # Ensure using copy
    if hasattr(orig_img, "copy"):
        clean_img = orig_img.copy()
    else:
        clean_img = orig_img
    
    return [], [], clean_img, None, "Reset all instances"

# Gradio interface
def create_ui():
    with gr.Blocks(title="uLLSAM Interactive Segmentation", theme=gr.themes.Soft(), analytics_enabled=False) as demo:
        # Set to English through environment variable
        gr.Markdown("# 🔬 uLLSAM Interactive Segmentation 🔬")        
        # Main image display area - left-right layout
        with gr.Row():
            # Left image input area (occupies 45% space, leave 5% as white space)
            with gr.Column(scale=45):
                image_input = gr.Image(
                    label="Input Image", 
                    type="pil", 
                    height=720,
                    width=720,
                    sources=["upload", "webcam", "clipboard"],
                )
            
            # Middle explanation area (occupies 10% space)
            with gr.Column(scale=10):
                gr.Markdown("""
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
            
            # Right result display area (occupies 45% space, leave 5% as white space)
            with gr.Column(scale=45):
                image_display = gr.Image(
                    label="Segmentation Results", 
                    interactive=False,
                    height=720,
                    width=720,
                )
        
        # Control panel area
        with gr.Row():
            # Left control panel - Model selection part
            with gr.Column(scale=1):
                with gr.Column(elem_id="model-control"):
                    gr.Markdown("### Model Parameters")
                    
                    model_dropdown = gr.Dropdown(
                        choices=["uLLSAM-B-ALL-epoch24", "uLLSAM-B-EM-epoch12", "uLLSAM-V1-EM-epoch12", 
                                "uLLSAM-B-LM-epoch12", "uLLSAM-V1-LM-epoch12"],
                        label="Choose a model",
                        value="uLLSAM-B-ALL-epoch24"
                    )
                    
                    with gr.Row():
                        device_radio = gr.Radio(
                            choices=["CUDA", "CPU"],
                            value="CUDA" if torch.cuda.is_available() else "CPU",
                            label="Device"
                        )
                        
                        dtype_radio = gr.Radio(
                            choices=["bfloat16", "float16", "float32"],
                            value="bfloat16",
                            label="quantization"
                        )
                    
                    load_button = gr.Button("load model")
                    model_status = gr.Textbox(label="model states", interactive=False)
            
            # Right control panel - Interactive control part
            with gr.Column(scale=1):
                with gr.Column(elem_id="interaction-control"):
                    gr.Markdown("### Interactive control")
                    
                    point_type = gr.Radio(
                        choices=["Positive", "Negative"],
                        value="Positive",
                        label="point type"
                    )
                    
                    with gr.Row():
                        clear_button = gr.Button("Clear all points")
                        generate_mask_button = gr.Button("Generate mask")
                        save_instance_button = gr.Button("Save instance")
                        
                    with gr.Row():
                        reset_button = gr.Button("Reset all instances")
                        mask_status = gr.Textbox(label="State information", interactive=False)
                        
                    with gr.Row():
                        export_path = gr.Textbox(
                            placeholder="/path/to/save/mask.tif",
                            label="Output path"
                        )
                        export_button = gr.Button("Export mask")
                    
                    with gr.Row():
                        caption_prompt = gr.Textbox(
                            placeholder="Describe the image in detail",
                            label="Caption prompt"
                        )
                        generate_caption_button = gr.Button("Generate caption")
        
        # Add output dialog
        with gr.Row():
            caption_output = gr.Textbox(
                label="Generated Caption",
                interactive=False,
                lines=10
            )
        
        # Hidden state variables
        points_data = gr.State([])
        labels_data = gr.State([])
        original_image_state = gr.State(None)
        final_mask_state = gr.State(None)  # State variable to store instance mask
        
        # Event binding
        load_button.click(
            fn=load_model,
            inputs=[model_dropdown, device_radio, dtype_radio],
            outputs=[model_status]
        )
        
        # Add image click event processing
        def add_point_by_click(img, evt: SelectData, point_choice, points, labels, final_mask):
            """Add point by clicking image"""
            global original_image_size, current_mask
            
            if img is None:
                print("Image is empty when clicked")
                return img, points, labels
            
            try:
                # Get clicked position coordinates
                x, y = evt.index
                
                # Add new point to list
                new_points = points + [[float(x), float(y)]]
                new_labels = labels + [1 if point_choice == "Positive" else 0]
                
                print(f"Add new point: ({x}, {y}), label: {1 if point_choice == 'Positive' else 0}")
                
                # Get actual image
                if isinstance(img, dict) and "image" in img:
                    img_pil = img["image"].copy()
                elif hasattr(img, "copy"):
                    img_pil = img.copy()
                else:
                    # Try to convert to PIL image
                    try:
                        img_pil = Image.fromarray(np.array(img)).copy()
                    except:
                        print(f"Warning: Unable to process image type: {type(img)}")
                        img_pil = img
                
                # Reset current mask because new point is added
                current_mask = None
                
                # Draw all points and saved instances
                overlay_image = visualize_masks(img_pil, None, final_mask, new_points, new_labels)
                
                return overlay_image, new_points, new_labels
            
            except Exception as e:
                print(f"Error processing click: {str(e)}")
                import traceback
                traceback.print_exc()
                return img, points, labels
        
        # Add image click event processing
        image_input.select(
            fn=add_point_by_click,
            inputs=[image_input, point_type, points_data, labels_data, final_mask_state],
            outputs=[image_input, points_data, labels_data]
        )
        
        # Image upload processing
        def on_image_upload(img):
            """Process image upload, save original image and clear points, initialize final_mask"""
            global instance_count, current_mask, padding_info
            
            if img is None:
                print("Uploaded image is empty")
                return None, [], None, None
            
            print(f"Image uploaded: type={type(img)}")
            
            # Use new processing function to process uploaded image
            processed_img = process_uploaded_image(img)
            if processed_img is None:
                print("Image processing failed")
                return None, [], None, None
                
            print(f"Processed image: type={type(processed_img)}, size={processed_img.size if hasattr(processed_img, 'size') else 'unknown'}")
            print(f"Saved padding information: {padding_info}")
            
            # Reset instance counter and current mask
            instance_count = 0
            current_mask = None
            
            # Clear points data and final_mask and return
            return processed_img, [], processed_img, None
        
        # Only save original image and clear points when image is uploaded, and display processed image
        image_input.upload(
            fn=on_image_upload,
            inputs=[image_input],
            outputs=[original_image_state, points_data, image_input, final_mask_state]
        )
        
        # Generate mask
        generate_mask_button.click(
            fn=process_points_and_generate_mask,
            inputs=[image_input, points_data, labels_data, image_input, final_mask_state],
            outputs=[image_display, mask_status, final_mask_state]
        )
        
        # Save instance - Modify to update Input Image and Segmentation Results
        save_instance_button.click(
            fn=save_instance,
            inputs=[image_input, final_mask_state, points_data, labels_data],
            outputs=[image_input, image_display, mask_status, final_mask_state, points_data, labels_data]
        )
        
        # Export mask
        export_button.click(
            fn=export_mask,
            inputs=[image_input, points_data, labels_data, export_path, final_mask_state],
            outputs=[mask_status, final_mask_state]
        )
        
        # Generate caption
        generate_caption_button.click(
            fn=generate_caption,
            inputs=[image_input, caption_prompt],
            outputs=[caption_output]
        )
        
        # Clear points function
        clear_button.click(
            fn=clear_points,
            inputs=[original_image_state, final_mask_state],
            outputs=[points_data, labels_data, image_input, final_mask_state]
        )
        
        # Reset all instances
        reset_button.click(
            fn=reset_instances,
            inputs=[original_image_state],
            outputs=[points_data, labels_data, image_input, final_mask_state, mask_status]
        )
    
    return demo

if __name__ == "__main__":
    # Delete environment variable setting, use launch method's localization parameter
    
    demo = create_ui()
    # demo.launch(
    #     share=True, 
    #     server_port=8081
    # ) 
    
    # Print server startup information
    print("Starting Gradio server...")
    
    # Try to use default port to start
    try:
        demo.launch(
            share=True,
            server_name="0.0.0.0",  # Allow external access
            debug=True,  # Enable debug mode
            server_port=9996,
        )
    except Exception as e:
        print(f"Server startup error: {e}")
        print("Trying backup port...")
        # If default port fails, try to explicitly specify port
        demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            debug=True,
        ) 


