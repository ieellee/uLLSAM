# --------------------------------------------------------
# InternVL-SAM
# FUDAN UNIVERSITY 
# Author: Manyu Li, 2025.03.28
# --------------------------------------------------------

import warnings
from typing import List, Optional, Tuple, Union
import math

import torch.utils.checkpoint
import transformers
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from .configuration_internvl_chat import InternVLChatConfig
from .conversation import get_conv_template
from .modeling_internlm2 import InternLM2ForCausalLM

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version
    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLSAMModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _supports_flash_attn_2 = True
    _no_split_modules = ['LlamaDecoderLayer', 'InternLM2DecoderLayer']

    def __init__(self, config: InternVLChatConfig, vision_model=None, prompt_encoder=None, mask_decoder=None, language_model=None, use_flash_attn=True):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        
        # imgsz = 1024 x 1024, patch_size = 16
        self.patch_size = 16
        self.select_layer = config.select_layer
        self.template = config.template
        
        # B x 256 x 64 x 64, 64 x 64 = 4096 vision tokens 
        self.num_image_token = 64 * 64 * (config.downsample_ratio ** 2)
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        
        # pass sam encoder
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            pass
            # raise ValueError("SAM vision_model must be provided")
        if prompt_encoder is not None:
            self.prompt_encoder = prompt_encoder
        else:
            pass
            # raise ValueError("SAM prompt_encoder must be provided")
        if mask_decoder is not None: 
            self.mask_decoder = mask_decoder
        else:
            pass
            # raise ValueError("SAM mask_decoder must be provided")
        # llm use InternLM2ForCausalLM
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(f'{config.llm_config.architectures[0]} is not implemented.')

        # sam hidden size is 256 ViT-B-16, llm hidden size is 2048
        sam_hidden_size = 256
        llm_hidden_size = config.llm_config.hidden_size # 2048

        # vision projector for align vision tokens and language tokens
        self.mlp1 = nn.Sequential(
            nn.LayerNorm(sam_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(sam_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        # tokens projector for convert to SAM dense prompt input 
        self.mlp2 = nn.Sequential(
            nn.LayerNorm(llm_hidden_size),
            nn.Linear(llm_hidden_size, sam_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.GELU(),
            nn.Linear(sam_hidden_size * int(1 / self.downsample_ratio) ** 2, sam_hidden_size * int(1 / self.downsample_ratio) ** 2)
        )
        # <|IMG_CONTEXT|> token id
        self.img_context_token_id = 92546
        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message

    def forward(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            image_flags: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            img_context_token_id: Optional[int] = None, 
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone() # 1 x max_length x 2048

        vit_embeds, image_embeddings = self.extract_feature(pixel_values) # 1 x 1024 x 2048, 1 x 256 x 64 x 64
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape # 1 x 1024 x 2048
        input_embeds = input_embeds.reshape(B * N, C)

        # if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        #     print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

        input_ids = input_ids.reshape(B * N) # [max length]
        selected = (input_ids == self.img_context_token_id)
        try:
            flat_vit_embeds = vit_embeds.reshape(-1, C) # 1024, 2048
            n_selected = selected.sum() # 1024
            if n_selected <= flat_vit_embeds.shape[0]:
                input_embeds[selected] = input_embeds[selected] * 0.0 + flat_vit_embeds[:n_selected]
            else:
                repeats = (n_selected + flat_vit_embeds.shape[0] - 1) // flat_vit_embeds.shape[0]
                repeated_embeds = flat_vit_embeds.repeat(repeats, 1)
                input_embeds[selected] = input_embeds[selected] * 0.0 + repeated_embeds[:n_selected]
        except Exception as e:
            print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                  f'vit_embeds.shape={vit_embeds.shape}')
            n_token = selected.sum()
            flat_vit_embeds = vit_embeds.reshape(-1, C)
            if n_token <= flat_vit_embeds.shape[0]:
                input_embeds[selected] = input_embeds[selected] * 0.0 + flat_vit_embeds[:n_token]
            else:
                repeats = (n_token + flat_vit_embeds.shape[0] - 1) // flat_vit_embeds.shape[0]
                repeated_embeds = flat_vit_embeds.repeat(repeats, 1)
                input_embeds[selected] = input_embeds[selected] * 0.0 + repeated_embeds[:n_token]

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        '''
            if outputs.hidden_states is not None, it means we need to use the 
            last output hidden state to enhance our SAM segmentation ability.
        '''
        if outputs.hidden_states is not None:
            image_token_mask = (input_ids == self.img_context_token_id)
            image_token_indices = torch.nonzero(image_token_mask, as_tuple=True)[0]  # obtain image token index
            if torch.any(image_token_mask):
                # hidden state do not need to shift
                start_idx = torch.min(image_token_indices)
                end_idx = torch.max(image_token_indices) + 1
                last_hidden_state = outputs.hidden_states[-1][:, start_idx:end_idx, :]
                last_hidden_state = self.text_aware_dense_feature(last_hidden_state)
            else:
                raise ValueError("Can not find vision token!")
            # last_hidden_state = outputs.hidden_states[-1][:, :1024, :]
            # last_hidden_state = self.text_aware_dense_feature(last_hidden_state) # B x 256 x 64 x 64
            ret = CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=last_hidden_state, # return last_hidden_state
                attentions=outputs.attentions
            )
            setattr(ret, "image_embeddings", image_embeddings)
            return ret

        ret = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )
        setattr(ret, "image_embeddings", image_embeddings)
        return ret

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, h, w, c = x.size()
        # N, H, W, C --> N, H, W * scale, C // scale
        x = x.reshape(n, h, int(w * scale_factor), int(c / scale_factor))
        # N, H, W * scale, C // scale --> N, W * scale, H, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, W * scale, H, C // scale --> N, W * scale, H * scale, C // (scale ** 2)
        x = x.reshape(n, int(w * scale_factor), int(h * scale_factor),
                   int(c / (scale_factor * scale_factor)))
        if self.ps_version == 'v1':
            warnings.warn("In ps_version 'v1', the height and width have not been swapped back, "
                          'which results in a transposed image.')
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):

        with torch.no_grad():
            vit_features = self.vision_model(pixel_values) # B x 256 x 64 x 64
        features = vit_features.permute(0, 2, 3, 1) # B x 64 x 64 x 256
        features = self.pixel_shuffle(features, scale_factor=self.downsample_ratio) # B x 32 x 32 x 1024
        features = features.reshape(features.shape[0], -1, features.shape[-1]) # 1 x 1024 x 1024
        features = self.mlp1(features) # 1 x 1024 x 2048
        
        return features, vit_features

    def text_aware_dense_feature(self, features):
        # input: features [1, 1024, 2048]
        features = self.mlp2(features)  # [1, 1024, 1024]
        features = features.reshape(features.shape[0], 
                                int(math.sqrt(features.shape[1])), 
                                int(math.sqrt(features.shape[1])), 
                                features.shape[2])  # [1, 32, 32, 1024]
        
        if self.ps_version != 'v1':
            features = features.permute(0, 2, 1, 3).contiguous() 
        
        n, h, w, c = features.size()
        features = features.reshape(n, h, int(w/self.downsample_ratio), int(c*self.downsample_ratio))
        features = features.permute(0, 2, 1, 3).contiguous()
        features = features.reshape(n, int(w/self.downsample_ratio), int(h/self.downsample_ratio), int(c*(self.downsample_ratio*self.downsample_ratio)))
        features = features.permute(0, 3, 1, 2).contiguous()  # [1, 256, 64, 64]
        
        return features

    def chat(self, tokenizer, pixel_values, question, generation_config, history=None, return_history=False,
             num_patches_list=None, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>', IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
             verbose=False):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = question + '\n<image>'

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip()) # '<|im_end|>'

        history = [] if history is None else history
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * int(self.num_image_token) * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        generation_config["eos_token_id"] = eos_token_id # modified
        
        # whether to obtain hidden_states
        output_hidden_states = generation_config.pop("output_hidden_states", False)
        
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **generation_config
        )
        
        if output_hidden_states and hasattr(generation_output, "hidden_states") and generation_output.hidden_states is not None:
            self.masks = generation_output.hidden_states
        
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
            if verbose:
                print(query_to_print, response)
            return response

    def batch_chat(self, tokenizer, pixel_values, questions, generation_config, num_patches_list=None,
                   history=None, return_history=False, IMG_START_TOKEN='<img>', IMG_END_TOKEN='</img>',
                   IMG_CONTEXT_TOKEN='<IMG_CONTEXT>', verbose=False, image_counts=None):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print('Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.')

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids'].to(self.device)
        attention_mask = model_inputs['attention_mask'].to(self.device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        
        output_hidden_states = generation_config.pop("output_hidden_states", False)
        
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            **generation_config
        )
        
        if output_hidden_states and hasattr(generation_output, "hidden_states") and generation_output.hidden_states is not None:
            self.masks = generation_output.hidden_states
        
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [response.split(template.sep.strip())[0].strip() for response in responses]
        return responses

    @torch.no_grad()
    def generate(
            self,
            pixel_values: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            generation_config: Optional[GenerationConfig] = None,
            output_hidden_states: Optional[bool] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds, _ = self.extract_feature(pixel_values) # 1 x 1024 x 2048
            input_embeds = self.language_model.get_input_embeddings()(input_ids) # 1 x 1081 x 2048
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C) # 1081 x 2048

            input_ids = input_ids.reshape(B * N) # 1081
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            # print(vit_embeds)
            flat_vit_embeds = vit_embeds.reshape(-1, C).to(input_embeds.device)
            n_selected = selected.sum()
            if n_selected <= flat_vit_embeds.shape[0]:
                input_embeds[selected] = flat_vit_embeds[:n_selected]
            else:
                repeats = (n_selected + flat_vit_embeds.shape[0] - 1) // flat_vit_embeds.shape[0]
                repeated_embeds = flat_vit_embeds.repeat(repeats, 1)
                input_embeds[selected] = repeated_embeds[:n_selected]

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    

if __name__ == "__main__":
    checkpoint_path = "path/to/sam_checkpoint"
    from segment_anything import sam_model_registry
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    vision_encoder = sam.image_encoder
    config = InternVLChatConfig.from_pretrained("path/to/InternVL2_5-2B", trust_remote_code=True)
    model = InternVLSAMModel(config, vision_model=vision_encoder, language_model=None)
    print("OK")