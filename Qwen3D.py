from transformers import BitsAndBytesConfig
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

import gc
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from torch import nn

from transformers.activations import QuickGELUActivation

from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLCausalLMOutputWithPast, QWEN2_VL_INPUTS_DOCSTRING, _CONFIG_FOR_DOC


from transformers.models.qwen2_vl.modeling_qwen2_vl import VisionMlp
from transformers.models.qwen2_vl.modeling_qwen2_vl import ACT2FN



'''

    def __init__(self, embedding_dim: int, embed_dim: int):
        super().__init__()
        self.adapter = VisionMlp(
            dim=embedding_dim,  # 864
            hidden_dim=embed_dim,  # 1536 
            hidden_act="quick_gelu"
        )
'''

class VisionAdapter(nn.Module):
    def __init__(self, embedding_dim: int, embed_dim: int):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim, embed_dim),  # 864 -> 1536
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),  # 1536 -> 1536
        )
        
    def forward(self, x):
        # x shape: (batch_size, 864, sequence_length)
        # Need to output: (sequence_length, embed_dim)
        
        # Transpose to get sequence_length as first dimension
        x = x.transpose(1, 2)  # -> (batch_size, sequence_length, 864)
        
        # Reshape to process all sequences at once
        batch_size, seq_len, emb_dim = x.shape
        x = x.reshape(-1, emb_dim)  # -> (batch_size * sequence_length, 864)
        
        # Apply adapter
        x = self.adapter(x)  # -> (batch_size * sequence_length, 1536)
        
        # Reshape back
        x = x.reshape(batch_size * seq_len, -1)  # -> (batch_size * sequence_length, 1536)
        
        return x

    
class VisionAdapterReduceBlocks(nn.Module):
    def __init__(self, embedding_dim: int, embed_dim: int, reduce_block: Tuple[int, int, int] = [2, 2, 2]):
        super().__init__()
        self.reduce_block = reduce_block
        self.adapter = nn.Sequential(
            nn.Linear(embedding_dim * reduce_block[0] * reduce_block[1] * reduce_block[2], embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        
    def forward(self, x):        
        # Convert input to float first
        x = x.to(torch.bfloat16)
        
        # Input shape is (batch_size, embedding_dim, sequence_length)
        batch_size, embedding_dim, seq_length = x.shape
        
        # E.g for 13824 video tokens, reshape to 24x24x24
        cube_size = int(seq_length ** (1/3)) + 1
        x = x.reshape(batch_size, embedding_dim, cube_size, cube_size, cube_size)
        
        # Use unfold to create reduce_block
        x = x.unfold(2, self.reduce_block[0], self.reduce_block[0])
        x = x.unfold(3, self.reduce_block[1], self.reduce_block[1])
        x = x.unfold(4, self.reduce_block[2], self.reduce_block[2])
       
        # Reshape to combine all reduce_block
        x = x.permute(0, 2, 3, 4, 1, 5, 6, 7)  # Move embedding_dim after spatial dims
        x = x.reshape(batch_size, -1, embedding_dim * self.reduce_block[0] * self.reduce_block[1] * self.reduce_block[2])
       
        # Process through adapter
        x = x.reshape(-1, embedding_dim * self.reduce_block[0] * self.reduce_block[1] * self.reduce_block[2])
        x = self.adapter(x)    

        return x

from torch.nn import CrossEntropyLoss
from transformers.utils import add_start_docstrings_to_model_forward, replace_return_docstrings

class Qwen2VLForConditionalGeneration3D(Qwen2VLForConditionalGeneration):
    def __init__(self, config, reduce_block = [2, 2, 2]):
            super().__init__(config)
            
            self.vision_adapter = VisionAdapterReduceBlocks(embedding_dim=864, embed_dim=1536, reduce_block=reduce_block)

    @add_start_docstrings_to_model_forward(QWEN2_VL_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Qwen2VLCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, Qwen2VLCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        >>> model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
        >>> processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

        >>> messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        >>> inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "The image shows a street scene with a red stop sign in the foreground. In the background, there is a large red gate with Chinese characters ..."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.visual.get_dtype())
                image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )
                image_mask = (
                    (input_ids == self.config.image_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                #pixel_values_videos = pixel_values_videos.type(self.visual.get_dtype())
                #video_embeds = self.vision_adapter(pixel_values_videos) # The only line that has changed so far (and the added adapter above)
                pixel_values_videos = pixel_values_videos.to(self.vision_adapter.adapter[0].weight.dtype)
                video_embeds = self.vision_adapter(pixel_values_videos)
                
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )
                video_mask = (
                    (input_ids == self.config.video_token_id)
                    .unsqueeze(-1)
                    .expand_as(inputs_embeds)
                    .to(inputs_embeds.device)
                )
                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        # if we get 4D attention mask we cannot calculate rope deltas anymore. TODO @raushan fixme
        if position_ids is None and (attention_mask is None or attention_mask.ndim == 2):
            # calculate RoPE index once per generation in the pre-fill stage only
            if (
                (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            ):
                position_ids, rope_deltas = self.get_rope_index(
                    input_ids, image_grid_thw, video_grid_thw, attention_mask
                )
                self.rope_deltas = rope_deltas
            # then use the prev pre-calculated rope-deltas to get the correct position ids
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:  # otherwise `deltas` is an int `0`
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                    delta = delta.to(position_ids.device)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs = self.model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return Qwen2VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )
