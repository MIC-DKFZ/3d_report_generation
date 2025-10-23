import random
from typing import List, Union

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import (
    Qwen2_5_VLProcessor,
    Qwen2_5_VLProcessorKwargs,
)
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.video_utils import VideoInput

from utils.vision_input_type import VisionInputType


class QwenVLProcessor3D(Qwen2_5_VLProcessor):
    r"""
    Constructs a Qwen2.5-VL processor which wraps a Qwen2.5-VL image processor and a Qwen2 tokenizer into a single processor.
    [`Qwen2_5_VLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2_5_VLProcessor.__call__`] and [`~Qwen2_5_VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        video_processor ([`Qwen2_5_VLVideoProcessor`], *optional*):
            The video processor is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """
    attributes = ["image_processor", "tokenizer"]#, "video_processor"]
    
    image_processor_class = "AutoImageProcessor"
    #video_processor_class = "AutoVideoProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, reduce_block=[2, 2, 2], vision_input_type=VisionInputType.EMBEDDING, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.reduce_block = reduce_block
        self.vision_input_type = vision_input_type

    def __call__(
        self,
        images: ImageInput = None,
        vision_inputs: List[torch.Tensor] = None,
        text: Union[TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2_5_VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `list[PIL.Image.Image]`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `list[str]`, `list[list[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `list[np.ndarray]`, `list[torch.Tensor]`):
                The image or batch of videos to be prepared. Each video can be a 4D NumPy array or PyTorch
                tensor, or a nested list of 3D frames. Both channels-first and channels-last formats are supported.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.
                - `'jax'`: Return JAX `jnp.ndarray` objects.

        Returns:
            [`BatchFeature`]: A [`BatchFeature`] with the following fields:

            - **input_ids** -- List of token ids to be fed to a model. Returned when `text` is not `None`.
            - **attention_mask** -- List of indices specifying which tokens should be attended to by the model (when
              `return_attention_mask=True` or if *"attention_mask"* is in `self.model_input_names` and if `text` is not
              `None`).
            - **pixel_values** -- Pixel values to be fed to a model. Returned when `images` is not `None`.
            - **pixel_values_videos** -- Pixel values of videos to be fed to a model. Returned when `videos` is not `None`.
            - **image_grid_thw** -- List of image 3D grid in LLM. Returned when `images` is not `None`.
            - **video_grid_thw** -- List of video 3D grid in LLM. Returned when `videos` is not `None`.
            - **second_per_grid_ts** -- List of video seconds per time grid. Returned when `videos` is not `None`.
        """
        output_kwargs = self._merge_kwargs(
            Qwen2_5_VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        image_inputs = videos_inputs = {}
        video_grid_thw = None

        if self.vision_input_type == VisionInputType.EMBEDDING:
            video_grid_thw = torch.tensor(
                np.array([np.array(v_e.shape[-3:]) // np.array(self.reduce_block) for v_e in vision_inputs])
            ).int()
            list_of_vision_embeddings = [torch.tensor(rearrange(v_e, 'b c h w d -> (b h w d) c')) for v_e in vision_inputs]
            # TODO fix the following padding (makes no sense right now (but works when all images have same size, otherwise crash with video_grid_thw above later))
            videos_input_embeddings = torch.nn.utils.rnn.pad_sequence(
               list_of_vision_embeddings, batch_first=True, padding_value=0).transpose(2, 1)
            videos_inputs = {
                "pixel_values_videos": videos_input_embeddings,
                "video_grid_thw": video_grid_thw
            }

        elif self.vision_input_type == VisionInputType.IMAGE:
            # Images are already preprocessed and cropped to final size in dataset!
            num_images = len(vision_inputs)
            processed_images = []
            
            for image_array in vision_inputs:
                # Remove singleton first dimension if present (wieso auch immer brauchen wir das bei blosc2, nicht aber bei nrrd)
                if image_array.ndim == 4 and image_array.shape[0] == 1:
                    image_array = image_array.squeeze(0)
                processed_images.append(image_array)
            
            processed_patches = np.stack(processed_images, axis=0)
            
            # Convert to tensor
            videos_input_images = processed_patches#orch.from_numpy(processed_patches)
            #TODO load block size from VE config
            video_grid_thw = torch.tensor([[videos_input_images[i].shape[0]//(self.reduce_block[0] * 8), 
                                            videos_input_images[i].shape[1]//(self.reduce_block[1] * 8), 
                                            videos_input_images[i].shape[2]//(self.reduce_block[2] * 8)] for i in range(num_images)])
            
            videos_inputs = {
                "pixel_values_videos": videos_input_images,
                "video_grid_thw": video_grid_thw
            }

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if video_grid_thw is not None:
            
            for i, grid_thw in enumerate(video_grid_thw):
                    
                # Calculate valid dimensions (divisible by reduce_block)
                valid_dims = (grid_thw // torch.as_tensor(self.reduce_block)) * torch.as_tensor(self.reduce_block)
                num_tokens = valid_dims.prod().item()
                
                text[i] = text[i].replace(self.video_token, self.video_token * num_tokens, 1)

        #text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        # TODO check why it fails with the above line!
        #text_inputs = self.tokenizer(text, **{"padding":True, "return_tensors": "pt"})

        # Original Qwen25VL stuff
        return_tensors = output_kwargs["text_kwargs"].pop("return_tensors", None)
        return_mm_token_type_ids = output_kwargs["text_kwargs"].pop("return_mm_token_type_ids", None)
        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])
        self._check_special_mm_tokens(text, text_inputs, modalities=["image", "video"])

        if return_mm_token_type_ids:
            array_ids = np.array(text_inputs["input_ids"])
            mm_token_type_ids = np.zeros_like(text_inputs["input_ids"])
            mm_token_type_ids[array_ids == self.image_token_id] = 1
            text_inputs["mm_token_type_ids"] = mm_token_type_ids.tolist()

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs}, tensor_type=return_tensors)