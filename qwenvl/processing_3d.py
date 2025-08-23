from typing import List, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.video_utils import VideoInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLProcessorKwargs
import torch.nn.functional as F
import random
from einops import rearrange
import torch
import numpy as np


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

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, reduce_block = [2, 2, 2], train_vision_encoder=False, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.reduce_block = reduce_block
        self.train_vision_encoder = train_vision_encoder

    def __call__(
        self,
        images: ImageInput = None,
        vision_embeddings: List[torch.Tensor] = None,
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

        if vision_embeddings and not self.train_vision_encoder:
            video_grid_thw = torch.tensor([v_e.shape[-3:] for v_e in vision_embeddings]).int()
            list_of_vision_embeddings = [torch.tensor(rearrange(v_e, 'b c h w d -> (b h w d) c')) for v_e in vision_embeddings]
            # TODO fix the following padding (makes no sense right now (but works when all images have same size, otherwise crash with video_grid_thw above later))
            videos_input_embeddings = torch.nn.utils.rnn.pad_sequence(
               list_of_vision_embeddings, batch_first=True, padding_value=0).transpose(2, 1)
            videos_inputs = {
                "pixel_values_videos": videos_input_embeddings,
                "video_grid_thw": video_grid_thw
            }

        elif self.train_vision_encoder:
            patch_size = (192, 192, 192)
            block_size = (8, 8, 8)
            
            # Pre-allocate numpy array for all processed patches (more memory efficient than list of tensors)
            num_images = len(vision_embeddings)
            processed_patches = np.zeros((num_images, *patch_size), dtype=np.float32)
            
            for i, image in enumerate(vision_embeddings):
                # Work with numpy array directly to avoid intermediate tensor creation
                if not isinstance(image, np.ndarray):
                    image_array = np.array(image, dtype=np.float32)
                else:
                    # Ensure float32 dtype without unnecessary copy
                    image_array = image if image.dtype == np.float32 else image.astype(np.float32)
                
                # Get image dimensions (assuming format: H, W, D)
                h, w, d = image_array.shape
                
                # Calculate padding
                pad_h = max(0, patch_size[0] - h)
                pad_w = max(0, patch_size[1] - w) 
                pad_d = max(0, patch_size[2] - d)
                
                if pad_h > 0 or pad_w > 0 or pad_d > 0:
                    # Split padding evenly on both sides to center the image
                    pad_h_before = pad_h // 2
                    pad_h_after = pad_h - pad_h_before
                    
                    pad_w_before = pad_w // 2
                    pad_w_after = pad_w - pad_w_before
                    
                    pad_d_before = pad_d // 2
                    pad_d_after = pad_d - pad_d_before
                    
                    # Use numpy padding (more memory efficient than torch F.pad for this case)
                    padding = ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after), (pad_d_before, pad_d_after))
                    image_array = np.pad(image_array, padding, mode='constant', constant_values=0)
                
                # Random patch extraction
                max_starts = [max(0, image_array.shape[i] - patch_size[i]) for i in range(3)]
                start_h, start_w, start_d = [int(np.clip(np.random.normal(m/2, m*0.1), 0, m)) for m in max_starts]
                
                # Extract patch directly into pre-allocated array (no intermediate storage)
                processed_patches[i] = image_array[
                    start_h:start_h + patch_size[0],
                    start_w:start_w + patch_size[1], 
                    start_d:start_d + patch_size[2]
                ]
                
                # Explicitly clean up intermediate array
                del image_array
            
            # Convert to tensor only at the very end (single allocation instead of multiple)
            videos_input_embeddings = torch.from_numpy(processed_patches)
            video_grid_thw = torch.tensor([[24, 24, 24] for _ in range(num_images)])
            
            videos_inputs = {
                "pixel_values_videos": videos_input_embeddings,
                "video_grid_thw": video_grid_thw
            }

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        if video_grid_thw is not None:
            merge_length = np.prod(self.reduce_block)
            index = 0
            for i in range(len(text)):
                while self.video_token in text[i]:
                    text[i] = text[i].replace(
                        # TODO: In the adapter we use unfold to create merger patches. If the image size is not divisible by the reduce block the last row/column will be dropped.
                        self.video_token, "<|placeholder|>" * ((video_grid_thw[index] - video_grid_thw[index] % torch.as_tensor(self.reduce_block)).prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.video_token)

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