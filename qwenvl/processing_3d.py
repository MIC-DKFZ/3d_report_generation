from typing import List, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput
from transformers.video_utils import VideoInput
from transformers.processing_utils import Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput

from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor, Qwen2_5_VLProcessorKwargs

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

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, reduce_block = [2, 2, 2], **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.reduce_block = reduce_block

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

        if vision_embeddings:
            video_grid_thw = torch.tensor([v_e.shape[-3:] for v_e in vision_embeddings]).int()
            list_of_vision_embeddings = [torch.tensor(rearrange(v_e, 'b c h w d -> (b h w d) c')) for v_e in vision_embeddings]
            # TODO fix the following padding (makes no sense right now (but works when all images have same size, otherwise crash with video_grid_thw above later))
            videos_input_embeddings = torch.nn.utils.rnn.pad_sequence(
               list_of_vision_embeddings, batch_first=True, padding_value=0).transpose(2, 1)
            videos_inputs = {
                "pixel_values_videos": videos_input_embeddings,
                "video_grid_thw": video_grid_thw
            }

        if not isinstance(text, list):
            text = [text]

        text = text.copy()  # below lines change text in-place

        # I just don't know why here is an image_pad token in the text (which needs
        # to be replaced in the next lines of code.
        for i in range(0, len(text)):
            if self.image_token in text[i]:
                text[i] = text[i].replace(self.image_token, self.video_token, 1)

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