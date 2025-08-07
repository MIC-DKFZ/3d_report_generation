from typing import List, Union

from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ImageInput, VideoInput
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin, Unpack
from transformers.tokenization_utils_base import PreTokenizedInput, TextInput
from transformers.utils import logging
from transformers import Qwen2VLProcessor

import torch
import numpy as np

class Qwen2VLProcessorKwargs(ProcessingKwargs, total=False):
    _defaults = {
        "text_kwargs": {
            "padding": False,
        },
    }


class Qwen2VLProcessor3D(Qwen2VLProcessor):
    r"""
    Constructs a Qwen2-VL processor which wraps a Qwen2-VL image processor and a Qwen2 tokenizer into a single processor.
    [`Qwen2VLProcessor`] offers all the functionalities of [`Qwen2VLImageProcessor`] and [`Qwen2TokenizerFast`]. See the
    [`~Qwen2VLProcessor.__call__`] and [`~Qwen2VLProcessor.decode`] for more information.
    Args:
        image_processor ([`Qwen2VLImageProcessor`], *optional*):
            The image processor is a required input.
        tokenizer ([`Qwen2TokenizerFast`], *optional*):
            The tokenizer is a required input.
        chat_template (`str`, *optional*): A Jinja template which will be used to convert lists of messages
            in a chat into a tokenizable string.
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["chat_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("Qwen2Tokenizer", "Qwen2TokenizerFast")

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, reduce_block = [2, 2, 2], **kwargs):
        self.image_token = "<|image_pad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.video_token = "<|video_pad|>" if not hasattr(tokenizer, "video_token") else tokenizer.video_token
        super().__init__(image_processor, tokenizer, chat_template=chat_template)

        self.reduce_block = reduce_block # this is fixed and parameter that will be changed during training

    def __call__(
        self,
        images: ImageInput = None,
        vision_embeddings: List[torch.Tensor] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        videos: VideoInput = None,
        **kwargs: Unpack[Qwen2VLProcessorKwargs],
    ) -> BatchFeature:
        """
        Main method to prepare for the model one or several sequences(s) and image(s). This method forwards the `text`
        and `kwargs` arguments to Qwen2TokenizerFast's [`~Qwen2TokenizerFast.__call__`] if `text` is not `None` to encode
        the text. To prepare the vision inputs, this method forwards the `vision_infos` and `kwrags` arguments to
        Qwen2VLImageProcessor's [`~Qwen2VLImageProcessor.__call__`] if `vision_infos` is not `None`.

        Args:
            images (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`, `List[torch.Tensor]`):
                The image or batch of images to be prepared. Each image can be a PIL image, NumPy array or PyTorch
                tensor. Both channels-first and channels-last formats are supported.
            text (`str`, `List[str]`, `List[List[str]]`):
                The sequence or batch of sequences to be encoded. Each sequence can be a string or a list of strings
                (pretokenized string). If the sequences are provided as list of strings (pretokenized), you must set
                `is_split_into_words=True` (to lift the ambiguity with a batch of sequences).
            videos (`np.ndarray`, `torch.Tensor`, `List[np.ndarray]`, `List[torch.Tensor]`):
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
        """
        output_kwargs = self._merge_kwargs(
            Qwen2VLProcessorKwargs,
            tokenizer_init_kwargs=self.tokenizer.init_kwargs,
            **kwargs,
        )

        if images is not None:
            image_inputs = self.image_processor(images=images, videos=None, **output_kwargs["images_kwargs"])
            image_grid_thw = image_inputs["image_grid_thw"]
        else:
            image_inputs = {}
            image_grid_thw = None


        if videos is not None:
            videos_inputs = self.image_processor(images=None, videos=videos, **output_kwargs["videos_kwargs"])
            video_grid_thw = videos_inputs["video_grid_thw"]
        else:
            videos_inputs = {}
            video_grid_thw = None


        if vision_embeddings:
            # Put the shape of each vision embedding in tensor
            video_grid_thw = torch.tensor([v_e.shape[-3:] for v_e in vision_embeddings]).int()
            # Flatten vision embeddings to [B, C, H*W*D] delete the batch dimension and transpose to [H*W*D, C]
            list_of_vision_embeddings = [
                torch.tensor(v_e).view(v_e.shape[0], v_e.shape[1], -1).squeeze(0).transpose(1, 0)
                for v_e in vision_embeddings]
            len_of_vision_embeddings = [v_e.size(0) for v_e in list_of_vision_embeddings]
            # Pad the vision embeddings to the same length and stack them in a tensor,
            # then transpose to [B, C, H*W*D]
            videos_input_embeddings = torch.nn.utils.rnn.pad_sequence(
                list_of_vision_embeddings, batch_first=True, padding_value=0).transpose(2, 1)
            # # Create padding mask for the vision embeddings for later reconstruction
            # padding_mask = torch.nn.utils.rnn.pad_sequence([torch.ones(l) for l in len_of_vision_embeddings], batch_first=True, padding_value=0)

            videos_inputs = {
                "pixel_values_videos": videos_input_embeddings,
                # TODO: do we need this self.reduce_block. Further maybe we need ceil instead of // (floor).
                # "video_grid_thw": torch.tensor((video_grid_thw // torch.tensor(self.reduce_block)).tolist())
                "video_grid_thw": torch.tensor((video_grid_thw).tolist()),
                # "padding_mask_vision_embeddings": padding_mask
            }

        else:
            videos_inputs = {}
            video_grid_thw = None

        if not isinstance(text, list):
            text = [text]

        #Dirty solution. TODO use own text input so that we do not need to replace video token
        for i in range(0, len(text)):
            if self.image_token in text[i]:
                text[i] = text[i].replace(self.image_token, self.video_token, 1)

        if image_grid_thw is not None:
            merge_length = self.image_processor.merge_size**2
            index = 0
            for i in range(len(text)):
                while self.image_token in text[i]:
                    text[i] = text[i].replace(
                        self.image_token, "<|placeholder|>" * (image_grid_thw[index].prod() // merge_length), 1
                    )
                    index += 1
                text[i] = text[i].replace("<|placeholder|>", self.image_token)

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

        text_inputs = self.tokenizer(text, **output_kwargs["text_kwargs"])

        return BatchFeature(data={**text_inputs, **image_inputs, **videos_inputs})