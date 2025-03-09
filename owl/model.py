from torch import nn
from torch.nn import CrossEntropyLoss


from transformers import (
    Qwen2PreTrainedModel,
)


class OwnOwlModel(Qwen2PreTrainedModel):
    def __init__(self, config, language_model, vision_dim=1152, embed_dim=896, tokenizer=None, debug=False):
        super().__init__(config)
        self.language_model = language_model

        self.vision2text_model = nn.Linear(vision_dim, embed_dim)
        self.processor = None

        self.tokenizer = tokenizer

        self.terminators = ["<|im_end|>", "<|endoftext|>"]

        self.debug = debug

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.language_model = decoder

    def get_decoder(self):
        return self.language_model

    def forward_image(self, image_embeds):
        if image_embeds is None:
            return None
        # dtype = self.language_model.model.embed_tokens.weight.dtype
        # with torch.inference_mode():
        #     image_embeds = self.vision_model(pixel_values.to(dtype), output_hidden_states=True).hidden_states[-2]

        if self.vision2text_model is not None:
            image_embeds = self.vision2text_model(image_embeds)
        else:
            pass

        return image_embeds

    def forward(self, image_embeds, input_ids, media_offset, attention_mask=None, labels=None, **kwargs):
        image_embeds_proj = self.forward_image(image_embeds)

        model_returns = self.language_model(
            image_embeds=image_embeds_proj,
            input_ids=input_ids,
            media_offset=media_offset,
            attention_mask=attention_mask,
            use_cache=True,
            # return_dict=True,
            # output_attentions=True,
            output_hidden_states=True,
            labels=labels,
        )

        # logits = model_returns["logits"]

        # loss = None
        # if labels is not None:
        #     # Upcast to float if we need to compute the loss to avoid potential precision issues
        #     logits = logits.float()
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

        # model_returns["loss"] = loss

        if self.debug and "batch_count" in kwargs and kwargs["batch_count"] % 100 == 0:
            outputs = model_returns["logits"]
            output_tokens = outputs.argmax(-1)
            decoded_output = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)
            print(decoded_output)

        return model_returns
