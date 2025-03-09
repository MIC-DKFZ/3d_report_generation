import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
)
from trl import SFTConfig, SFTTrainer

# Import OwnOwlModel from plygrnd.py if it is in the same workspace.
from model import OwnOwlModel  # ensure plygrnd.py is in PYTHONPATH


from transformers import TrainerCallback, TrainerControl, TrainerState

import wandb


class PrintFirstEvalBatchCallback(TrainerCallback):
    def __init__(self, tokenizer=None, processor=None):
        self.tokenizer = tokenizer
        self.printed = False
        self.sample_count = 2

        messages = [
            {
                "role": "user",
                "content": """<|image|>\n Describe this image.""",
            },
            {"role": "assistant", "content": "chest:"},
        ]

        # model_inputs = processor(messages, images=[image], videos=None, cut_enable=True)

        text = processor.build_text_qwen(messages)
        self.model_inputs = processor.encode_text_sft(text)

    # def on_prediction_step(self, args, state: TrainerState, control: TrainerControl, **kwargs):
    #     # The first time on_prediction_step is called during evaluation,
    #     # print the outputs (if available)
    #     if not self.printed:
    #         outputs = kwargs.get("outputs", None)
    #         if outputs is not None:
    #             print("First evaluation batch outputs:", outputs)
    #         else:
    #             print("First evaluation batch reached (no outputs available)")
    #         self.printed = True
    #     return control

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, metrics=None, **kwargs):
        print(f"Output samples from validation dataset after epoch 1:")
        # if state.epoch * 100 % 10 == 0:
        print(f"Epoch {state.epoch} reached")
        model.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():  # Disable gradient calculations
            for i, inputs in enumerate(eval_dataloader):
                if i >= self.sample_count:  # Limit the number of samples to print
                    break

                image_embeds = inputs["image_embeds"].to(args.device)
                image_lin = model.vision2text_model(image_embeds)

                new_input = inputs["input_ids"][:, :23]
                new_media_offset = inputs["media_offset"][:, :23]

                outputs = model.language_model.generate(
                    input_ids=new_input.to(args.device),
                    image_embeds=image_lin,
                    # attention_mask=inputs["attention_mask"].to(args.device),
                    media_offset=new_media_offset.to(args.device),
                    max_new_tokens=350,  # Adjust max_new_tokens as needed
                )

                # xd = model.language_model.forward(input_ids=input_ids)

                # Decode input and generated output
                # decoded_input = self.tokenizer.batch_decode(new_input, skip_special_tokens=True)
                decoded_output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                print(decoded_output)

                # print(f"\nSample {i+1}:")
                # print(f"  Input:     {decoded_input[0]}")  # Print the first item in the batch
                # print(f"  Generated: {decoded_output[0]}")  # Print the first item in the batch
                # # If you have labels and want to compare:
                # decoded_label = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                # print(f"  Label:     {decoded_label[0]}")

            model.train()  # Ensure model is back in training mode


class VLMTrainer:
    def __init__(
        self,
        model_id="mPLUG/mPLUG-Owl3-1B-241014",
        output_dir="owl-owl3-1b",
        mnt_dir="/home/david/mnt/",
        images_postix="E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings/192_reshaped/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/2024_MICCAI24_AMOS-MM/",
    ):
        self.model_id = model_id
        self.output_dir = output_dir
        self.mnt_dir = mnt_dir
        # images used to load npz embeddings
        self.images_path = os.path.join(self.mnt_dir, images_postix)
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.trainer = None
        self.device = torch.device("cuda:0")
        self.batch_count = 0

    def clear_memory(self):
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "processor"):
            del self.processor
        if hasattr(self, "trainer"):
            del self.trainer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    def setup_model(self):
        """
        Loads the configuration, language model, processor and then builds the OwnOwlModel.
        """
        config = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
        language_model = AutoModel.from_pretrained(
            # self.model_id, attn_implementation="sdpa", torch_dtype=torch.half, trust_remote_code=True
            self.model_id,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # Initialize processor via the language model
        processor = language_model.init_processor(tokenizer)
        processor.inference_mode = False

        # Create our custom OwnOwlModel. Adjust vision_dim and embed_dim if needed.
        own_model = OwnOwlModel(
            config,
            language_model=language_model.language_model,
            vision_dim=864,
            embed_dim=896,
            tokenizer=tokenizer,
            debug=False,
        )
        own_model.to(self.device)  # .half()

        # Freeze all parameters first.
        for param in own_model.parameters():
            param.requires_grad = False

        # Unfreeze only the vision2text_model parameters.
        # (Ensure that vision2text_model is an attribute of OwnOwlModel.)
        for param in own_model.vision2text_model.parameters():
            param.requires_grad = True

        for m_module in own_model.modules():
            print(m_module)
            # if (
            #     "HyperQwen2Attention" in str(m_module.__class__)
            #     or "HyperQwen2SdpaAttention" in str(m_module.__class__)
            #     or "HyperQwen2FlashAttention2" in str(m_module.__class__)
            # ):
            if "HyperQwen2DecoderLayer" in str(m_module.__class__):
                if m_module.is_hyper_enabled:
                    for param in m_module.self_attn.parameters():
                        param.requires_grad = True
            # if isinstance(module, HyperQwen2Attention):
            #     for param in module.parameters():
            #         param.requires_grad = True

        self.model = own_model
        self.processor = processor
        self.tokenizer = tokenizer

    def load_owl_dataset(self, dataset_json_path, split="train"):
        """
        Loads a dataset from a JSON file.
        For each sample, load image embeddings from an npz file
        (reshaping as in plygrnd.py) and save the report string.
        The text encoding is deferred to collate_fn.
        """
        with open(dataset_json_path, "r") as f:
            data = json.load(f)

        # Assume training samples are in key "training" (or change as needed)
        samples = data["training"] if split == "train" else data["validation"]
        dataset = []
        for sample in samples:
            # derive npz file name from sample image path
            np_img_name = sample["image"][2:].replace(".nii.gz", "_embedding.npz")
            img_path = os.path.join(self.images_path, np_img_name)
            if not os.path.exists(img_path):
                continue

            # Build a report by joining the key:value pairs
            report = "; ".join([k + ": " + v for k, v in sample["labels"]["report"]["findings"].items()])

            # Save the image tensor and report for later text processing in collate_fn.
            dataset.append(
                {
                    "img_path": img_path,
                    # "pixel_values": img_tensor,  # image embeddings
                    "report": report,
                }
            )
        return dataset

    def collate_fn(self, batch):
        """
        Collate a list of samples.
        Batch the image tensors and build/pad the text inputs.
        """
        # Stack image tensors (assumed to have the same shape)

        image_tensors = []

        for sample in batch:
            # Load and reshape the embedding as in plygrnd.py
            try:
                np_img = np.load(sample["img_path"])["embeddings"]
            except Exception as e:
                print(f"Error loading {sample["img_path"]}: {e}")
                continue

            # Flatten and transpose dimensions
            np_img_flat = np_img.reshape(np_img.shape[0], np_img.shape[1], -1)
            np_img_flat = np_img_flat.transpose(0, 2, 1)
            img_tensor = torch.tensor(np_img_flat)  # .half()
            image_tensors.append(img_tensor)

        images = torch.cat(image_tensors, dim=0)

        input_ids_list = []
        media_offset_list = []
        # For each sample, build the messages and encode using the processor.
        for sample in batch:
            messages = [
                {"role": "user", "content": "<|image|>\nDescribe this image."},
                {"role": "assistant", "content": sample["report"]},
            ]
            text = self.processor.build_text_qwen(messages)
            model_inputs = self.processor.encode_text_sft(text)
            input_ids_list.append(model_inputs["input_ids"].squeeze(0))
            media_offset_list.append(model_inputs["media_offset"])

        # Pad text inputs using the tokenizer.
        padded = self.tokenizer.pad({"input_ids": input_ids_list}, return_tensors="pt", padding="do_not_pad")
        # padded_media_offsets = self.tokenizer.pad({"input_ids": media_offset_list}, return_tensors="pt", padding="longest")
        pad_shape = padded["input_ids"].shape

        # Stack media offsets
        padded_media_offsets = []
        for media_offset in media_offset_list:
            media_offset = media_offset.squeeze(0)
            # turn to int tensor
            media_offset = media_offset.to(torch.int32)
            if media_offset.shape[0] < pad_shape[1]:
                pad = torch.zeros(
                    pad_shape[1] - media_offset.shape[0],
                    media_offset.shape[1],
                    dtype=torch.int32,
                )  # -1 for padding
                pad[:, :] = media_offset[-1, :]
                media_offset = torch.cat([media_offset, pad], dim=0)
            else:
                media_offset = media_offset[: pad_shape[1], :]
            padded_media_offsets.append(media_offset)
        media_offsets = torch.stack(padded_media_offsets, dim=0)
        # media_offsets = media_offsets.astype(np.int32)

        # Use input_ids as labels (mask pad tokens with -100)
        labels = padded["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch_out = {
            "image_embeds": images,
            "input_ids": padded["input_ids"],
            "attention_mask": padded["attention_mask"],
            "labels": labels,
            "media_offset": media_offsets,
            "batch_count": self.batch_count,
        }

        self.batch_count += 1

        return batch_out

    def get_training_args(self):
        return SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            learning_rate=2e-4,
            lr_scheduler_type="constant",
            logging_steps=10,
            eval_steps=10,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=200,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            bf16=True,
            # tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False,
            report_to=["wandb"],  # added to log metrics to wandb
            run_name="VLMTrainerRun",  # optional, set a name for the run
        )

    def train(self, dataset_json_path):
        """
        Main training function that uses the owl pipeline.
        """
        # Setup model, processor and tokenizer
        self.setup_model()

        # Initialize wandb
        wandb.init(project="vlm_project", reinit=True)

        # Load dataset using the owl pipeline
        train_dataset = self.load_owl_dataset(dataset_json_path, split="train")
        # For demonstration purposes, take a subset of validation samples if needed.
        val_dataset = self.load_owl_dataset(dataset_json_path, split="val")
        print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

        training_args = self.get_training_args()

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset[:50],
            data_collator=self.collate_fn,
            tokenizer=self.tokenizer,
            callbacks=[PrintFirstEvalBatchCallback(tokenizer=self.tokenizer, processor=self.processor)],
        )

        self.trainer.train()
        self.clear_memory()

        # Finish wandb run
        wandb.finish()

    def test_dataset(self, dataset_json_path):
        print("Loading Owl dataset...")
        dataset = self.load_owl_dataset(dataset_json_path, split="train")
        print(f"Total examples: {len(dataset)}")
        if dataset:
            sample = dataset[0]
            print("Sample keys:", list(sample.keys()))
            # print("Pixel values shape:", sample["pixel_values"].shape)
            # print("Input IDs shape:", sample["input_ids"].shape)
            # print("Media offset shape:", sample["media_offset"].shape)
        else:
            print("No samples loaded.")


if __name__ == "__main__":
    # dataset_json_path should point to the JSON file with dataset info.
    dataset_json_path = "/home/david/mnt/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/dataset.json"

    trainer = VLMTrainer()
    # trainer.setup_model()
    # trainer.test_dataset(dataset_json_path)
    trainer.train(dataset_json_path)
