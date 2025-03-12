import os
import random
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import gc
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))


from callbacks import PrintEvalBatchCallback, PrintFirstEvalBatchCallback

# Import OwnOwlModel from plygrnd.py if it is in the same workspace.
from model import OwnOwlModel  # ensure plygrnd.py is in PYTHONPATH
from transformers import TrainerCallback, TrainerControl, TrainerState

import wandb
from data.datasets import AMOSMMDataset, LIDCDataset


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

    def pad_inputs(self, input_ids_list, input_ids_list_prompt_only, media_offset_list):
        # Pad text inputs using the tokenizer.
        padded = self.tokenizer.pad(
            {"input_ids": input_ids_list}, return_tensors="pt", padding="do_not_pad"
        )
        padded_prompt_only = self.tokenizer.pad(
            {"input_ids": input_ids_list_prompt_only},
            return_tensors="pt",
            padding="do_not_pad",
        )
        pad_shape = padded["input_ids"].shape

        padded_media_offsets = []
        for media_offset in media_offset_list:
            media_offset = media_offset.squeeze(0).to(torch.int32)
            # If the media offset is shorter than the pad width, pad it with the last row.
            if media_offset.shape[0] < pad_shape[1]:
                pad_amount = pad_shape[1] - media_offset.shape[0]
                pad = torch.zeros(pad_amount, media_offset.shape[1], dtype=torch.int32)
                pad[:] = media_offset[-1, :]
                media_offset = torch.cat([media_offset, pad], dim=0)
            else:
                media_offset = media_offset[: pad_shape[1], :]
            padded_media_offsets.append(media_offset)
        media_offsets = torch.stack(padded_media_offsets, dim=0)
        return padded, padded_prompt_only, media_offsets

    # media_offsets = media_offsets.astype(np.int32)

    def collate_fn(self, batch):
        """
        Collate a list of samples.
        Batch the image tensors and build/pad the text inputs.
        """
        # Stack image tensors (assumed to have the same shape)

        image_tensors = []
        input_ids_list = []
        input_ids_list_prompt_only = []
        media_offset_list = []

        for sample in batch:

            # if len(sample["qa_text"]) > 0:
            #     rand_idx = random.randint(0, len(sample["qa_text"]) - 1)
            #     messages = sample["qa_text"][rand_idx]
            #     messages_prompt_only = sample["qa_text_prompt_only"][rand_idx]
            # else:
            messages = sample["formatted_qa"]
            messages_prompt_only = sample["formatted_qa_prompt_only"]

            # Flatten and transpose dimensions
            img_tensor = sample["embedding_tensor"]
            image_tensors.append(img_tensor)

            # rand_idx = random.randint(0, len(sample["qa_text"]) - 1)

            # if "tokenized" in sample:
            #     model_inputs = sample["tokenized"]["qa_text_tokens"][rand_idx]
            #     model_inputs_prompt_only = sample["tokenized"]["qa_text_prompt_only_tokens"][rand_idx]
            # else:
            #     model_inputs = self.processor.encode_text_sft(self.processor.build_text_qwen(messages))
            #     model_inputs_prompt_only = self.processor.encode_text_sft(
            #         self.processor.build_text_qwen(messages_prompt_only)
            #     )

            model_inputs = sample["tokenized_qa_text_tokens"]
            model_inputs_prompt_only = sample["tokenized_qa_text_prompt_only_tokens"]

            input_ids_list.append(model_inputs["input_ids"].squeeze(0))
            input_ids_list_prompt_only.append(
                model_inputs_prompt_only["input_ids"].squeeze(0)
            )
            media_offset_list.append(model_inputs["media_offset"])

        images = torch.cat(image_tensors, dim=0)

        padded, padded_prompt_only, media_offsets = self.pad_inputs(
            input_ids_list, input_ids_list_prompt_only, media_offset_list
        )

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
            "input_ids_prompt_only": padded_prompt_only["input_ids"],
        }

        self.batch_count += 1

        return batch_out

    def get_training_args(self):
        return SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=20,
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
        # train_dataset = self.load_owl_dataset(dataset_json_path, split="train")

        tokenize_fn = lambda x: self.processor.encode_text_sft(
            self.processor.build_text_qwen(x)
        )

        train_dataset = AMOSMMDataset(
            dataset_json_path=dataset_json_path,
            model_name="owl",
            split="train",
            embeddings_path="/home/david/mnt/cluster_data/m340g/vision_embeddings/192_reshaped/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/2024_MICCAI24_AMOS-MM",
            tokenize_fn=tokenize_fn,
        )
        val_dataset = AMOSMMDataset(
            dataset_json_path=dataset_json_path,
            model_name="owl",
            split="val",
            embeddings_path="/home/david/mnt/cluster_data/m340g/vision_embeddings/192_reshaped/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/2024_MICCAI24_AMOS-MM",
            n_samples=50,
            tokenize_fn=tokenize_fn,
        )

        # train_dataset = LIDCDataset(
        #     dataset_json_path=dataset_json_path,
        #     model_name="owl",
        #     split="train",
        #     embeddings_path="/home/david/mnt/E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings/LIDC-IDRI/192_sliding_window/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/npz",
        # )
        # val_dataset = LIDCDataset(
        #     dataset_json_path=dataset_json_path,
        #     model_name="owl",
        #     split="val",
        #     embeddings_path="/home/david/mnt/E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings/LIDC-IDRI/192_sliding_window/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/npz",
        #     # n_samples=50,
        # )

        # For demonstration purposes, take a subset of validation samples if needed.
        # val_dataset = self.load_owl_dataset(dataset_json_path, split="val")
        print(
            f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples"
        )

        training_args = self.get_training_args()

        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collate_fn,
            tokenizer=self.tokenizer,
            callbacks=[
                PrintEvalBatchCallback(
                    tokenizer=self.tokenizer, processor=self.processor
                )
            ],
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
    dataset_json_path = "/home/david/mnt/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/"
    # dataset_json_path = "/home/david/ws/vlm/data/lidc"

    trainer = VLMTrainer()
    # trainer.setup_model()
    # trainer.test_dataset(dataset_json_path)
    trainer.train(dataset_json_path)
