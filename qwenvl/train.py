from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration

from data.datasets import AMOSMMDataset
from processing_3d import Qwen2VLProcessor3D
from Qwen3D import Qwen2VLForConditionalGeneration3D
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
import torch
import os
from pathlib import Path
import json
import gc
import time


class VLMTrainer:
    def __init__(self, 
                 dataset_path: str,
                 model_id="Qwen/Qwen2-VL-2B-Instruct", 
                 output_dir="qwen2-2b-amos-mm", 
                 train_only_vision_adapter=False):
        self.model_id = model_id
        self.output_dir = output_dir
        self.processor = None
        self.model = None
        self.peft_model = None
        self.trainer = None
        self.train_only_vision_adapter = train_only_vision_adapter
        
        # Get the base path from environment variable
        self.embeddings_path = os.environ.get('ROCKET_VLM_VISION_EMBEDDINGS')
        if self.embeddings_path is None:
            raise ValueError("Environment variable ROCKET_VLM_VISION_EMBEDDINGS is not set")

        # TODO - rename class to VLMAmosMMTrainer
        self.train_dataset = AMOSMMDataset(dataset_json_path=dataset_path,
                                           model_name="qwen",
                                           split='train',
                                           embeddings_path=self.embeddings_path)
        self.val_dataset = AMOSMMDataset(dataset_json_path=dataset_path,
                                         model_name="qwen",
                                         split='val',
                                         embeddings_path=self.embeddings_path)[:50]
        self.train_dataset.verify()
        print(f"Loaded {len(self.train_dataset)} training samples and {len(self.val_dataset)} validation samples")

    def clear_memory(self):
        """Clear GPU memory and cached objects"""
        if hasattr(self, 'model'): del self.model
        if hasattr(self, 'processor'): del self.processor
        if hasattr(self, 'trainer'): del self.trainer
        if hasattr(self, 'peft_model'): del self.peft_model
        
        time.sleep(2)
        gc.collect()
        time.sleep(2)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(2)
        gc.collect()
        time.sleep(2)
        print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    def collate_fn(self, examples):
        # Get the texts and images, and apply the chat template
        texts = [self.processor.apply_chat_template(example, tokenize=False) for example in examples]
        embedding_paths = [example[1]["content"][0]["embedding_path"] for example in examples]
        
        batch = self.processor(
            text=texts,
            vision_embedding_paths=embedding_paths,
            return_tensors="pt",
            padding=True
        )
        
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        if isinstance(self.processor, Qwen2VLProcessor3D):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
            
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100


        video_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.video_token)]
        for video_token_id in video_tokens:
            labels[labels == video_token_id] = -100
    
            
        batch["labels"] = labels
        return batch

    def setup_model(self):
        """Initialize model, processor and PEFT configuration"""
        # Initialize processor
        self.processor = Qwen2VLProcessor3D.from_pretrained(self.model_id, cache_dir="/home/r403k/.cache/huggingface/hub", local_files_only=True)  
        
        if self.train_only_vision_adapter:
            # For adapter-only training, don't use 4-bit quantization for the adapter
            self.model = Qwen2VLForConditionalGeneration3D.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            
            # Freeze all parameters
            for name, param in self.model.named_parameters():
                if "vision_adapter" not in name:  # Freeze everything except vision adapter
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                    print(f"Unfreezing: {name}")
            
            # Print trainable parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"Total parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")
            
            return None
        else:
            # 4-bit quantization for full model training (maybe not needed for 24gb vram workstation)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

            self.model = Qwen2VLForConditionalGeneration3D.from_pretrained(
                self.model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config
            )

            peft_config = LoraConfig(
                lora_alpha=16,
                lora_dropout=0.05,
                r=8,
                bias="none",
                target_modules=["q_proj", "v_proj"],
                task_type="CAUSAL_LM",
            )

            self.peft_model = get_peft_model(self.model, peft_config)
            self.peft_model.print_trainable_parameters()
            
            return peft_config

    def get_training_args(self):
        """Configure training arguments"""
        return SFTConfig(
            output_dir=self.output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            optim="adamw_torch_fused",
            learning_rate=2e-4,
            lr_scheduler_type="constant",
            logging_steps=10,
            eval_steps=50,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=50,
            save_total_limit=2,  # Only keep 2 checkpoints
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            load_best_model_at_end=True,
            bf16=True,
            tf32=True,
            max_grad_norm=0.3,
            warmup_ratio=0.03,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataset_kwargs={"skip_prepare_dataset": True},
            remove_unused_columns=False
        )

    def train(self):
        """Main training function"""
        # Setup model and get configurations

        peft_config = self.setup_model()
        training_args = self.get_training_args()

        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.collate_fn,
            peft_config=None if self.train_only_vision_adapter else peft_config,
            # tokenizer=self.processor.tokenizer,
        )

        # Train and save
        self.trainer.train()
        #self.trainer.save_model(self.output_dir)
        self.clear_memory()


if __name__ == "__main__":
    dataset_path = "/home/r403k/ResourcesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/dataset.json"
    
    # Initialize trainer
    trainer = VLMTrainer(dataset_path, model_id="/home/r403k/.cache/huggingface/hub/Qwen2-VL-2B-Instruct", train_only_vision_adapter=True)
    
    # Run training
    trainer.train()