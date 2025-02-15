from transformers import BitsAndBytesConfig, Qwen2VLForConditionalGeneration
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
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", output_dir="qwen2-2b-amos-mm"):
        self.model_id = model_id
        self.output_dir = output_dir
        self.processor = None
        self.model = None
        self.peft_model = None
        self.trainer = None
        
        # Get the base path from environment variable
        self.embeddings_path = os.environ.get('ROCKET_VLM_VISION_EMBEDDINGS')
        if self.embeddings_path is None:
            raise ValueError("Environment variable ROCKET_VLM_VISION_EMBEDDINGS is not set")

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

    def get_embedding_path(self, image_path):
        """Convert image path to embedding path"""
        base_name = image_path.replace('.nii.gz', '')
        return str(Path(self.embeddings_path) / f"{base_name}_0000_embedding.npz")

    def format_qa_data(self, embedding_path, qa_pair):
        """Format QA pair for training"""
        options_text = "\n".join([f"{key}. {value}" for key, value in qa_pair["options"].items()])
        correct_answer_text = qa_pair["options"][qa_pair["answer"]]
        
        if "reasoning" not in qa_pair:
            qa_pair["reasoning"] = ""

        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a medical imaging AI trained to answer questions about CT scans."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "embedding_path": embedding_path,
                    },
                    {
                        "type": "text",
                        "text": f"{qa_pair['question']}\n\n{options_text}"
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": f"The answer is {qa_pair['answer']}. {correct_answer_text}. {qa_pair['reasoning']}"
                    }
                ],
            },
        ]

    def format_region_question(self, embedding_path, region, findings):
        """Create a simple question about a specific region"""
        if not findings:
            return None
            
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a medical imaging AI trained to analyze CT scans."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "embedding_path": embedding_path,
                    },
                    {
                        "type": "text",
                        "text": f"What do you see in the {region} region of this CT scan?"
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": findings
                    }
                ],
            },
        ]

    def load_amos_dataset(self, dataset_path, split='train'):
        """Load AMOS-MM dataset with simple region-based questions"""
        with open(dataset_path) as f:
            data = json.load(f)
        
        samples = data['training'] if split == 'train' else data['validation']
        
        formatted_data = []
        for sample in samples:
            embedding_path = self.get_embedding_path(sample['image'])
            
            if 'labels' in sample and 'report' in sample['labels']:
                # Add region-specific questions
                for region in ['chest', 'abdomen', 'pelvis']:
                    region_qa = self.format_region_question(
                        embedding_path,
                        region,
                        sample['labels']['report']['findings'][region]
                    )
                    if region_qa:
                        formatted_data.append(region_qa)
                
                # Format QA pairs
                if 'qa' in sample['labels']:
                    for qa_pair in sample['labels']['qa']:
                        formatted_qa = self.format_qa_data(embedding_path, qa_pair)
                        formatted_data.append(formatted_qa)
        
        return formatted_data

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
        self.processor = Qwen2VLProcessor3D.from_pretrained(self.model_id)
        
        # Setup quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Initialize model
        self.model = Qwen2VLForConditionalGeneration3D.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )

        # Setup LoRA configuration
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
            eval_steps=10,
            eval_strategy="steps",
            save_strategy="steps",
            save_steps=20,
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

    def train(self, dataset_path):
        """Main training function"""
        # Load datasets
        train_dataset = self.load_amos_dataset(dataset_path, split='train')
        val_dataset = self.load_amos_dataset(dataset_path, split='val')
        print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")

        # Setup model and get configurations

        peft_config = self.setup_model()
        training_args = self.get_training_args()

        # Initialize trainer
        self.trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.collate_fn,
            peft_config=peft_config,
            tokenizer=self.processor.tokenizer,
        )

        # Train and save
        self.trainer.train()
        #self.trainer.save_model(self.output_dir)
        self.clear_memory()

    def test_dataset(self, dataset_path):
        """Test function to verify dataset loading"""
        print("Loading dataset...")
        train_dataset = self.load_amos_dataset(dataset_path, split='train')
        
        print(f"\nTotal number of samples: {len(train_dataset)}")
        
        # Print example of region-specific question
        print("\nExample of region question:")
        for sample in train_dataset:
            if "do you see" in sample[1]['content'][1]['text']:
                print(f"Question: {sample[1]['content'][1]['text']}")
                print(f"Answer: {sample[2]['content'][0]['text']}")
                break
        
        # Print example of original QA
        print("\nExample of original QA:")
        for sample in train_dataset:
            if "The answer is" in sample[2]['content'][0]['text']:
                print(f"Question: {sample[1]['content'][1]['text']}")
                print(f"Answer: {sample[2]['content'][0]['text']}")
                break

if __name__ == "__main__":
    dataset_path = "/home/m340g/DrivesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/dataset.json"
    
    # Initialize trainer
    trainer = VLMTrainer()
    
    # Test dataset loading
    trainer.test_dataset(dataset_path)
    
    # Run training
    trainer.train(dataset_path)