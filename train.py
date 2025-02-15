from transformers import BitsAndBytesConfig
import torch
from transformers import Qwen2VLForConditionalGeneration
from processing_3d import Qwen2VLProcessor3D
from Qwen3D import Qwen2VLForConditionalGeneration3D
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model
import os
from pathlib import Path
import json
import gc
import time
import random

# Get the base path from environment variable
ROCKET_VLM_VISION_EMBEDDINGS = os.environ.get('ROCKET_VLM_VISION_EMBEDDINGS')
if ROCKET_VLM_VISION_EMBEDDINGS is None:
    raise ValueError("Environment variable ROCKET_VLM_VISION_EMBEDDINGS is not set")

def clear_memory():
    # Memory cleanup implementation remains the same
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
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

def get_embedding_path(image_path):
    """Convert image path to embedding path"""
    # Extract the base name without extension
    base_name = Path(image_path).stem.replace('.nii', '')
    return str(Path(ROCKET_VLM_VISION_EMBEDDINGS) / f"{base_name}_0000_embedding.npz")

def format_qa_data(embedding_path, qa_pair):
    """Format QA pair for training"""
    # Format options as a string
    options_text = "\n".join([f"{key}. {value}" for key, value in qa_pair["options"].items()])
    
    # Get the correct answer's text
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

def format_report_data(embedding_path, report_data):
    """Format report data for training"""
    # Combine findings from different sections
    findings = []
    for section in ['chest', 'abdomen', 'pelvis']:
        if report_data['findings'][section]:
            findings.append(f"{section.capitalize()}: {report_data['findings'][section]}")
    full_report = " ".join(findings)

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a medical imaging AI trained to generate reports from CT scans."
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
                    "text": "Generate a detailed report for this CT scan."
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": full_report
                }
            ],
        },
    ]


def format_region_question(embedding_path, region, findings):
    """Create a simple question about a specific region"""
    if not findings:  # Skip if no findings for this region
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



def load_amos_dataset(dataset_path, split='train'):
    """Load AMOS-MM dataset with simple region-based questions"""
    with open(dataset_path) as f:
        data = json.load(f)
    
    if split == 'train':
        samples = data['training']
    else:  # validation
        samples = data['validation']
    
    formatted_data = []
    for sample in samples:
        embedding_path = get_embedding_path(sample['image'])
        
        if 'labels' in sample and 'report' in sample['labels']:
            # Add region-specific questions
            for region in ['chest', 'abdomen', 'pelvis']:
                region_qa = format_region_question(
                    embedding_path,
                    region,
                    sample['labels']['report']['findings'][region]
                )
                if region_qa:
                    formatted_data.append(region_qa)
            
            # Format QA pairs using the format_qa_data function
            if 'qa' in sample['labels']:
                for qa_pair in sample['labels']['qa']:
                    formatted_qa = format_qa_data(embedding_path, qa_pair)
                    formatted_data.append(formatted_qa)
    
    return formatted_data

def collate_fn_3d(examples):
    """Collate function implementation remains the same"""
    processor = examples[0]["processor"]
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]
    embedding_paths = [example[1]["content"][0]["embedding_path"] for example in examples]
    
    batch = processor(
        text=texts,
        vision_embedding_paths=embedding_paths,
        return_tensors="pt",
        padding=True
    )
    
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    if isinstance(processor, Qwen2VLProcessor3D):
        image_tokens = [151652, 151653, 151655]
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100
        
    batch["labels"] = labels
    return batch

def train_3d_model(dataset_path):
    """Main training function"""
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Load datasets
    train_dataset = load_amos_dataset(dataset_path, split='train')
    val_dataset = load_amos_dataset(dataset_path, split='val')
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # Example of data format
    #print("\nExample training sample:")
    #print(json.dumps(train_dataset[0], indent=2))
    
    # Load model and processor
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Qwen2VLForConditionalGeneration3D.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config
    )

    processor = Qwen2VLProcessor3D.from_pretrained(model_id)

    # LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # Training configuration
    training_args = SFTConfig(
        output_dir="qwen2-2b-amos-mm",
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

    # Add processor to datasets for collate_fn
    for item in train_dataset + val_dataset:
        item["processor"] = processor

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn_3d,
        peft_config=peft_config,
        tokenizer=processor.tokenizer,
    )

    # Train and save
    trainer.train()
    trainer.save_model(training_args.output_dir)
    clear_memory()



def test_dataset(dataset_path):
    """Test function to verify dataset loading"""
    print("Loading dataset...")
    train_dataset = load_amos_dataset(dataset_path, split='train')
    
    print(f"\nTotal number of samples: {len(train_dataset)}")
    
    # Print example of region-specific question
    print("\nExample of region question:")
    for sample in train_dataset:
        if "do you see" in sample[1]['content'][1]['text']:
            print(sample)
            print(f"Question: {sample[1]['content'][1]['text']}")
            print(f"Answer: {sample[2]['content'][0]['text']}...")
            break
    
    # Print example of original QA
    print("\nExample of original QA:")
    for sample in train_dataset:
        if "The answer is" in sample[2]['content'][0]['text']:
            print(sample)
            print(f"Question: {sample[1]['content'][1]['text']}")
            print(f"Answer: {sample[2]['content'][0]['text']}")
            break
if __name__ == "__main__":
    dataset_path = "/home/m340g/DrivesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/dataset.json"

    test_dataset(dataset_path)

    train_3d_model(dataset_path)