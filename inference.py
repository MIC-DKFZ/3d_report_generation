from transformers import BitsAndBytesConfig
import torch
from transformers import Qwen2VLForConditionalGeneration
from processing_3d import Qwen2VLProcessor3D
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from Qwen3D import Qwen2VLForConditionalGeneration3D


from transformers import Qwen2VLProcessor
import gc
import time


import os
from pathlib import Path

# Get the base path from environment variable
ROCKET_VLM_VISION_EMBEDDINGS = os.environ.get('ROCKET_VLM_VISION_EMBEDDINGS')
if ROCKET_VLM_VISION_EMBEDDINGS is None:
    raise ValueError("Environment variable ROCKET_VLM_VISION_EMBEDDINGS is not set")


system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""


def clear_memory():
    # Delete variables if they exist in the current global scope
    if 'inputs' in globals(): del globals()['inputs']
    if 'model' in globals(): del globals()['model']
    if 'processor' in globals(): del globals()['processor']
    if 'trainer' in globals(): del globals()['trainer']
    if 'peft_model' in globals(): del globals()['peft_model']
    if 'bnb_config' in globals(): del globals()['bnb_config']
    time.sleep(2)

    # Garbage collection and clearing CUDA memory
    gc.collect()
    time.sleep(2)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    time.sleep(2)
    gc.collect()
    time.sleep(2)

    print(f"GPU allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"GPU reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

def format_data(sample):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample['query'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["label"][0]
                }
            ],
        },
    ]


def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2],  # Use the sample without the system message
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # load example embedding here use DrivesDKFZ/E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings/192_reshaped/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/2024_MICCAI24_AMOS-MM/imagesTr/
    vision_embedding_paths = [Path(ROCKET_VLM_VISION_EMBEDDINGS) / "imagesTr/amos_7385_0000_embedding.npz"]
    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        vision_embedding_paths=vision_embedding_paths,  # give the file path to the processor, it will load and return the volume
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    for key, value in model_inputs.items():
        print(f"{key}: shape={value.shape}, value={value}")

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]  # Return the first decoded output text

def inference():
    dataset_id = "HuggingFaceM4/ChartQA"
    train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id,
                                                             split=['train[:10%]', 'val[:10%]', 'test[:10%]'])

    train_dataset = [format_data(sample) for sample in train_dataset]
    eval_dataset = [format_data(sample) for sample in eval_dataset]
    test_dataset = [format_data(sample) for sample in test_dataset]

    model_id = "Qwen/Qwen2-VL-2B-Instruct"

    model = Qwen2VLForConditionalGeneration3D.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )

    processor = Qwen2VLProcessor3D.from_pretrained(model_id)
    # Example of how to call the method with sample:
    output = generate_text_from_sample(model, processor, train_dataset[0])
    print(output)

    clear_memory()

if __name__=="__main__":
    inference()