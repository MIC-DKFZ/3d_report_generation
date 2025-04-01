from train import VLMTrainer
import torch 
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    # Trainer,
    # TrainingArguments
)
# from torch.utils.data import DataLoader
from train import PrintFirstEvalBatchCallback
from model import OwnOwlModel

from trl import SFTTrainer
import gc
import time
import os
import numpy as np
import json

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

def inference(model_id, dataset_json_path):
    # TODO: later get the dataset with using a Dataset class
    vlm_trainer = VLMTrainer(model_id=model_id)
    val_dataset = vlm_trainer.load_owl_dataset(dataset_json_path, split='val')

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    language_model = AutoModel.from_pretrained(
        model_id,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    vlm_trainer.tokenizer = tokenizer
    # Initialize processor via the language model
    processor = language_model.init_processor(tokenizer)
    processor.inference_mode = True
    vlm_trainer.processor = processor

    # Create our custom OwnOwlModel. Adjust vision_dim and embed_dim if needed.
    # TODO: did we set the torch dtype anywhere
    own_model = OwnOwlModel(
        config,
        language_model=language_model.language_model,
        vision_dim=864,
        embed_dim=896,
        tokenizer=tokenizer,
        debug=False,
    )
    own_model.eval().cuda()
    
    # Freeze all parameters
    for param in own_model.parameters():
        param.requires_grad = False

    # inputs = prepare_data(tokenizer, processor, val_dataset[:5])
    inputs = vlm_trainer.collate_fn(val_dataset[:5])

    with torch.no_grad():  # Disable gradient calculations
        image_embeds = inputs["image_embeds"].to("cuda")
        image_lin = own_model.vision2text_model(image_embeds)

        new_input = inputs["input_ids"][:, :23]
        new_media_offset = inputs["media_offset"][:, :23]

        outputs = own_model.language_model.generate(
            input_ids=new_input.to("cuda"),
            image_embeds=image_lin,
            # attention_mask=inputs["attention_mask"].to("cuda"),
            media_offset=new_media_offset.to("cuda"),
            max_new_tokens=256,  # Adjust max_new_tokens as needed
        )

        # Trim the generated ids to remove the input ids
        trimmed_generated_ids = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(new_input, outputs)
        ]
        decoded_output = tokenizer.batch_decode(trimmed_generated_ids, skip_special_tokens=True)
        print(decoded_output)

    clear_memory()

    results = []
    input_text = tokenizer.batch_decode([outputs[0][:len(new_input[0])]], skip_special_tokens=True)[0]
    for input, output in zip(val_dataset[:5], decoded_output):
        img_path = input["img_path"]

        results.append({
            "img_path": img_path,
            "input_text": input_text,
            "output_text": output,
            "gt_text": input["report"],
        })  

    # Define the output directory and file path
    output_dir = "./results"
    output_file_path = os.path.join(output_dir, "inference_results_owl.json")

    # Check if the directory exists before creating it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(output_file_path, "w") as f:
        json.dump(results, f, indent=4)
    
    # # TODO: once we have a proper dataset class here we can easily swith to huggingface Trainer similar to below
    # training_args = vlm_trainer.get_training_args()
    # trainer = SFTTrainer(
    #     model=own_model,
    #     args=training_args,
    #     eval_dataset=val_dataset[:5],
    #     data_collator=vlm_trainer.collate_fn,
    #     tokenizer=tokenizer,
    # )
    
    # # Run Validation
    # eval_results = trainer.predict(test_dataset=val_dataset[:5])
    # vlm_trainer.clear_memory()
    # print("Validation Results:", eval_results)

if __name__=="__main__":
    dataset_json_path = "./E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/dataset.json"
    model_id = "mPLUG/mPLUG-Owl3-1B-241014"
    inference(model_id, dataset_json_path)