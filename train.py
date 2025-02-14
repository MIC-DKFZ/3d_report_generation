from transformers import BitsAndBytesConfig
import torch
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor
from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model

import gc
import time

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


"""## 2. Load Dataset 📁"""

system_message = """You are a Vision Language Model specialized in interpreting visual data from chart images.
Your task is to analyze the provided chart image and respond to queries with concise answers, usually a single word, number, or short phrase.
The charts include a variety of types (e.g., line charts, bar charts) and contain colors, labels, and text.
Focus on delivering accurate, succinct answers based on the visual information. Avoid additional explanation unless absolutely necessary."""

"""We’ll format the dataset into a chatbot structure for interaction. Each interaction will consist of a system message, followed by the image and the user's query, and finally, the answer to the query.

💡For more usage tips specific to this model, check out the [Model Card](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct#more-usage-tips).
"""

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

"""For educational purposes, we’ll load only 10% of each split in the dataset. However, in a real-world use case, you would typically load the entire set of samples.

"""



dataset_id = "HuggingFaceM4/ChartQA"
train_dataset, eval_dataset, test_dataset = load_dataset(dataset_id, split=['train[:10%]', 'val[:10%]', 'test[:10%]'])

"""Let’s take a look at the structure of the dataset. It includes an image, a query, a label (which is the answer), and a fourth feature that we’ll be discarding."""

print("train_dataset", train_dataset)

"""Now, let’s format the data using the chatbot structure. This will allow us to set up the interactions appropriately for our model."""

train_dataset = [format_data(sample) for sample in train_dataset]
eval_dataset = [format_data(sample) for sample in eval_dataset]
test_dataset = [format_data(sample) for sample in test_dataset]

print("Formatted train_dataset[0]", train_dataset[0])
print("Formatted eval_dataset[0]", eval_dataset[0])
print("Formatted test_dataset[0]", test_dataset[0])

"""## 3. Load Model and Check Performance! 🤔"""


model_id = "Qwen/Qwen2-VL-2B-Instruct"

"""Next, we’ll load the model and the tokenizer to prepare for inference."""

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)

"""To evaluate the model's performance, we’ll use a sample from the dataset. First, let’s take a look at the internal structure of this sample.

"""

print("Sample from train_dataset", train_dataset[0])

"""We’ll use the sample without the system message to assess the VLM's raw understanding. Here’s the input we will use:"""

print("Sample input for model", train_dataset[0][1:2])

"""Now, let’s take a look at the chart corresponding to the sample. Can you answer the query based on the visual information?

"""

print("Chart image from sample", train_dataset[0][1]['content'][0]['image'])

"""Let’s create a method that takes the model, processor, and sample as inputs to generate the model's answer. This will allow us to streamline the inference process and easily evaluate the VLM's performance.

"""



def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2],  # Use the sample without the system message
        tokenize=False,
        add_generation_prompt=True
    )

    # Process the visual input from the sample
    image_inputs, _ = process_vision_info(sample)

    # Prepare the inputs for the model
    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

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

# Example of how to call the method with sample:
output = generate_text_from_sample(model, processor, train_dataset[0])
print(output)



clear_memory()

"""## 4. Fine-Tune the Model using TRL
### 4.1 Load the Quantized Model for Training ⚙️ """


# BitsAndBytesConfig int-4 config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load model and tokenizer
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

# Print model details
print("Model Configuration:", model.config)
print("Model Parameters:", model.num_parameters())
print("Model Architecture:", model)

processor = Qwen2VLProcessor.from_pretrained(model_id)

"""### 4.2 Set Up QLoRA and SFTConfig 🚀
"""


# Configure LoRA
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8,
    bias="none",
    target_modules=["q_proj", "v_proj"],
    task_type="CAUSAL_LM",
)

# Apply PEFT model adaptation
peft_model = get_peft_model(model, peft_config)

# Print trainable parameters
peft_model.print_trainable_parameters()


# Configure training arguments
training_args = SFTConfig(
    output_dir="qwen2-2b-instruct-trl-sft-ChartQA",  # Directory to save the model
    num_train_epochs=3,  # Number of training epochs
    per_device_train_batch_size=1,  # Batch size for training
    per_device_eval_batch_size=1,  # Batch size for evaluation
    gradient_accumulation_steps=8,  # Steps to accumulate gradients
    gradient_checkpointing=True,  # Enable gradient checkpointing for memory efficiency
    # Optimizer and scheduler settings
    optim="adamw_torch_fused",  # Optimizer type
    learning_rate=2e-4,  # Learning rate for training
    lr_scheduler_type="constant",  # Type of learning rate scheduler
    # Logging and evaluation
    logging_steps=10,  # Steps interval for logging
    eval_steps=10,  # Steps interval for evaluation
    eval_strategy="steps",  # Strategy for evaluation
    save_strategy="steps",  # Strategy for saving the model
    save_steps=20,  # Steps interval for saving
    metric_for_best_model="eval_loss",  # Metric to evaluate the best model
    greater_is_better=False,  # Whether higher metric values are better
    load_best_model_at_end=True,  # Load the best model after training
    # Mixed precision and gradient settings
    bf16=True,  # Use bfloat16 precision
    tf32=True,  # Use TensorFloat-32 precision
    max_grad_norm=0.3,  # Maximum norm for gradient clipping
    warmup_ratio=0.03,  # Ratio of total steps for warmup
    # Hub and reporting
    push_to_hub=True,  # Whether to push model to Hugging Face Hub
    report_to="wandb",  # Reporting tool for tracking metrics
    # Gradient checkpointing settings
    gradient_checkpointing_kwargs={"use_reentrant": False},  # Options for gradient checkpointing
    # Dataset configuration
    dataset_text_field="",  # Text field in dataset
    dataset_kwargs={"skip_prepare_dataset": True},  # Additional dataset options
    #max_seq_length=1024  # Maximum sequence length for input
)

training_args.remove_unused_columns = False  # Keep unused columns in dataset

"""### 4.3 Training the Model 🏃

We will log our training progress using [Weights & Biases (W&B)](https://wandb.ai/). Let’s connect our notebook to W&B to capture essential information during training.
"""

import wandb

wandb.init(
    project="qwen2-2-instruct-trl-sft-ChartQA",  # change this
    name="qwen2-2b-instruct-trl-sft-ChartQA",  # change this
    config=training_args,
)

"""We need a collator function to properly retrieve and batch the data during the training procedure. This function will handle the formatting of our dataset inputs, ensuring they are correctly structured for the model. Let's define the collator function below.

👉 Check out the TRL official example [scripts]( https://github.com/huggingface/trl/blob/main/examples/scripts/sft_vlm.py#L87) for more details.

"""

# Create a data collator to encode text and image pairs
def collate_fn(examples):
    # Get the texts and images, and apply the chat template
    texts = [processor.apply_chat_template(example, tokenize=False) for example in examples]  # Prepare texts for processing
    image_inputs = [process_vision_info(example)[0] for example in examples]  # Process the images to extract inputs


    # Tokenize the texts and process the images
    batch = processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)  # Encode texts and images into tensors

    #print("BATCH", batch)

    # The labels are the input_ids, and we mask the padding tokens in the loss computation
    labels = batch["input_ids"].clone()  # Clone input IDs for labels
    labels[labels == processor.tokenizer.pad_token_id] = -100  # Mask padding tokens in labels

    # Ignore the image token index in the loss computation (model specific)
    if isinstance(processor, Qwen2VLProcessor):  # Check if the processor is Qwen2VLProcessor
        image_tokens = [151652, 151653, 151655]  # Specific image token IDs for Qwen2VLProcessor
    else:
        image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]  # Convert image token to ID

    # Mask image token IDs in the labels
    for image_token_id in image_tokens:
        labels[labels == image_token_id] = -100  # Mask image token IDs in labels

    batch["labels"] = labels  # Add labels to the batch

    return batch  # Return the prepared batch

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=collate_fn,
    peft_config=peft_config,
    tokenizer=processor.tokenizer,
)

trainer.train()

trainer.save_model(training_args.output_dir)


'''
"""## 5. Testing the Fine-Tuned Model 🔍

Now that we've successfully fine-tuned our Vision Language Model (VLM), it's time to evaluate its performance! In this section, we will test the model using examples from the ChartQA dataset to see how well it answers questions based on chart images. Let's dive in and explore the results! 🚀

Let's clean up the GPU memory to ensure optimal performance 🧹
"""

clear_memory()

"""We will reload the base model using the same pipeline as before."""

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)

"""We will attach the trained adapter to the pretrained model. This adapter contains the fine-tuning adjustments we made during training, allowing the base model to leverage the new knowledge without altering its core parameters. By integrating the adapter, we can enhance the model's capabilities while maintaining its original structure.

"""

adapter_path = "sergiopaniego/qwen2-2b-instruct-trl-sft-ChartQA"
model.load_adapter(adapter_path)

"""We will utilize the previous sample from the dataset that the model initially struggled to answer correctly."""

train_dataset[0][:2]

train_dataset[0][1]['content'][0]['image']

output = generate_text_from_sample(model, processor, train_dataset[0])
output

"""Since this sample is drawn from the training set, the model has encountered it during training, which may be seen as a form of cheating. To gain a more comprehensive understanding of the model's performance, we will also evaluate it using an unseen sample.

"""

test_dataset[10][:2]

test_dataset[10][1]['content'][0]['image']

output = generate_text_from_sample(model, processor, test_dataset[10])
output

"""The model has successfully learned to respond to the queries as specified in the dataset. We've achieved our goal! 🎉✨

💻 I’ve developed an example application to test the model, which you can find [here](https://huggingface.co/spaces/sergiopaniego/Qwen2-VL-7B-trl-sft-ChartQA). You can easily compare it with another Space featuring the pre-trained model, available [here](https://huggingface.co/spaces/GanymedeNil/Qwen2-VL-7B).
"""

from IPython.display import IFrame

IFrame(src="https://sergiopaniego-qwen2-vl-7b-trl-sft-chartqa.hf.space", width=1000, height=800)

"""## 6. Compare Fine-Tuned Model vs. Base Model + Prompting 📊

We have explored how fine-tuning the VLM can be a valuable option for adapting it to our specific needs. Another approach to consider is directly using prompting or implementing a RAG system, which is covered in another [recipe](https://huggingface.co/learn/cookbook/multimodal_rag_using_document_retrieval_and_vlms).

Fine-tuning a VLM requires significant amounts of data and computational resources, which can incur costs. In contrast, we can experiment with prompting to see if we can achieve similar results without the overhead of fine-tuning.

Let's again clean up the GPU memory to ensure optimal performance 🧹
"""

clear_memory()

"""🏗️ First, we will load the baseline model following the same pipeline as before.

"""

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

processor = Qwen2VLProcessor.from_pretrained(model_id)

"""📜 In this case, we will again use the previous sample, but this time we will include the system message as follows. This addition helps to contextualize the input for the model, potentially improving its response accuracy.



"""

train_dataset[0][:2]

"""Let's see how it performs!"""

text = processor.apply_chat_template(
    train_dataset[0][:2], tokenize=False, add_generation_prompt=True
)

image_inputs, _ = process_vision_info(train_dataset[0])

inputs = processor(
    text=[text],
    images=image_inputs,
    return_tensors="pt",
)

inputs = inputs.to("cuda")

generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

output_text[0]

"""💡 As we can see, the model generates the correct answer using the pretrained model along with the additional system message, without any training. This approach may serve as a viable alternative to fine-tuning, depending on the specific use case.
"""

'''