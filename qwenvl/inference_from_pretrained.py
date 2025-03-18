import torch
import os
from pathlib import Path
import argparse
import json
from tqdm import tqdm
import time
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model and processor classes
from Qwen3D import Qwen2VLForConditionalGeneration3D
from processing_3d import Qwen2VLProcessor3D

# Import dataset, formatters, and extractors
from data.datasets import AMOSMMDataset


class VLMInference:
    def __init__(self, checkpoint_path, base_processor_model):
        """
        Initialize the VLM inference class.
        
        Args:
            checkpoint_path: Path to fine-tuned checkpoint
            base_processor_model: Base model ID to use for the processor
        """
        self.checkpoint_path = checkpoint_path
        self.base_processor_model = base_processor_model
        
        # Set up embeddings path - same as during training
        self.embeddings_path = os.environ.get('ROCKET_VLM_VISION_EMBEDDINGS')
        if self.embeddings_path is None:
            raise ValueError("Environment variable ROCKET_VLM_VISION_EMBEDDINGS is not set")
        
        # Load model and processor
        self.setup_model()
    
    def setup_model(self):
        """Load processor from base model and model from checkpoint"""
        print(f"Loading processor from base model: {self.base_processor_model}")
        self.processor = Qwen2VLProcessor3D.from_pretrained(self.base_processor_model)
        
        print(f"Loading model from checkpoint: {self.checkpoint_path}")
        self.model = Qwen2VLForConditionalGeneration3D.from_pretrained(
            self.checkpoint_path,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # Set model to evaluation mode
        self.model.eval()
        print(f"Model loaded successfully and set to evaluation mode")

    def process_example(self, example):
        """
        Process a single example using the prompt-only part of a conversation.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            inputs: Processed inputs ready for the model
            prompt_conversation: The conversation prompt (without answers)
            metadata: Additional information about the example
        """
        metadata = {}
        
        # For QA examples
        if "qa_text_prompt_only" in example and len(example["qa_text_prompt_only"]) > 0:
            # Get the prompt-only conversation (system + user message only)
            prompt_conversation = example["qa_text_prompt_only"][0]  # Use first QA pair
            metadata["type"] = "qa"
            
            # Get ground truth for evaluation (but don't pass to model)
            if "qa_text" in example and len(example["qa_text"]) > 0:
                full_conversation = example["qa_text"][0]
                for msg in full_conversation:
                    if msg.get("role") == "assistant":
                        metadata["ground_truth"] = msg.get("content", [{}])[0].get("text", "")
            
            # Get question details from item_qa if available
            if "item_qa" in example and len(example["item_qa"]) > 0:
                qa_item = example["item_qa"][0]
                metadata["question"] = qa_item.get("question", "")
                metadata["options"] = qa_item.get("options", {})
                metadata["answer"] = qa_item.get("answer", "")
        
        # For region description examples
        elif "report_text_prompt_only" in example and len(example["report_text_prompt_only"]) > 0:
            # Get the prompt-only conversation (system + user message only)
            prompt_conversation = example["report_text_prompt_only"][0]  # Use first region
            metadata["type"] = "region"
            
            # Get ground truth for evaluation (but don't pass to model)
            if "report_text" in example and len(example["report_text"]) > 0:
                full_conversation = example["report_text"][0]
                for msg in full_conversation:
                    if msg.get("role") == "assistant":
                        metadata["ground_truth"] = msg.get("content", [{}])[0].get("text", "")
            
            # Get region details if available
            if "item_report" in example and "region_questions" in example["item_report"]:
                for region, region_data in example["item_report"]["region_questions"].items():
                    if region_data:
                        metadata["region"] = region
                        break
        
        else:
            raise ValueError("Could not find any prompt_only conversations in the example")
        
        # Extract user query for display
        user_query = None
        embedding_path = None
        
        for msg in prompt_conversation:
            if msg.get("role") == "user":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                user_query = item.get("text", "")
                            elif item.get("type") == "image" and "embedding_path" in item:
                                embedding_path = item["embedding_path"]
        
        if not embedding_path and "embedding_path" in example:
            embedding_path = example["embedding_path"]
        
        metadata["user_query"] = user_query
        metadata["embedding_path"] = embedding_path
        
        # Apply chat template to the conversation
        text = self.processor.apply_chat_template(prompt_conversation, tokenize=False)
        
        # Process inputs
        inputs = self.processor(
            text=text,
            vision_embedding_paths=[embedding_path],
            return_tensors="pt",
            padding=True
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        return inputs, prompt_conversation, metadata
    
    def generate(self, inputs, max_new_tokens=500, temperature=0.7, top_p=0.9):
        """
        Generate text response with no special formatting.
        
        Args:
            inputs: Processed inputs from process_example
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Raw generated text response
        """
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p
            )
        
        # Decode output without special formatting
        generated_text = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        
        # Extract just the model's response
        if "ASSISTANT:" in generated_text:
            response = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            response = generated_text
        
        return response
    
    def run_inference_on_dataset(self, dataset_path, split="val", output_file=None):
        """
        Run inference on the entire dataset.
        
        Args:
            dataset_path: Path to the dataset
            split: Dataset split to use
            output_file: File to save results
            
        Returns:
            List of results
        """
        # Load dataset
        dataset = AMOSMMDataset(
            dataset_json_path=dataset_path,
            model_name="qwen",
            split=split,
            embeddings_path=self.embeddings_path
        )
        
        print(f"Running inference on {len(dataset)} examples from {split} split")
        
        # Process all samples
        results = []
        for idx in tqdm(range(len(dataset)), desc="Running inference"):
            try:
                example = dataset[idx]
                
                # Process and generate
                inputs, prompt_conversation, metadata = self.process_example(example)
                response = self.generate(inputs)
                
                # Save full result with metadata
                result = {
                    "index": idx,
                    "query": metadata["user_query"],
                    "response": response,
                    "ground_truth": metadata.get("ground_truth", ""),
                    "embedding_path": metadata["embedding_path"],
                    "question_type": metadata["type"],
                }
                
                # Add additional metadata based on question type
                if metadata["type"] == "qa":
                    result.update({
                        "question": metadata.get("question", ""),
                        "options": metadata.get("options", {}),
                        "correct_answer": metadata.get("answer", "")
                    })
                elif metadata["type"] == "region":
                    result.update({
                        "region": metadata.get("region", "")
                    })
                
                # Try to extract image ID if available
                try:
                    if metadata["embedding_path"]:
                        # Extract image ID from embedding path
                        basename = os.path.basename(metadata["embedding_path"])
                        image_id = basename.split("_embedding")[0]
                        result["image_id"] = image_id
                except Exception:
                    pass
                
                results.append(result)
                
                # Print progress updates periodically - FIXED: don't print system/user messages
                if idx % 1 == 0:  # Changed from every example to every 10 examples
                    print(f"\nProcessed {idx}/{len(dataset)} examples")
                    print(f"Type: {metadata['type']}")
                    print(f"Query: {metadata['user_query']}")
                    print(f"Model response: {response}")
                    # FIXED: Also print ground truth
                    print(f"Ground truth: {metadata.get('ground_truth', 'Not available')}")
                    print("-" * 50)  # Add separator for clarity
                
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                import traceback
                traceback.print_exc()
        
        # Save results if output file provided
        if output_file:
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen2-VL model on AMOS dataset")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/m340g/Projects/rocket-vlm/qwen2-2b-amos-mm/checkpoint-3050",
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--base_processor", type=str, 
                        default="Qwen/Qwen2-VL-2B-Instruct",
                        help="Base model ID to use for the processor")
    parser.add_argument("--dataset", type=str, 
                        default="/home/m340g/DrivesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM", 
                        help="Path to the dataset directory")
    parser.add_argument("--split", type=str, default="val", 
                        help="Dataset split to use (train, val, test)")
    parser.add_argument("--output_file", type=str, default=None,
                        help="Path to save inference results (JSON)")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling parameter")
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output for each example")
    
    args = parser.parse_args()
    
    # Set default output file if not provided
    if not args.output_file:
        timestamp = int(time.time())
        args.output_file = f"inference_results_{args.split}_{timestamp}.json"
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory: {output_dir}")
    
    # Initialize inference
    inference = VLMInference(args.checkpoint, args.base_processor)
    
    # Run inference on dataset
    results = inference.run_inference_on_dataset(
        args.dataset,
        split=args.split,
        output_file=args.output_file
    )
    
    print(f"Completed inference on {len(results)} samples")
    print(f"Results saved to {args.output_file}")
    
    # Calculate basic statistics
    question_types = {}
    for result in results:
        q_type = result.get("question_type", "unknown")
        if q_type not in question_types:
            question_types[q_type] = 0
        question_types[q_type] += 1
    
    print("\nQuestion type statistics:")
    for q_type, count in question_types.items():
        print(f"  {q_type}: {count} questions ({count/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()