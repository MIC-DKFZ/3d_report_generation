import torch
import os
from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm
import glob
import time

from Qwen3D import Qwen2VLForConditionalGeneration3D
from processing_3d import Qwen2VLProcessor3D
from data.datasets import AMOSMMDataset
from data.formatters import get_formatter


class VLMInference:
    def __init__(self, checkpoint_path="/home/m340g/Projects/rocket-vlm/qwen2-2b-amos-mm/checkpoint-3050", 
                base_processor_model="Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initialize the VLM inference class.
        
        Args:
            checkpoint_path: Path to fine-tuned checkpoint
            base_processor_model: Base model ID to use for the processor
        """
        self.checkpoint_path = checkpoint_path
        self.base_processor_model = base_processor_model
        self.processor = None
        self.model = None
        
        # Set up embeddings path (same as in training)
        self.embeddings_path = os.environ.get('ROCKET_VLM_VISION_EMBEDDINGS')
        if self.embeddings_path is None:
            raise ValueError("Environment variable ROCKET_VLM_VISION_EMBEDDINGS is not set")
        
        # Load model and processor
        self.setup_model()
    
    def setup_model(self):
        """Load processor from base model and model from checkpoint"""
        try:
            # Load processor from the base model (which has the preprocessor config)
            print(f"Loading processor from base model: {self.base_processor_model}")
            self.processor = Qwen2VLProcessor3D.from_pretrained(self.base_processor_model)
            
            # Load model from the fine-tuned checkpoint
            print(f"Loading model from checkpoint: {self.checkpoint_path}")
            self.model = Qwen2VLForConditionalGeneration3D.from_pretrained(
                self.checkpoint_path,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            
            # Set model to evaluation mode
            self.model.eval()
            print(f"Model and processor loaded successfully")
            
        except Exception as e:
            print(f"Error in setup_model: {e}")
            import traceback
            traceback.print_exc()
            raise
    

    # In the process_example method, add this near the beginning:
    def process_example(self, example):
        """Process a single example"""
        for msg in example:
            if isinstance(msg, dict):
                role = msg.get('role', 'unknown')
                content = msg.get('content', [])
                if isinstance(content, list):
                    content_text = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get('type') == 'text':
                                content_text.append(item.get('text', ''))
                            elif item.get('type') == 'image':
                                content_text.append('[IMAGE]')
                    content = ' '.join(content_text)
        """
        Process a single example from the dataset, similar to the collate_fn in training.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            inputs: Processed inputs ready for the model
            user_query: The text query from the user
        """
        try:
            # Apply chat template to the example
            text = self.processor.apply_chat_template(example, tokenize=False)
            
            # Get embedding path - making sure it matches the dataset format
            embedding_path = example[1]["content"][0]["embedding_path"]
            
            # Process inputs
            inputs = self.processor(
                text=text,
                vision_embedding_paths=[embedding_path],
                return_tensors="pt",
                padding=True
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Extract user query for display
            user_query = None
            for msg in example:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = msg.get("content")
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and item.get("type") == "text":
                                user_query = item.get("text")
                                break
                    break
            
            if not user_query:
                user_query = "Describe this medical image"
            
            return inputs, user_query, embedding_path
            
        except Exception as e:
            print(f"Error processing example: {e}")
            print("Example structure:", example)
            raise
    
    def generate(self, inputs, max_new_tokens=500, temperature=0.7, top_p=0.9):
        """
        Generate text response.
        
        Args:
            inputs: Processed inputs from process_example
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text response
        """
        try:
            # Generate
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p
                )
            
            # Decode output
            generated_text = self.processor.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            
            # Extract just the model's response
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text
            
            # Properly clean the response by removing anything before "The answer is" for QA questions
            if "The answer is" in response:
                response = response[response.find("The answer is"):].strip()
            elif "A." in response and "B." in response:
                # Handle case where model repeats the options
                lines = response.split("\n")
                answer_start = -1
                for i, line in enumerate(lines):
                    if "A." in line and i < len(lines) - 1 and "B." in lines[i+1]:
                        answer_start = i + 4  # Skip past the options (A, B, C, D)
                        break
                
                if answer_start >= 0 and answer_start < len(lines):
                    response = "\n".join(lines[answer_start:])
            
            # Clean up unwanted phrases and system/user/assistant text
            response = self._clean_response_text(response)
            
            return response
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
            

    def run_test_mode(self, dataset_path, split="train", num_samples=5, question_types=None):
        """
        Run a true test without seeing any answers.
        """
        # Load dataset
        dataset = AMOSMMDataset(
            dataset_json_path=dataset_path,
            model_name="qwen",
            split=split,
            embeddings_path=self.embeddings_path
        )
        
        # Filter and select samples
        samples = []
        for i in range(len(dataset)):
            example = dataset[i]
            # Filter by question type if needed
            if question_types:
                is_qa = any("The answer is" in str(msg.get("content", [{}])[0].get("text", "")) 
                        for msg in example if isinstance(msg, dict) and msg.get("role") == "assistant")
                if ("qa" in question_types and is_qa) or ("region" in question_types and not is_qa):
                    samples.append((i, example))
            else:
                samples.append((i, example))
        
        if len(samples) > num_samples:
            samples = samples[:num_samples]
        
        # Process samples in true test mode
        results = []
        for idx, (orig_idx, example) in enumerate(samples):
            # Create a clean example with ONLY system and user messages
            clean_example = []
            ground_truth = None
            
            for msg in example:
                if isinstance(msg, dict):
                    if msg.get("role") == "system":
                        clean_example.append(msg)
                    elif msg.get("role") == "user":
                        clean_example.append(msg)
                    elif msg.get("role") == "assistant":
                        # Save ground truth but don't add to clean example
                        content = msg.get("content")
                        if isinstance(content, list) and len(content) > 0:
                            ground_truth = content[0].get("text", None)
            
            # Process and generate
            inputs, user_query, _ = self.process_example(clean_example)
            response = self.generate(inputs)
            
            # Print and save results
            print(f"Query: {user_query}")
            print(f"Response: {response}")
            print(f"Ground Truth: {ground_truth}")
            
            results.append({
                "index": orig_idx,
                "query": user_query,
                "response": response,
                "ground_truth": ground_truth
            })

            print("------------------------------------------------------------")
        
        return results
    
    def _clean_response_text(self, text):
        """Clean up the response text by removing unwanted elements"""
        # Remove common unwanted artifacts
        unwanted_phrases = [
            "We use it is not a problem.",
            "We use it is not clear.",
            "We have a new approach",
            "a medical imaging AI trained to",
        ]
        
        for phrase in unwanted_phrases:
            text = text.replace(phrase, "").strip()
        
        # Remove system/user/assistant prefixes
        role_prefixes = ["system\n", "user\n", "assistant\n"]
        for prefix in role_prefixes:
            text = text.replace(prefix, "")
        
        # Remove any repeated question text
        if "What do you see in the" in text:
            parts = text.split("What do you see in the")
            if len(parts) > 1:
                remainder = parts[1]
                if "region of this CT scan?" in remainder:
                    text = remainder.split("region of this CT scan?")[-1].strip()
        
        # Clean up any trailing or leading whitespace
        text = text.strip()
        
        return text

    def run_inference_from_dataset(self, dataset_path, split="train", num_samples=5, filter_type=None):
        """
        Run inference on samples from dataset.
        
        Args:
            dataset_path: Path to the dataset JSON
            split: Dataset split to use
            num_samples: Number of samples to run inference on
            filter_type: Optional filter for "qa" or "region" question types
        """
        try:
            # Load dataset using the AMOSMMDataset class
            dataset = AMOSMMDataset(
                dataset_json_path=dataset_path,
                model_name="qwen",
                split=split,
                embeddings_path=self.embeddings_path
            )
            
            print(f"Loaded {len(dataset)} samples from {split} split")
            
            # Filter by question type if requested
            if filter_type:
                subset = []
                for i in range(len(dataset)):
                    example = dataset[i]
                    # Check if it's a QA question or region question based on format
                    is_qa = any("The answer is" in msg.get("content", [{}])[0].get("text", "") 
                               for msg in example if isinstance(msg, dict) and msg.get("role") == "assistant")
                    if (filter_type == "qa" and is_qa) or (filter_type == "region" and not is_qa):
                        subset.append(example)
                
                print(f"Filtered to {len(subset)} {filter_type} samples")
            else:
                subset = [dataset[i] for i in range(min(num_samples, len(dataset)))]
            
            # Limit to num_samples
            if len(subset) > num_samples:
                subset = subset[:num_samples]
            
            results = []
            for i, example in enumerate(tqdm(subset, desc=f"Running inference on {split} split")):
                try:
                    # Process the example
                    inputs, user_query, embedding_path = self.process_example(example)
                    
                    # Generate response
                    response = self.generate(inputs)
                    
                    # Get ground truth response if available
                    ground_truth = None
                    for msg in example:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            content = msg.get("content")
                            if isinstance(content, list) and len(content) > 0:
                                ground_truth = content[0].get("text", None)
                                break
                    
                    # Store results
                    results.append({
                        "index": i,
                        "query": user_query,
                        "response": response,
                        "ground_truth": ground_truth,
                        "embedding_path": embedding_path,
                        "example": example  # Store full example for reference
                    })
                    
                    # Print results
                    print(f"\nSample {i+1}/{len(subset)}")
                    print(f"Query: {user_query}")
                    print(f"Response: {response}")
                    if ground_truth:
                        print(f"Ground Truth: {ground_truth}")
                    print("-" * 80)
                    
                except Exception as e:
                    print(f"Error processing sample {i}: {e}")
                    continue
                
            # Save results to file
            timestamp = int(time.time())
            output_file = f"inference_results_{split}_{timestamp}.json"
            with open(output_file, "w") as f:
                json.dump(results, f, indent=2)
            
            print(f"Results saved to {output_file}")
            return results
            
        except Exception as e:
            print(f"Error in run_inference_from_dataset: {e}")
            import traceback
            traceback.print_exc()
            return []

            
    def _modify_example_for_test_mode(self, example):
        """
        Modify an example to hide answer for test mode.
        
        Args:
            example: The original formatted example
            
        Returns:
            Modified example with answer removed completely
        """
        import copy
        modified = copy.deepcopy(example)
        
        # For true test mode, we need to completely remove the assistant's answer
        # Rather than modifying it, we'll remove it entirely from the example 
        # so the model has to generate the answer from scratch
        
        # Keep only the system and user messages
        test_example = []
        for msg in modified:
            if isinstance(msg, dict) and msg.get("role") in ["system", "user"]:
                test_example.append(msg)
        
        return test_example
            
    def create_custom_prompt_with_formatter(self, image_id, prompt_type, custom_text=None):
        """
        Create a custom prompt for a specific image using formatters.
        
        Args:
            image_id: ID of the image (used to construct embedding path)
            prompt_type: Type of prompt ('region' or 'qa')
            custom_text: Custom text for prompt
            
        Returns:
            Formatted prompt
        """
        # Construct embedding path
        embedding_path = str(Path(self.embeddings_path) / f"{image_id}_embedding.npz")
        
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding not found: {embedding_path}")
        
        # Use the formatters from data.formatters
        format_region_question = get_formatter("qwen_amos_regions")
        format_qa_data = get_formatter("qwen_amos_qa")
        
        if prompt_type == "region":
            region = "abdomen"  # Default region
            if custom_text:
                # Extract region from custom text
                for r in ["chest", "abdomen", "pelvis"]:
                    if r in custom_text.lower():
                        region = r
                        break
            
            return format_region_question(
                embedding_path,
                region,
                ""  # No findings provided since we're generating them
            )
        elif prompt_type == "qa":
            # Create a dummy QA pair
            qa_pair = {
                "question": custom_text or "Describe what you see in this CT scan.",
                "options": {"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"},
                "answer": "A"  # Dummy answer
            }
            return format_qa_data(embedding_path, qa_pair)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen2-VL model on AMOS dataset")
    parser.add_argument("--checkpoint", type=str, 
                        default="/home/m340g/Projects/rocket-vlm/qwen2-2b-amos-mm/checkpoint-3050",
                        help="Path to fine-tuned checkpoint")
    parser.add_argument("--base_processor", type=str, 
                        default="Qwen/Qwen2-VL-2B-Instruct",
                        help="Base model ID to use for the processor")
    parser.add_argument("--dataset", type=str, 
                        default="/home/m340g/DrivesDKFZ/E132-Projekte/Projects/2024_MICCAI24_AMOS-MM/dataset.json", 
                        help="Path to the dataset JSON file")
    parser.add_argument("--split", type=str, default="val", 
                        help="Dataset split to use (train, val)")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of samples to process")
    parser.add_argument("--question_types", nargs="+", choices=["qa", "region"], default=None,
                        help="Filter for specific question types (can specify multiple)")
    parser.add_argument("--hide_answers", action="store_true", default=False,
                        help="Hide answers in ground truth display (for evaluation)")
    parser.add_argument("--test_mode", action="store_true", default=True,
                        help="Run in test mode: hide answers in prompts to evaluate model performance")
    parser.add_argument("--custom_prompt", type=str, default=None,
                        help="Custom prompt text to use with --custom_image")
    parser.add_argument("--custom_image", type=str, default=None,
                        help="Image ID to run custom inference on (without file extension)")
    parser.add_argument("--custom_type", type=str, choices=["qa", "region"], default="region",
                        help="Type of prompt to create for custom image")
    parser.add_argument("--output_dir", type=str, default=".", 
                        help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize inference with both checkpoint and base processor model
    inference = VLMInference(args.checkpoint, args.base_processor)
    
    if args.custom_image:
        # Run custom inference on a specific image
        try:
            print(f"Running custom inference on image ID: {args.custom_image}")
            example = inference.create_custom_prompt_with_formatter(
                args.custom_image,
                args.custom_type,
                args.custom_prompt
            )
            
            if example:
                inputs, user_query, embedding_path = inference.process_example(example)
                response = inference.generate(inputs)
                
                print(f"\nQuery: {user_query}")
                print(f"Response: {response}")
                
                # Save result to file
                timestamp = int(time.time())
                output_file = os.path.join(args.output_dir, f"custom_inference_{timestamp}.json")
                with open(output_file, "w") as f:
                    json.dump({
                        "image_id": args.custom_image,
                        "query": user_query,
                        "response": response,
                        "prompt_type": args.custom_type
                    }, f, indent=2)
                
                print(f"Result saved to {output_file}")
            else:
                print("Failed to create custom prompt")
        except Exception as e:
            print(f"Error in custom inference: {e}")
    else:
        # Run inference using dataset
        print(f"Running inference on dataset: {args.dataset}")

        results = inference.run_test_mode(
            args.dataset,
            split=args.split,
            num_samples=args.num_samples,
            question_types=args.question_types
        )
        print(f"Completed inference on {len(results)} samples")


if __name__ == "__main__":
    main()