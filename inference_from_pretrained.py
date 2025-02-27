import torch
import os
from pathlib import Path
import argparse
import json
import numpy as np
from tqdm import tqdm
import glob

from Qwen3D import Qwen2VLForConditionalGeneration3D
from processing_3d import Qwen2VLProcessor3D
from data.datasets import AMOSMMDataset


class VLMInference:
    def __init__(self, base_model_id="Qwen/Qwen2-VL-2B-Instruct"):
        """
        Initialize the VLM inference class.
        
        Args:
            base_model_id: Base model ID to use
        """
        self.base_model_id = base_model_id
        self.processor = None
        self.model = None
        
        # Set up embeddings path (same as in training)
        self.embeddings_path = os.environ.get('ROCKET_VLM_VISION_EMBEDDINGS')
        if self.embeddings_path is None:
            raise ValueError("Environment variable ROCKET_VLM_VISION_EMBEDDINGS is not set")
        
        # Load model and processor
        self.setup_model()
    
    def setup_model(self):
        """Load the base model (no checkpoint)"""
        print(f"Loading base model from {self.base_model_id}...")
        
        try:
            # Load processor from base model
            self.processor = Qwen2VLProcessor3D.from_pretrained(self.base_model_id)
            
            # Load base model
            self.model = Qwen2VLForConditionalGeneration3D.from_pretrained(
                self.base_model_id,
                device_map="auto",
                torch_dtype=torch.bfloat16
            )
            
            # Set model to evaluation mode
            self.model.eval()
            print(f"Model loaded successfully")
            
        except Exception as e:
            print(f"Error in setup_model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def process_example(self, example):
        """
        Process a single example from the dataset, similar to the collate_fn in training.
        
        Args:
            example: A single example from the dataset
            
        Returns:
            inputs: Processed inputs ready for the model
            user_query: The text query from the user
        """
        try:
            # Extract text and embedding path according to the format in collate_fn
            # Apply chat template to the first element of the example
            text = self.processor.apply_chat_template(example[0], tokenize=False)
            
            # Get embedding path from the second element
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
            for msg in example[0]:
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
            
            # Extract the model's response
            if "ASSISTANT:" in generated_text:
                response = generated_text.split("ASSISTANT:")[-1].strip()
            else:
                response = generated_text
            
            return response
            
        except Exception as e:
            print(f"Error during generation: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"

    def run_inference_from_dataset(self, dataset_path, split="train", num_samples=5):
        """
        Run inference on samples from dataset.
        
        Args:
            dataset_path: Path to the dataset JSON
            split: Dataset split to use
            num_samples: Number of samples to run inference on
        """
        try:
            # Load dataset using the same approach as in the training script
            dataset = AMOSMMDataset(
                dataset_json_path=dataset_path,
                model_name="qwen",
                split=split,
                embeddings_path=self.embeddings_path
            )
            
            print(f"Loaded {len(dataset)} samples from {split} split")
            
            # Take a subset if needed
            if len(dataset) > num_samples:
                subset_indices = list(range(min(num_samples, len(dataset))))
                subset = [dataset[i] for i in subset_indices]
            else:
                subset = dataset
            
            results = []
            for i, example in enumerate(tqdm(subset, desc=f"Running inference on {split} split")):
                try:
                    # Process the example
                    inputs, user_query, embedding_path = self.process_example(example)
                    
                    # Generate response
                    response = self.generate(inputs)
                    
                    # Get ground truth response if available
                    ground_truth = None
                    for msg in example[0]:
                        if isinstance(msg, dict) and msg.get("role") == "assistant":
                            content = msg.get("content")
                            if content:
                                ground_truth = content
                                break
                    
                    # Store results
                    results.append({
                        "index": i,
                        "query": user_query,
                        "response": response,
                        "ground_truth": ground_truth,
                        "embedding_path": embedding_path
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
                
            return results
            
        except Exception as e:
            print(f"Error in run_inference_from_dataset: {e}")
            import traceback
            traceback.print_exc()
            return []

    def run_inference_on_embedding(self, embedding_path, user_query, system_prompt=None):
        """
        Run inference on a specific embedding file.
        
        Args:
            embedding_path: Path to the embedding file
            user_query: Text query from user
            system_prompt: Optional system prompt
            
        Returns:
            The model's response
        """
        try:
            # Create messages in the same format as the dataset
            messages = []
            
            # Add system prompt if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message with image reference
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_query},
                    {"type": "image", "content": [{"embedding_path": embedding_path}]}
                ]
            })
            
            # Apply chat template
            text = self.processor.apply_chat_template(messages, tokenize=False)
            
            # Process inputs
            inputs = self.processor(
                text=text,
                vision_embedding_paths=[embedding_path],
                return_tensors="pt",
                padding=True
            )
            
            # Move to same device as model
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            response = self.generate(inputs)
            
            return response
            
        except Exception as e:
            print(f"Error running inference on embedding: {e}")
            import traceback
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def find_embedding_files(self, directory=None, extension=None):
        """
        Find all embedding files in a directory.
        
        Args:
            directory: Directory to search, defaults to imagesVal in embeddings_path
            extension: File extension to search for (.npy or .pt)
            
        Returns:
            List of embedding file paths
        """
        if directory is None:
            # Default to imagesVal directory
            directory = os.path.join(self.embeddings_path, "imagesTr")
            
        
        print(f"Searching for embedding files in: {directory}")
        
        # Find all embedding files
        embedding_files = []
        
        # If extension is specified, search only for that extension
        if extension:
            pattern = os.path.join(directory, f"**/*{extension}")
            embedding_files = glob.glob(pattern, recursive=True)
        else:
            # Otherwise, search for both .npy and .pt files
            npy_files = glob.glob(os.path.join(directory, "**/*.npy"), recursive=True)
            pt_files = glob.glob(os.path.join(directory, "**/*.pt"), recursive=True)
            embedding_files = npy_files + pt_files
        
        print(f"Found {len(embedding_files)} embedding files")
        
        # Print a few examples
        if embedding_files:
            print("Examples:")
            for path in embedding_files[:3]:
                print(f"  - {path}")
        
        return embedding_files

    def run_inference_on_val_directory(self, val_dir=None, file_extension=None, num_samples=None, query=None, system_prompt=None):
        """
        Run inference on all embedding files in the validation directory.
        
        Args:
            val_dir: Validation directory, defaults to imagesVal in embeddings_path
            file_extension: File extension to search for (.npy or .pt)
            num_samples: Number of samples to process, defaults to all
            query: Text query to use
            system_prompt: System prompt to use
            
        Returns:
            List of results
        """
        # Find all embedding files
        embedding_files = self.find_embedding_files(val_dir, file_extension)
        
        if not embedding_files:
            print("No embedding files found in the validation directory")
            return []
        
        # Limit to num_samples if specified
        if num_samples and num_samples < len(embedding_files):
            embedding_files = embedding_files[:num_samples]
        
        # Default query and system prompt
        if not query:
            query = "Describe what you can see in this medical image in detail."
        
        if not system_prompt:
            system_prompt = "You are a helpful medical AI assistant that analyzes medical images."
        
        # Run inference on each file
        results = []
        for i, embedding_path in enumerate(tqdm(embedding_files, desc="Running inference on validation images")):
            try:
                # Generate response
                response = self.run_inference_on_embedding(embedding_path, query, system_prompt)
                
                # Store results
                results.append({
                    "index": i,
                    "embedding_path": embedding_path,
                    "query": query,
                    "response": response
                })
                
                # Print results
                print(f"\nSample {i+1}/{len(embedding_files)}")
                print(f"Embedding: {embedding_path}")
                print(f"Response: {response}")
                print("-" * 80)
                
            except Exception as e:
                print(f"Error processing embedding {embedding_path}: {e}")
                continue
        
        # Save results to file
        timestamp = torch.cuda.current_timestamp() if hasattr(torch.cuda, 'current_timestamp') else int(time.time())
        output_file = f"inference_results_{timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Run inference with Qwen2-VL model")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen2-VL-2B-Instruct",
                        help="Base model ID")
    parser.add_argument("--dataset", type=str, default=None, 
                        help="Path to the dataset JSON file")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (train, val, test)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process")
    parser.add_argument("--embedding_path", type=str, help="Path to a specific embedding file for single inference")
    parser.add_argument("--val_dir", type=str, help="Directory containing validation embeddings")
    parser.add_argument("--query", type=str, default="Describe what you see in this medical image.", 
                        help="Query for inference")
    parser.add_argument("--system_prompt", type=str, 
                        default="You are a helpful medical AI assistant that analyses medical images.", 
                        help="System prompt")
    parser.add_argument("--file_extension", type=str, help="File extension to search for (.npy or .pt)")
    
    args = parser.parse_args()
    
    # Initialize inference with just the base model (no checkpoint loading)
    inference = VLMInference(args.base_model)
    
    # Run single inference if embedding_path provided
    if args.embedding_path:
        print(f"Running single inference on {args.embedding_path}...")
        response = inference.run_inference_on_embedding(
            args.embedding_path, 
            args.query, 
            args.system_prompt
        )
        print(f"\nQuery: {args.query}")
        print(f"Response: {response}")
    
    # Run on dataset if dataset path provided
    elif args.dataset:
        print(f"Running inference on dataset: {args.dataset}")
        results = inference.run_inference_from_dataset(
            args.dataset, 
            split=args.split, 
            num_samples=args.num_samples
        )
        print(f"Completed inference on {len(results)} samples")
    
    # Run on validation directory
    else:
        print(f"Running inference on validation directory")
        results = inference.run_inference_on_val_directory(
            val_dir=args.val_dir,
            file_extension=args.file_extension,
            num_samples=args.num_samples,
            query=args.query,
            system_prompt=args.system_prompt
        )
        print(f"Completed inference on {len(results)} validation samples")


if __name__ == "__main__":
    import time
    main()