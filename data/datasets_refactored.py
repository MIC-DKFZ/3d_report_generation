import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# new dataset combining datasets.py, formatters.py, extractors.py hopefully without causing trouble

class MedicalQADataset(Dataset):
    """
    Unified dataset class for medical imaging QA tasks
    
    Args:
        dataset_json_path (str): Path to the dataset JSON file
        model_name (str): Name of the model (e.g., 'qwen', 'owl')
        split (str): Split name ('train', 'val', or 'test')
        embeddings_path (str | Path): Root path to the vision embeddings directory
        n_samples (int, optional): Number of samples to load (for debugging)
        tokenize_fn (callable, optional): Function to tokenize text
    """
    
    def __init__(
        self, 
        dataset_json_path, 
        model_name, 
        split="train", 
        embeddings_path=None,
        n_samples=None, 
        tokenize_fn=None
    ):
        self.model_name = model_name
        self.split = split
        self.embeddings_path = Path(embeddings_path) if isinstance(embeddings_path, str) else embeddings_path
        self.n_samples = n_samples
        self.tokenize_fn = tokenize_fn
        
        # Set default embeddings path if not provided
        if embeddings_path is None:
            self.embeddings_path = Path(os.environ.get("ROCKET_VLM_VISION_EMBEDDINGS"))
        
        # Load data from JSON
        self.samples = self._load_data(dataset_json_path)
        
        # Tokenize if function provided
        if self.tokenize_fn:
            self._tokenize_samples()
    
    def _load_data(self, dataset_json_path):
        """Load and process data from JSON file"""
        # Determine file path based on split
        file_path = os.path.join(dataset_json_path, f"{self.split}_questions.json")
        
        # Load JSON data
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Limit samples if n_samples is set
        if self.n_samples:
            data = data[:self.n_samples]
        
        processed_samples = []
        
        for item in data:
            # Get image path and embedding path
            image_path = item["image"] if "image" in item else None
            
            # Handle different ID fields
            subject_id = item.get("subject_id", "")
            scan_id = item.get("scan_id", "")
            series_uid = item.get("series_uid", subject_id)
            
            # Determine embedding path TODO add a dataset identifier and use it here
            if "series_uid" in item:
                embedding_path = self._get_embedding_path_lidc(item["series_uid"])
            else:
                embedding_path = self._get_embedding_path_amos(item["image"])
            
            # Skip if embedding file doesn't exist
            if not os.path.exists(embedding_path):
                continue
            
            # Format question and answer
            question = item["question"]
            answer = item["answer"]
            answer_type = item.get("answer_type", "CLOSED")
            content_type = item.get("content_type", "")
            region = item.get("region", "")
            
            # Create conversation format based on model type
            qa_text, qa_text_prompt_only = self._format_conversation(
                question, answer, embedding_path, answer_type, content_type, region
            )
            
            # Create sample
            sample = {
                "image_path": image_path,
                "embedding_path": embedding_path,
                "subject_id": subject_id,
                "scan_id": scan_id,
                "series_uid": series_uid,
                "qa_text": [qa_text],
                "qa_text_prompt_only": [qa_text_prompt_only],
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
                "content_type": content_type,
                "region": region
            }
            
            processed_samples.append(sample)
        
        return processed_samples
    
    def _get_embedding_path_amos(self, image_path):
        """Convert AMOS image path to embedding path"""
        base_name = image_path.replace(".nii.gz", "")
        return str(os.path.join(self.embeddings_path, f"{base_name}_embedding.npy"))
    
    def _get_embedding_path_lidc(self, series_uid):
        """Convert LIDC series_uid to embedding path"""
        base_name = series_uid.replace(".", "_")
        return str(Path(self.embeddings_path) / f"{base_name}_embedding.npz")
    
    def _format_conversation(self, question, answer, embedding_path, answer_type, content_type, region):
        """Format conversation based on model type"""

        system_text = "You are a medical imaging AI trained to answer questions about CT scans."
        
        # Create user message
        if self.model_name == "qwen":
            user_message = {
                "role": "user",
                "content": [
                    {"type": "image", "embedding_path": embedding_path},
                    {"type": "text", "text": question}
                ]
            }
        elif self.model_name == "owl":
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": f"<|image|>\n{question}"}]
            }
        else:
            raise ValueError(f"Unsupported model type: {self.model_name}")
        
        # Create assistant message
        assistant_message = {
            "role": "assistant",
            "content": [{"type": "text", "text": answer}]
        }
        
        # Create system message
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": system_text}]
        }
        
        # Full conversation and prompt-only conversation
        full_conversation = [system_message, user_message, assistant_message]
        prompt_only_conversation = [system_message, user_message]
        
        return full_conversation, prompt_only_conversation
    
    def _tokenize_samples(self):
        """Tokenize text samples if tokenize_fn is provided"""
        for sample in self.samples:
            sample["tokenized"] = {}
            sample["tokenized"]["qa_text_tokens"] = [self.tokenize_fn(text) for text in sample["qa_text"]]
            sample["tokenized"]["qa_text_prompt_only_tokens"] = [
                self.tokenize_fn(text) for text in sample["qa_text_prompt_only"]
            ]
    
    def _load_embedding(self, embedding_path):
        """Load embedding from file"""
        if embedding_path.endswith(".npy"):
            return np.load(embedding_path)
        elif embedding_path.endswith(".npz"):
            npz_file = np.load(embedding_path)
            # Attempt to load the "embeddings" key if it exists, otherwise load the first array
            if "embedding" in npz_file:
                return npz_file["embedding"]
            else:
                return npz_file[npz_file.files[0]]
        else:
            raise ValueError(f"Unsupported embedding file format: {embedding_path}")
    
    def _embedding_2_tensor(self, embedding):
        """Convert embedding to tensor format"""
        np_img_flat = embedding.reshape(embedding.shape[0], embedding.shape[1], -1)
        np_img_flat = np_img_flat.transpose(0, 2, 1)
        return torch.tensor(np_img_flat)
    
    def __getitem__(self, index):
        """Get dataset item by index"""
        sample = self.samples[index]
        
        # Create a new dictionary to avoid modifying the stored one
        result = dict(sample)
        result["embedding"] = self._load_embedding(sample["embedding_path"])
        result["embedding_tensor"] = self._embedding_2_tensor(result["embedding"])
        
        return result
    
    def __len__(self):
        """Get dataset length"""
        return len(self.samples)
    
    def verify(self):
        """Test function to verify dataset loading"""
        print(f"\nTotal number of samples: {len(self)}")
        
        if len(self.samples) > 0:
            sample = self.samples[0]
            print(f"\nExample question: {sample['question']}")
            print(f"Answer: {sample['answer']}")
            print(f"Content type: {sample['content_type']}")
            if sample.get('region'):
                print(f"Region: {sample['region']}")