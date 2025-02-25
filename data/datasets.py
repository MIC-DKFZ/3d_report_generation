import json
import os
from pathlib import Path

from torch.utils.data import Dataset
from data.formatters import get_formatter


class AMOSMMDataset(Dataset):
    """
    AMOS-MM dataset class
    Args:
        dataset_json_path (str): Path to the dataset JSON file
        model_name (str): Name of the vlm the dataset is used for. Used to infere the formatter for the questions
        split (str): Split name ('train' or 'val')
        embeddings_path (str): Path to the vision embeddings directory
    """
    def __init__(self,
                 dataset_json_path,
                 model_name,
                 split='train',
                 embeddings_path=Path(os.getenv("ROCKET_VLM_VISION_EMBEDDINGS"))):
        super(AMOSMMDataset).__init__()
        self.format_region_question = get_formatter(f"{model_name}_amos_regions")
        self.format_qa_data = get_formatter(f"{model_name}_amos_qa")
        with open(dataset_json_path, 'r') as f:
            self.data = json.load(f)
        self.split = split
        self.embeddings_path = embeddings_path
        self.samples = self.format_data()

    def format_data(self):
        if self.split == 'train':
            samples = self.data['training']
        elif self.split == 'val':
            samples = self.data['validation']
        else:
            raise ValueError(f'Invalid split name {self.split}')
        formatted_data = []
        for sample in samples:
            embedding_path = self.get_embedding_path(sample['image'])

            # Skip if embedding file doesn't exist
            if not os.path.exists(embedding_path):
                continue
                #raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

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

    def get_embedding_path(self, image_path):
        """Convert image path to embedding path"""
        base_name = image_path.replace('.nii.gz', '')
        return str(Path(self.embeddings_path) / f"{base_name}_embedding.npz")

    def verify(self):
        """Test function to verify dataset loading"""
        print(f"\nTotal number of samples: {len(self)}")

        # Print example of region-specific question
        print("\nExample of region question:")
        for sample in self.samples:
            if "do you see" in sample[1]['content'][1]['text']:
                print(f"Question: {sample[1]['content'][1]['text']}")
                print(f"Answer: {sample[2]['content'][0]['text']}")
                break

        # Print example of original QA
        print("\nExample of original QA:")
        for sample in self.samples:
            if "The answer is" in sample[2]['content'][0]['text']:
                print(f"Question: {sample[1]['content'][1]['text']}")
                print(f"Answer: {sample[2]['content'][0]['text']}")
                break

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)