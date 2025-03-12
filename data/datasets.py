import json
import os
from pathlib import Path

from torch.utils.data import Dataset
from data.formatters import get_formatter

SSL_MODEL_NAME = "EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres"

class AMOSMMDataset(Dataset):
    """
    AMOS-MM dataset class
    Args:
        dataset_json_path (str): Path to the dataset JSON file
        model_name (str): Name of the vlm the dataset is used for. Used to infere the formatter for the questions
        split (str): Split name ('train' or 'val')
        embeddings_path (str | Path): Root path to the vision embeddings directory
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
        self.embeddings_path = Path(embeddings_path) / "192_reshaped" / SSL_MODEL_NAME / "2024_MICCAI24_AMOS-MM"
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
        return str(self.embeddings_path / f"{base_name}_embedding.npy")

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


class LIDCDataset(Dataset):
    """
    LIDC dataset class
    Args:
        dataset_json_path (str): Path where the dataset jsons are located.
            Should contain train_questions.json, val_questions.json, and test_questions.json
        model_name (str): Name of the vlm the dataset is used for. Used to infere the formatter for the questions
        split (str): Split name ('train', 'val', or 'test')
        embeddings_path (str | Path): Path to the vision embeddings directory
    """
    def __init__(self,
                 dataset_json_path,
                 model_name,
                 split='train',
                 embeddings_path=Path(os.getenv("ROCKET_VLM_VISION_EMBEDDINGS"))):
        super(LIDCDataset).__init__()
        self.format_qa_data = get_formatter(f"{model_name}_lidc_qa")
        self.split = split
        if self.split == "train":
            dataset_json_path = os.path.join(dataset_json_path, "train_questions.json")
        elif self.split == "val":
            dataset_json_path = os.path.join(dataset_json_path, "val_questions.json")
        elif self.split == "test":
            dataset_json_path = os.path.join(dataset_json_path, "test_questions.json")
        else:
            raise ValueError(f'Invalid split name {self.split}')
        with open(dataset_json_path, 'r') as f:
            self.data = json.load(f)
        self.embeddings_path = Path(embeddings_path) / "LIDC-IDRI" / "192_sliding_window" / SSL_MODEL_NAME / "npz"
        print(self.embeddings_path)
        print(self.embeddings_path.exists())
        self.samples = self.format_data()

    def get_embedding_path(self, series_uid):
        """Convert image path to embedding path"""
        base_name = series_uid.replace(".", "_")
        return str(Path(self.embeddings_path) / f"{base_name}_embedding.npz")

    def format_data(self):
        formatted_data = []
        for sample in self.data:
            embedding_path = self.get_embedding_path(sample['series_uid'])

            # Skip if embedding file doesn't exist
            if not os.path.exists(embedding_path):
                continue
                #raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

            # Format QA pairs
            formatted_qa = self.format_qa_data(embedding_path, sample)
            formatted_data.append(formatted_qa)
        return formatted_data

    def verify(self):
        """Test function to verify dataset loading"""
        print(f"\nTotal number of samples: {len(self)}")

        # Print example of region-specific question
        print("\nExample question:")
        print(self.samples[0])


    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)
