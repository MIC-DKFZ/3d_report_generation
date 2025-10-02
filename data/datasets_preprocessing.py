import json
import os
import sys
from abc import ABC, abstractmethod
from argparse import ArgumentParser
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from data.extractors import AMOSMMExtractor, CTRATEExtractor, LIDCExtractor
from utils import logger


class MedicalImageDatasetPreprocessor(ABC):
    """
    Base class for medical imaging datasets with vision-language models

    Args:
        dataset_path (str): Path to the dataset directory
        split (str): Split name ('train', 'val' or 'test')
        results_file (str): Path to save the processed dataset json file
        max_workers (int, optional): Number of parallel workers for data loading
        overwrite (bool): Whether to overwrite existing preprocess dataset json file
    """

    def __init__(
        self,
        dataset_path,
        split="train",
        results_file=None,
        max_workers=None,
        overwrite=False
    ):
        super().__init__()

        self.dataset_path = Path(dataset_path)
        self.split = split
        self.results_file = results_file
        self.max_workers = max_workers or min(48, (os.cpu_count() or 1))
        logger.info(f"Using {self.max_workers} workers for dataset loading.")

        # Load and process data
        self.data = self._load_data()

        self.samples = self.format_data()

        answer_type_counter = Counter(sample["answer_subtype"] for sample in self.samples)
        logger.info(f"Answer type distribution: {answer_type_counter}")

        if not os.path.exists(self.results_file) or overwrite:
            if os.path.exists(self.results_file):
                logger.warning(f"Overwriting existing results file: {self.results_file}")
            logger.info(f"Writing dataset to {self.results_file}")
            os.makedirs(os.path.dirname(self.results_file), exist_ok=True)
            with open(self.results_file, "w") as f:
                json.dump(self.samples, f, indent=4)
            logger.info(f"Wrote {len(self.samples)} samples.")

    @abstractmethod
    def _load_data(self):
        """Load data from JSON file(s)"""
        pass

    @abstractmethod
    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        pass

    def format_data(self):
        """Format raw data into structured samples with parallel processing"""
        extracted_samples = self.extract_sample_list(self.data)
        
        return extracted_samples

    def __len__(self):
        return len(self.samples)


class AMOSMMDatasetPreprocessor(MedicalImageDatasetPreprocessor):
    """
    AMOS-MM dataset class

    Args:
        dataset_path (str): Path to the dataset directory
        split (str): Split name ('train', 'val' or 'test')
        results_file (str): Path to save the processed dataset json file
        max_workers (int, optional): Number of parallel workers for data loading
        overwrite (bool): Whether to overwrite existing preprocess dataset json file
    """

    def _load_data(self):
        json_file_path = os.path.join(self.dataset_path, "dataset.json")
        with open(json_file_path, "r") as f:
            data = json.load(f)

        if self.split == "train":
            samples = data["training"]
        elif self.split == "val":
            samples = data["validation"]
        else:
            raise ValueError(f"Invalid split name {self.split}")

        return samples

    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        extracted_samples = AMOSMMExtractor.extract_sample_list(samples)
        return extracted_samples


class CTRATEDatasetPreprocessor(MedicalImageDatasetPreprocessor):
    """
    CT-RATE dataset class for CT scan analysis

    Args:
        dataset_path (str): Path to the dataset directory
        split (str): Split name ('train' or 'val')
        results_file (str): Path to save the processed dataset json file
        max_workers (int, optional): Number of parallel workers for data loading
        overwrite (bool): Whether to overwrite existing preprocess dataset json file
        vqa_path (str, optional): Path to the VQA data directory
        error_path_txt (str, optional): Path to txt file with error cases, i.e. brain images
    """
    def __init__(
        self,
        dataset_path,
        split="train",
        results_file=None,
        max_workers=None,
        overwrite=False,
        vqa_path=None,
        error_path_txt=None
    ):
        self.vqa_path = vqa_path
        self.error_path_txt = error_path_txt
        self.exclude_image_ids = ["train_1815_a_4", "train_2597_a_2"]
        super().__init__(
            dataset_path=dataset_path,
            split=split,
            results_file=results_file,
            max_workers=max_workers,
            overwrite=overwrite
        )
    
    @staticmethod
    def _load_csv_row(row_data):
        """Helper function to load a single CSV row for CTRATEDataset"""
        row, split, vqa_path = row_data
        
        volume_name = row['VolumeName']
        # Extract image_id from volume name (e.g., valid_1_a_1.nii.gz -> valid_1_a_1)
        image_id = volume_name.replace('.nii.gz', '')

        # Map CSV columns to expected structure
        findings = row.get('Findings_EN', '')
        impressions = row.get('Impressions_EN', '')
        clinical_info = row.get('ClinicalInformation_EN', '')
        technique = row.get('Technique_EN', '')

        sample = {
            "image_id": image_id,
            "tasks": {
                "classification": {},  # CSV doesn't have classification data
                "report_gen": findings,
                "diagnosis": impressions,
            },
            # Store additional CSV fields if needed
            "clinical_information": clinical_info,
            "technique": technique,
        }
        
        # Load VQA data if vqa_path is provided (reuse existing logic)
        if vqa_path is not None:
            try:
                # Extract case identifier from volume name
                # valid_1_a_1.nii.gz -> valid_1_a
                case_id = '_'.join(image_id.split('_')[:-1])
                
                # Determine VQA folder based on case type
                if case_id.startswith('train_'):
                    vqa_folder = "train_cases"
                elif case_id.startswith('valid_'):
                    vqa_folder = "validation_cases"
                else:
                    vqa_folder = None
                
                if vqa_folder:
                    vqa_file_path = os.path.join(vqa_path, vqa_folder, f"{case_id}.json")
                    
                    if os.path.exists(vqa_file_path):
                        with open(vqa_file_path, "r") as f:
                            vqa_data = json.load(f)
                        sample["vqa_data"] = vqa_data.get("qa_pairs", [])
                    else:
                        logger.debug(f"VQA file not found: {vqa_file_path}")
                            
            except Exception as e:
                logger.error(f"Error loading VQA data for {image_id}: {e}")
        
        return sample

    def _load_data(self):
        """Load data from CSV reports instead of YML files"""
        logger.info("Loading data from CSV reports...")
        
        # Load appropriate CSV file based on split
        reports_dir = os.path.join(self.dataset_path, "radiology_text_reports")
        
        if self.split == "train":
            csv_file = os.path.join(reports_dir, "train_reports.csv")
        elif self.split == "val":
            csv_file = os.path.join(reports_dir, "validation_reports.csv")
        else:
            raise ValueError(f"Invalid split name {self.split} for CSV loading")

        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        logger.info(f"Loading CSV file: {csv_file}")
        df = pd.read_csv(csv_file)
        
        logger.info(f"Processing {len(df)} CSV rows in parallel...")
        
        row_data_list = [(row, self.split, self.vqa_path) 
                        for _, row in df.iterrows()]
        
        # Use ThreadPoolExecutor for parallel processing
        samples = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_row = {executor.submit(self._load_csv_row, row_data): idx 
                        for idx, row_data in enumerate(row_data_list)}
            
            for future in tqdm(as_completed(future_to_row), total=len(row_data_list), desc="Loading CSV rows"):
                result = future.result()
                if result is not None:
                    samples.append(result)

        logger.info(f"Successfully loaded {len(samples)} samples from CSV")
        logger.info(f"Number of samples before filtering: {len(samples)}")
        if self.error_path_txt is not None:
            with open(self.error_path_txt, "r") as f:
                error_paths = f.read().splitlines()
            error_paths = [os.path.basename(p).replace(".nii.gz","") for p in error_paths]
            if self.split == "train":
                error_paths.extend(self.exclude_image_ids)
            samples = [s for s in samples if s["image_id"] not in error_paths]
        logger.info(f"Number of samples after filtering: {len(samples)}")
        return samples

    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        extracted_samples = CTRATEExtractor.extract_sample_list(
            samples
        )
        return extracted_samples

# WARNING: This is currently not in use and untested
class LIDCDatasetPreprocessor(MedicalImageDatasetPreprocessor):
    """
    LIDC dataset class

    Args:
        dataset_json_path (str): Path where the dataset jsons are located
        model_name (str): Name of the VLM the dataset is used for
        split (str): Split name ('train', 'val', or 'test')
        embeddings_path (str | Path): Path to the vision embeddings directory
        n_samples (int, optional): Number of samples to load (for debugging)
    """

    def _load_data(self):
        if self.split == "train":
            file_path = os.path.join(self.dataset_path, "train_questions.json")
        elif self.split == "val":
            file_path = os.path.join(self.dataset_path, "val_questions.json")
        elif self.split == "test":
            file_path = os.path.join(self.dataset_path, "test_questions.json")
        else:
            raise ValueError(f"Invalid split name {self.split}")

        with open(file_path, "r") as f:
            return json.load(f)

    def extract_sample_list(self, samples):
        """Extract sample list from raw data"""
        extracted_samples = LIDCExtractor.extract_sample_list(samples)
        return extracted_samples

def parse_args():
    arg_parser = ArgumentParser(description="Preprocess datasets to json files")
    arg_parser.add_argument("--dataset", type=str, required=True, help="Dataset to preprocess (ctrate, amosmm")
    arg_parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset")
    arg_parser.add_argument("--results_file", type=str, required=True, help="Path to save the processed json file")
    arg_parser.add_argument("--split", type=str, default="train", help="Dataset split to process (train, val)")
    arg_parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing results file")
    arg_parser.add_argument("--vqa_path", type=str, default=None, help="Path to the VQA data directory (for CTRATE only)")
    arg_parser.add_argument("--error_path_txt", type=str, default=None, help="Path to txt file with error cases, i.e. brain images (for CTRATE only)")
    args = arg_parser.parse_args()

    if args.dataset == "ctrate" and args.vqa_path is None:
        arg_parser.error("The --vqa_path argument is required for the CTRATE dataset.")
    elif args.dataset != "ctrate" and args.vqa_path is not None:
        arg_parser.error("The --vqa_path argument is only applicable for the CTRATE dataset.")

    return args


def main():
    logger.remove()
    logger.add(sys.stdout, colorize=True, enqueue=True)
    
    args = parse_args()
    if args.dataset.lower() == "ctrate":
        CTRATEDatasetPreprocessor(
            dataset_path=args.dataset_path,
            results_file=args.results_file,
            split=args.split,
            vqa_path=args.vqa_path,
            error_path_txt=args.error_path_txt,
            overwrite=args.overwrite
        )
    elif args.dataset.lower() == "amosmm":
        AMOSMMDatasetPreprocessor(
            dataset_path=args.dataset_path,
            results_file=args.results_file,
            split=args.split,
            overwrite=args.overwrite
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")


if __name__=="__main__":
    main()
