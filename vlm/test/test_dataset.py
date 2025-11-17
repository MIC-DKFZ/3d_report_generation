import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from vlm.data.datasets import CTRATEDataset
from vlm.utils import logger
from vlm.utils.vision_input_type import VisionInputType


def main_cli():
    arg_parser = argparse.ArgumentParser(description="Check if vision input files exist for samples in a dataset JSON file.")
    arg_parser.add_argument(
        "--json_path",
        type=str,
        required=True,
        help="Path to the dataset JSON file.",
    )
    arg_parser.add_argument(
        "--vision_input_path",
        type=str,
        required=True,
        help="Base path to the vision input files.",
    )
    arg_parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split (e.g., 'train', 'val').",
    )
    args = arg_parser.parse_args()
    return args
    
def main():
    cli_args = main_cli()
    test_dataset_json_sample_exists(cli_args)

def check_sample_missing(dataset, sample):
    vision_input_path = dataset._get_vision_input_path(sample["image_id"])
    if not vision_input_path.exists():
        logger.error(f"Missing: {vision_input_path} for image_id {sample['image_id']}")
        return 1
    return 0


def test_dataset_json_sample_exists(cli_args):
    model_name = "qwen"
    dataset = CTRATEDataset(
            dataset_json_path=cli_args.json_path,
            model_type=model_name,
            split=cli_args.split,
            vision_input_path=cli_args.vision_input_path,
            vision_input_type=VisionInputType.EMBEDDING,
            answer_types=[{"TEXT": ["REPORT_DIAGNOSIS"]}],
    )
    missing_count = 0

    with ThreadPoolExecutor(max_workers=24) as executor:
        futures = {executor.submit(check_sample_missing, dataset, sample): sample for sample in dataset.samples}
        with tqdm(total=len(futures), desc="Processing samples") as pbar:
            for future in as_completed(futures):
                missing_count += future.result()
                pbar.update(1)
    
    if missing_count == 0:
        logger.info("All vision input files exist.")
    else:
        logger.error(f"{missing_count} vision input files are missing.")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, colorize=True, enqueue=True)
    main()
