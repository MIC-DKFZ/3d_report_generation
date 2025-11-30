import argparse
from loguru import logger
from vlm.utils.load_and_save_files import load_json, save_json


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--in_dataset_json", type=str)
    parser.add_argument("--out_dataset_json", type=str)
    parser.add_argument("--answer_subtype", type=str)

    args = parser.parse_args()

    in_dataset_json = load_json(args.in_dataset_json)
    logger.info(f"Number of samples before filtering: {len(in_dataset_json)}")

    out_dataset_json = [d for d in in_dataset_json if d["answer_subtype"] == args.answer_subtype]
    out_dataset_json.sort(key=lambda d: d["image_id"])
    logger.info(f"Number of samples after filtering: {len(out_dataset_json)}")

    save_json(out_dataset_json, args.out_dataset_json)
    logger.info(f"Successfully saved the new dataset JSON at {args.out_dataset_json}")

if __name__ == "__main__":
    main()
