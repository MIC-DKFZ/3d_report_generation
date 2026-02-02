import argparse
from pathlib import Path
from loguru import logger
from vlm.utils.load_and_save_files import load_json, save_json


def _get_patient_id(s):
    return "_".join(s.split("_")[:2])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset_json", type=str)

    args = parser.parse_args()

    train_dataset_json_path = Path(args.train_dataset_json)
    parent_dir = train_dataset_json_path.parent

    train_dataset_json = load_json(train_dataset_json_path)
    new_train_dataset_json = []

    seen_ids = set()
    for d in train_dataset_json:
        patient_id = _get_patient_id(d["image_id"])
        if patient_id in seen_ids:
            continue
        new_train_dataset_json.append(d)
        seen_ids.add(patient_id)

    new_train_dataset_json.sort(key=lambda d: d["image_id"])

    out_json_path = parent_dir / "ctrate_train_one_recon.json"
    save_json(new_train_dataset_json, out_json_path)

    logger.info(f"Successfully saved the new dataset JSON at {out_json_path}")

if __name__ == "__main__":
    main()
