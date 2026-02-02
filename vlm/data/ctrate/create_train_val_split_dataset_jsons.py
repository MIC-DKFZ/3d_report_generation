import argparse
from pathlib import Path
from loguru import logger
from vlm.utils.load_and_save_files import load_json, save_json


def _get_patient_id(s):
    return "_".join(s.split("_")[:2])

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_dataset_json", type=str)
    parser.add_argument("--split_ids_json", type=str)

    args = parser.parse_args()

    train_dataset_json_path = Path(args.train_dataset_json)
    parent_dir = train_dataset_json_path.parent

    all_train_dataset_json = load_json(train_dataset_json_path)
    split_ids_json = load_json(args.split_ids_json)

    train_patient_ids = set(split_ids_json["train"])
    val_patient_ids = set(split_ids_json["val"])

    shared_patient_ids = train_patient_ids & val_patient_ids
    assert not shared_patient_ids, f"Leak detected: {shared_patient_ids}"

    new_train_dataset_json = []
    new_val_dataset_json = []
    for dic in all_train_dataset_json:
        patient_id = _get_patient_id(dic["image_id"])
        if patient_id  in train_patient_ids:
            new_train_dataset_json.append(dic)
        elif patient_id in val_patient_ids:
            new_val_dataset_json.append(dic)
        else:
            raise ValueError(f"Patient ID not found in either split: {patient_id}")

    new_train_dataset_json.sort(key=lambda d: d["image_id"])
    new_val_dataset_json.sort(key=lambda d: d["image_id"])

    save_json(new_train_dataset_json, parent_dir / "ctrate_train.json")
    save_json(new_val_dataset_json, parent_dir / "ctrate_val.json")

    logger.info(f"Successfully saved the new dataset JSONs in {parent_dir}")

if __name__ == "__main__":
    main()
