import json
import os
import sys
from sklearn.model_selection import KFold
import numpy as np

# Add the project root to sys.path to allow importing from the 'data' module
# This assumes the script is in 'vlm/plygrnd' and 'data' is in 'vlm/data'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.datasets import CTRATEDataset  # pylint: disable=import-error
from utils import logger  # Assuming utils.logger is available for consistent logging

# --- CONFIGURATION ---
# !!! IMPORTANT: SET THESE PATHS TO YOUR ACTUAL DATASET AND EMBEDDINGS DIRECTORIES !!!
CTRATE_DATASET_ROOT_DIR = "/mnt/mic_rocket/mic_rocket/data/raw_converted/CTRATE/"  # e.g., "/datasets/CTRate"
CTRATE_EMBEDDINGS_DIR = "/mnt/cluster_data/r403k/rocket_vision_language/ctrate/reshaped_flattened"

# Output configuration
OUTPUT_DIR = os.path.dirname(__file__)  # Save JSON in the same directory as the script
OUTPUT_FILENAME = "ctrate_train_5fold_splits.json"
N_FOLDS = 5
RANDOM_STATE = 42  # For reproducible splits
# --- END CONFIGURATION ---


def create_ctrate_splits():
    """
    Loads the CTRATE training dataset, creates 5-fold splits, and saves them to a JSON file.
    """
    if "/path/to/" in CTRATE_DATASET_ROOT_DIR or "/path/to/" in CTRATE_EMBEDDINGS_DIR:
        logger.error("ERROR: Please configure CTRATE_DATASET_ROOT_DIR and CTRATE_EMBEDDINGS_DIR in the script.")
        return

    logger.info(f"Loading CTRATE dataset from: {CTRATE_DATASET_ROOT_DIR}")
    logger.info(f"Using embeddings path: {CTRATE_EMBEDDINGS_DIR}")

    # try:
    # Initialize CTRATEDataset for the 'train' split.
    # 'model_name' is required by the dataset class constructor.
    # 'embeddings_path' is crucial as CTRATEDataset filters samples based on existing embeddings.
    ctrate_train_dataset = CTRATEDataset(
        dataset_json_path=CTRATE_DATASET_ROOT_DIR,
        model_name="qwen",  # Or any other model_name compatible with your setup, e.g., "owl"
        split="train",
        embeddings_path=CTRATE_EMBEDDINGS_DIR,
    )
    # except FileNotFoundError as e:
    #     logger.error(
    #         f"Error initializing CTRATEDataset: {e}. One of the paths might be incorrect or dataset structure is not as expected."
    #     )
    #     logger.error(
    #         "Please ensure CTRATE_DATASET_ROOT_DIR points to the root of the CTRATE dataset (containing 'metadata' subdir),"
    #     )
    #     logger.error("and CTRATE_EMBEDDINGS_DIR points to the directory with precomputed embeddings.")
    #     return
    # except Exception as e:
    #     logger.error(f"An unexpected error occurred while initializing CTRATEDataset: {e}")
    #     return

    if not ctrate_train_dataset.samples:
        logger.warning("No samples found in the CTRATE training set. ")
        logger.warning(
            "This could be due to incorrect paths, missing metadata, missing embeddings, or an empty training set definition in metadata."
        )
        return

    # Extract unique image_ids from the loaded training samples.
    # Each sample in `ctrate_train_dataset.samples` comes from `CTRATEExtractor`,
    # which might produce multiple question-answer items per original image.
    # We need unique image IDs for splitting the images themselves.
    image_ids = sorted(list(set([sample["image_id"] for sample in ctrate_train_dataset.samples])))

    if not image_ids:
        logger.warning("No unique image IDs extracted from the dataset samples. Cannot create splits.")
        return

    logger.info(f"Found {len(image_ids)} unique image IDs in the training set.")

    # Initialize KFold for creating the splits
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    image_ids_array = np.array(image_ids)  # KFold works well with numpy arrays

    splits_data = {}
    logger.info(f"Creating {N_FOLDS}-fold splits...")
    for fold_idx, (train_indices, val_indices) in enumerate(kf.split(image_ids_array)):
        train_ids = image_ids_array[train_indices].tolist()
        val_ids = image_ids_array[val_indices].tolist()

        splits_data[f"fold_{fold_idx}"] = {"train_ids": train_ids, "val_ids": val_ids}
        logger.info(f"Fold {fold_idx}: {len(train_ids)} train IDs, {len(val_ids)} val IDs")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logger.info(f"Output directory created: {OUTPUT_DIR}")
    for fold, ids in splits_data.items():

        # Save the splits data to a JSON file
        output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME.replace(".json", f"_{fold}.json"))
        logger.info(f"Saving {fold} split to: {output_path}")

        try:
            with open(output_path, "w") as f:
                json.dump(ids, f, indent=4)
            logger.info(f"Successfully created and saved {N_FOLDS}-fold split file: {output_path}")
        except IOError as e:
            logger.error(f"Error writing JSON to {output_path}: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred while writing JSON: {e}")

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    logger.info(f"All splits saved to: {output_path}")

    try:
        with open(output_path, "w") as f:
            json.dump(splits_data, f, indent=4)
        logger.info(f"Successfully created and saved {N_FOLDS}-fold split file: {output_path}")
    except IOError as e:
        logger.error(f"Error writing JSON to {output_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred while writing JSON: {e}")


if __name__ == "__main__":
    create_ctrate_splits()
