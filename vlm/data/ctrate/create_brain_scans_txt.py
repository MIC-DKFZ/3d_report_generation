import argparse
from pathlib import Path
import pandas as pd
from loguru import logger


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--lung_vol_csv", type=str,
                        help="Path to the CSV file containing the filepaths and their corresponding lung volume.")
    parser.add_argument("--min_vol", type=int,
                        help="The minimum lung volume in [mL] required for the image to not be considered a brain scan.")
    parser.add_argument("--output_txt", type=str,
                        help="Absolute path to the output text file containing the image IDs of the brain scans.")

    args = parser.parse_args()

    csv_path = args.lung_vol_csv
    min_vol = args.min_vol
    output_txt = args.output_txt

    data = pd.read_csv(csv_path)
    data["case"] = data["filepath"].apply(lambda p: Path(p).name.replace(".nii.gz", ""))

    brain_image_ids = sorted(data.loc[data['volume_ml'] < min_vol, "case"].values.tolist())

    with open(output_txt, "w") as f:
        for brain_image_id in brain_image_ids:
            f.write(f"{brain_image_id}\n")

    logger.info(f"Successfully saved the brain scans list with {len(brain_image_ids)} image IDs at {output_txt}")

if __name__ == "__main__":
    main()
