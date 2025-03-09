import os
import yaml

conditions = [
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening",
]


base_path = "/home/david/mnt/mic_rocket/mic_rocket/data/raw_converted/CTRATE/"

metadata_dir = os.path.join(base_path, "metadata")
data_dir = os.path.join(base_path, "images")

metadata_yml_files = [f for f in os.listdir(metadata_dir) if f.endswith(".yml")]

for metadata_file in metadata_yml_files:
    metadata_path = os.path.join(metadata_dir, metadata_file)
    data_path = os.path.join(data_dir, metadata_file.replace(".yml", ".nrrd"))

    if not os.path.exists(data_path):
        continue

    with open(metadata_path, "r") as f:
        metadata = yaml.safe_load(f)

    print(metadata)

    findings = metadata["metadata"]["Findings_EN"]
    impressions = metadata["metadata"]["Impressions_EN"]
    classifications = {c: metadata["metadata"][c] for c in conditions}

    itm_dict = {
        "image_path": data_path,
        "tasks":
            {
                "classification": classifications,
                "report_gen": findings,
                "diagnosis": impressions
            }

