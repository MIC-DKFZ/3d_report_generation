# CT-RATE Dataset JSONs

`create_dataset_jsons.py` expects the following files in the input/output directory:

- `lung_volumes_train_ctratev2.csv`: Lung-volume measurements for the training split, used to identify low-volume brain scans.
- `lung_volumes_test_ctratev2.csv`: Lung-volume measurements for the test split, used to identify low-volume brain scans.
- `corrupt_scans_ids.txt`: Additional scan IDs to exclude.
- `train_val_split.json`: Patient-level train/validation split applied to the filtered training data.

The script also creates and uses these exclusion files:

- `brain_scans_train_ids.txt`
- `brain_scans_test_ids.txt`
- `exclusion_ids.txt`

All required files and the script outputs are stored in:

`.../e230-thrp-data/mic_rocket/data/raw/CTRATEv2_dataset_jsons`
