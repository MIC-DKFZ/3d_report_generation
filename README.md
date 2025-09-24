# Running instructions
## Setup
For installing the requirements, run
```
pip install -e .
```

## Data Location

### AMOS-MM
* Dataset path: `/p/scratch/hai_1021/AMOS-MM`
* Train dataset json: `/p/scratch/hai_1021/AMOS-MM/dataset_jsons/amosmm_train.json`
* Val dataset json: `/p/scratch/hai_1021/AMOS-MM/dataset_jsons/amosmm_val.json`
* 1mm isotropic images in blosq: `/p/scratch/hai_1021/AMOS-MM_1.0x1.0x1.0/`
* Embeddings: `/p/scratch/hai_1023/rocket_vlm/data/AMOS-MM/embeddings/Dataset807_Rocketv2_exclude_CTRATE_val/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth16__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/AMOS-MM_1.0x1.0x1.0/`

#### Statistics of Train and Val dataset json
* Train dataset json: `Answer type distribution: {'MULTIPLE_CHOICE': 13751, 'FINDINGS': 2780, 'IMPRESSIONS': 2240}`
* Val dataset json: `Answer type distribution: {'MULTIPLE_CHOICE': 2787, 'FINDINGS': 883, 'IMPRESSIONS': 729}`

### CTRATEv2

* Dataset path: `/p/scratch/hai_1021/CTRATEv2/CT-RATE/dataset`
* Train dataset json: `/p/scratch/hai_1021/CTRATEv2/CT-RATE/dataset/dataset_jsons/ctrate_train.json`
* Val dataset json: `/p/scratch/hai_1021/CTRATEv2/CT-RATE/dataset/dataset_jsons/ctrate_val.json`
* 1mm isotropic images in blosq: `/p/scratch/hai_1021/CTRATEv2_1.0x1.0x1.0/`
* Lung Crop 1mm isotropic images in blosq: `/p/scratch/hai_1021/CTRATEv2_LungCrop_1.0x1.0x1.0/`
* Embeddings: `/p/scratch/hai_1023/rocket_vlm/data/CTRATEv2/embeddings/Dataset807_Rocketv2_exclude_CTRATE_val/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth16__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/CTRATEv2_1.0x1.0x1.0`
* Lung crop embeddings: `/p/scratch/hai_1023/rocket_vlm/data/CTRATEv2/embeddings/Dataset807_Rocketv2_exclude_CTRATE_val/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth16__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/CTRATEv2_LungCrop_1.0x1.0x1.0`

For dataset json preprocessing (if you want to create the dataset jsons from scratch with `dataset_preprocessing.py`):
* Train error paths (brain images): `/p/scratch/hai_1021/CTRATEv2/CT-RATE/dataset/error_paths/train_error_paths.txt` 
* Val error paths (brain images): `/p/scratch/hai_1021/CTRATEv2/CT-RATE/dataset/error_paths/val_error_paths.txt`

#### Statistics of Train and Val dataset json
* Train dataset json: `Answer type distribution: {'VQA': 865242, 'REPORT_DIAGNOSIS': 46406, 'REPORT': 46406, 'DIAGNOSIS': 46406}`
* Val dataset json `Answer type distribution: {'VQA': 56202, 'REPORT_DIAGNOSIS': 3005, 'REPORT': 3005, 'DIAGNOSIS': 3005}`


    
## Training
To run a training, execute
```
python train.py --model_type <model_type> --dataset_path <path_to_dataset> --embeddings_path <path_to_embeddings>
```
* Current supported model types: "owl" and "qwen"
* Current supported datasets with embeddings: See [Data Location](#data-location)

## Inference
To run inference, execute
```
python inference.py --model_type <model_type> --dataset_path <path_to_dataset> --embeddings_path <path_to_embeddings>
```
* Current supported model types: "owl" and "qwen"
* Current supported datasets with embeddings: See [Data Location](#data-location)

# Coding guidlines
## Code formatting
* Black is used as code formatter
* To setup code formatting in VS Code, add the following to your ``settings.json``

```
"[python]": {
        "editor.formatOnType": true,
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.formatOnSave": true,
        "editor.formatOnPaste": false, 
        "editor.formatOnSaveMode": "file",
        "editor.codeActionsOnSave": {
            "source.organizeImports": "explicit"
        }
    },
"flake8.severity": {
        "E": "Warning",
        "F": "Information",
        "W": "Information"
    },
"flake8.args": ["--max-line-length=99"],
```

## Logging
* Logging is done with loguru
* Add logging statements into the code like this
    ```
    from utils import logger
    logger.info("This is an info")
    ```
* For training and inference, the logs are saved to the `output_dir` as `<yyy>-<mm>-<dd>_<HH>-<MM>-<SS>_train_<uid>.log` (`inference` respectively) 
* For logging the loss, a separate logging level is introduced
    ```
    logger.log("LOSS", f"Step: {global_step} | Training Loss: {loss:.5f}")
    ```