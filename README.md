# Running instructions
## Setup
For installing the requirements, run
```
pip install -e .
```

## Data Location

### Datasets
* AMOS-MM dataset: ``E132-Projekte/Projects/2024_MICCAI24_AMOS-MM``
* LIDC-IDRI dataset: ``E132-Projekte/Projects/2025_MIC_Rocket_VLM/LIDC-IDRI``. For information how to generate this VQA dataset, see [here](data/lidc/README.md)

### Embeddings
* All image embeddings are located in ``E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings``
    * AMOS-MM reshaped ``E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings/192_reshaped/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/2024_MICCAI24_AMOS-MM``
    * AMOS-MM sliding window ``E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings/192_sliding_window/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/2024_MICCAI24_AMOS-MM``
    * LIDC-IDRI sliding window ``E132-Projekte/Projects/2025_MIC_Rocket_VLM/vision_embeddings/LIDC-IDRI/192_sliding_window/EvaMAETrainer__nnsslPlans__mask_ratio07__vit_patch_size888__embed_dim864__encoder_eva_depth24__encoder_eva_numheads12__decoder_eva_depth8__decoder_eva_numheads12__batch_size8__initial_lr00003__attention_drop_rate02__3d_fullres/npz``

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
