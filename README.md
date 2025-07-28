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
* All image embeddings are located in `/p/scratch/hai_1023/rocket_vlm/data/ctrate/embeddings/`
    
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