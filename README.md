# Running instructions
## Setup
For installing the requirements, run
```
pip install -e .
```

### Local Setup and Environment Variables

1. **Copy the entire folder** located at `$CLUSTER_DATA/j385i/vlm` to your local machine.  
   *(It might take some time due to the vision encoder and some vision data (10 train and 5 val samples for each preprocessed dataset).)*

2. **Update the paths** in the configuration files located in `./configs/bases` (marked with `<TODO>`) according to the directory where you copied the folder.  
   Only the configuration files within `./bases/dataset_configs` and `./bases/model_configs` require modification.

3. **Set the following two environment variables:**

   - **`VLM_CONFIG_DIR`**  
     Should point to `./configs`.  
     The path/file provided after the `--train_config` flag as well as all base configurations (explained below) will be resolved relative to this directory.  

     **Examples:**  
     - `run.yml` → `$VLM_CONFIG_DIR/run.yml`  
     - `lr_ablation/run.yml` → `$VLM_CONFIG_DIR/lr_ablation/run.yml`

   - **`VLM_OUTPUT_DIR`**  
     Can point to any directory where results will be stored (e.g., `./results`).  
     The folder structure in the output folder is as follows:  
     ```
     $VLM_OUTPUT_DIR/<DATASET_NAME>/<VISION_INPUT_FOLDER_NAME>/(<TRAIN_CONFIG_PATH> + '__' + <DATETIME_TAG> + '__' + <JOB_ID/UID>)
     ```

     **Examples:**  
     Training with `dataset_name=ctrate` and `vision_input_path=/p/scratch/hai_1021/CTRATEv2_LungCrop_1.0x1.0x1.0`:
     - `run.yml` → `$VLM_OUTPUT_DIR/ctrate/CTRATEv2_LungCrop_1.0x1.0x1.0/run__2025-10-06_20-56-58__12733926`
     - `lr_ablation/run.yml` → `$VLM_OUTPUT_DIR/lr_ablation/ctrate/CTRATEv2_LungCrop_1.0x1.0x1.0/lr_ablation/run__2025-10-06_20-56-58__12733926`
    
## Training, Inference & Evaluation

To run a training, execute
```
python train.py --train_config <relative_path_to_config_file>
```

### Start a Run with Config Files

Three basic example YAML files are provided to start a test training run:

- `Qwen_ADAPTER_Qwen.yml` *(adapter-only)*
- `Qwen_ADAPTER_Qwen_LM_Lora.yml` *(frozen VE, LM lora, HEAD lora)*
- `Qwen_VE_Lora_ADAPTER_Qwen_LM_Lora.yml` *(adapter fft, the rest with lora)*

Each final configuration file consists of five main sections and an additional `base_configs` field:

- `base_configs` — list of config file paths to merge into the main config file (earlier configs have higher priority)
- `general` — *(GeneralConfig)*
- `model` — *(ModelConfig)*
- `data` — *(DataConfig)*
- `trainer` — *(SFTTrainerConfig)*
- `eval` — *(EvaluationConfig)*

### Base Configurations

The `base_configs` field allows combining multiple configuration files into a single final training configuration.  
You can also specify extra configurations using the `--add_base_configs` CLI flag.  
Configurations passed this way are inserted at the beginning of the `base_configs` list, taking precedence.

This can be useful if you want to omit the dataset config in your run configurations (and pass them via `--add_base_configs`) so they can be reused for all datasets (nnUNet-style).  
This is the case for all three example configurations.

To maximize flexibility, you can create **separate configuration files** for each subdictionary (especially for `model`, `data`, and `trainer`) and use them as base configurations to construct the final training configuration.  
Each of these components has a dedicated directory:

- `./bases/model_configs`
- `./bases/dataset_configs`
- `./bases/trainer_configs`

Files within these directories define parameters specific to their respective sections.

If you would like to understand how configuration files are processed internally, what default values exist, or which arguments are supported, take a look at `vlm/configs/run_config.py`

Shared configuration helpers live alongside the schemas in `vlm/configs`: `cli.py` handles
dotted CLI overrides, `merging.py` merges configuration dictionaries, and `compatibility.py`
migrates known legacy fields in saved run configurations.

### The Evaluation Configuration
The `EvaluationConfig` manages both the validation callback during training and the **automatic inference and evaluation after training**.
To enable both during training, set `eval.run_inference` and `eval.run_eval` to `True`.
More parameters and detailed information can be found in the `EvaluationConfig` class definition.

### Overriding Specific Fields

You can also override individual fields in the training configuration directly from the command line (CLI).
To do this, add a flag that points to the field you want to modify — relative to RunConfig — followed by the new value.

Example: the following command overrides the number of training epochs and the number of validation samples used during training:

```
python train.py --train_config Qwen_ADAPTER_Qwen.yml \
    --add_base_configs bases/dataset_configs/ctrate_lungcrop_embedding.yml --trainer.num_train_epochs 5 --data.n_val_samples 50
```

**WARNING**: Overriding fields such as `data.dataset_name`, `data.vision_input_path` may result in unexpected behavior, since they are used to determine the output folder even before
the override process takes place within the train configuration. Some fields such as `run_name` or `output_dir` which are determined internally must not be provided or overwritten as well.
	 

### The Easy Way to Run Ablations

If you have a working configuration that you want to run ablations with, you can use it as a base config and just override 
the one value that you want to ablate (and optionally the 'wandb_project' argument). 
Additionally, if you group the ablation configurations in a seperate folder, this structure will be forwarded to the output folder 
(as explained above in [Setup for Local Debugging](#local-setup)).


### Running Experiments on JSC

Example run commands can be found in `_cluster/sweep_launch.sh`. 
This script internally submits jobs using the `_cluster/vlm_train.sh` sbatch script.
Any flags you pass to the sbatch command are forwarded directly to `train.py`. In the provided examples, 
this is used to specify the dataset configuration via the `--add_base_configs` flag.
You can also pass any `general`, `model`, `data`, etc. arguments in the same way, 
these will override the corresponding values in the final configuration ([Overriding Specific Fields](#overriding-specific-fields)).

The example run commands should work out of the box after you configure the SBATCH parameters and update the necessary variables in the section marked `CONFIGURABLE`.

The default configuration directory is located at `/p/scratch/hai_1023/rocket_vlm/configs/rocket_default`.
Create a copy of the folder next to it to have your own configuration directory.

#### Symlinks for Vision Input Paths

When using different preprocessed datasets or embeddings from various vision encoders, 
each run should use a distinct `VISION_INPUT_FOLDER_NAME` to ensure that result folders are stored in separate directories.
The `/p/scratch/hai_1023/rocket_vlm/data/symlinks` directory contains symbolic links to the corresponding image or embedding folders, 
named clearly to indicate their contents.

The resulting directory structure of `$VLM_OUTPUT_DIR` would look like this:

```
$VLM_OUTPUT_DIR
├── amosmm
│   └── ...
└── ctrate
    ├── CTRATEv2_LungCrop_1.0x1.0x1.0_Primus_MAE
    ├── CTRATEv2_LungCrop_1.0x1.0x1.0_Primus_Contrastive
    ├── CTRATEv2_LungCrop_1.0x1.0x1.0_ResEncL_MAE
    └── CTRATEv2_LungCrop_1.0x1.0x1.0_ResEncL_Contrastive 
```

---

### Debug Mode

You can add the `debug.yml` file as a base config, which overrides arguments such as `batch_size` or `eval_steps` to debug-friendly values.
This is very convenient if you want to start a quick test run for a new configuration or after you made changes to the codebase.

**Example run command for debugging:**

```
python train.py --train_config Qwen_ADAPTER_Qwen.yml \  
    --add_base_configs debug.yml bases/dataset_configs/ctrate_lungcrop_embedding.yml
```

---

## 🛠️ TODO: Inference
To run inference, execute
```
python inference.py --model_type <model_type> --dataset_path <path_to_dataset> --embeddings_path <path_to_embeddings>
```
* Current supported model types: "qwen"
* Current supported datasets with embeddings: See [Data Location](#data-location)
---

## 🛠️ TODO: Evaluation
To run evaluation, execute
```
python eval.py --model_type <model_type> --dataset_path <path_to_dataset> --embeddings_path <path_to_embeddings>
```
* Current supported model types: "qwen"
* Current supported datasets with embeddings: See [Data Location](#data-location)

---

## Data Location

> **TODO:** Update this section regarding new CTRATE paths and train-val-test split numbers.

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

---

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
* For training, the logs are saved to the `$VLM_OUTPUT_DIR` as `train.log`
* For logging the loss, a separate logging level is introduced
    ```
    logger.log("LOSS", f"Step: {global_step} | Training Loss: {loss:.5f}")
    ```
  
### Logging with DDP
When using the default `logger` functions under DDP, each rank writes to the log file, producing duplicate entries.  
Although this may be desirable in some cases, logging is typically handled by the **main rank only**.  
To only log via the main rank, use the same logging functions with a `_main` suffix, for example, `logger.info_main(...)`.
For more details, look at the `MainRankLoggingProtocol` at `utils/logging.py`.
