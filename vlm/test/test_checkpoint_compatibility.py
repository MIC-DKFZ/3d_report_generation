import pytest
import torch
from safetensors.torch import save_file

from vlm.configs.compatibility import normalize_saved_config
from vlm.configs.data_config import DataConfig
from vlm.configs.general_config import GeneralConfig
from vlm.training.checkpoint_state import load_model_checkpoint


class CheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.language_model = torch.nn.Module()
        self.model.language_model.base_model = torch.nn.Linear(2, 2, bias=False)
        self.model.vision_adapter = torch.nn.Linear(2, 2, bias=False)


def test_obsolete_eval_sample_limit_is_ignored_without_mutating_input():
    original = {"eval": {"n_samples": None, "max_new_tokens": 512}}
    normalized, warnings = normalize_saved_config(original)
    assert normalized == {"eval": {"max_new_tokens": 512}}
    assert original["eval"]["n_samples"] is None
    assert any("eval.n_samples" in warning for warning in warnings)


@pytest.mark.parametrize("value", [False, True])
def test_obsolete_image_resize_flag_is_ignored(value):
    original = {"data": {"dataset_name": "ctrate", "answer_types": "all", "do_resize": value}}
    normalized, warnings = normalize_saved_config(original)
    assert "do_resize" not in DataConfig.model_validate(normalized["data"]).model_dump()
    assert any("data.do_resize" in warning for warning in warnings)


@pytest.mark.parametrize("value", [False, True])
def test_obsolete_optimizer_initialization_flag_is_ignored(value):
    original = {"general": {"init_optimizer_scheduler_from_checkpoint": value, "seed": 3}}
    normalized, warnings = normalize_saved_config(original)
    assert GeneralConfig.model_validate(normalized["general"]).seed == 3
    assert any("general.init_optimizer_scheduler_from_checkpoint" in w for w in warnings)


@pytest.mark.parametrize("dataset_name", ["ctrate", "merlin", "bonn", "amosmm", "toothfairy4"])
@pytest.mark.parametrize(
    "retrieval_path,use_images", [(None, False), ("retrieval.json", False), ("retrieval.json", True)]
)
def test_matching_saved_dataset_class_is_removed(dataset_name, retrieval_path, use_images):
    data = {
        "dataset_name": dataset_name,
        "answer_types": "report",
        "retrieval_json_path": retrieval_path,
        "retrieval_use_image_in_prompt": use_images,
    }
    class_name = DataConfig.model_validate(data).get_dataset_class().__name__
    original = {"data": {**data, "dataset_class_name": class_name}}
    normalized, warnings = normalize_saved_config(original)
    assert normalized == {"data": data}
    assert original["data"]["dataset_class_name"] == class_name
    assert any("data.dataset_class_name" in warning for warning in warnings)


def test_null_saved_dataset_class_is_removed():
    original = {"data": {"dataset_class_name": None}}
    normalized, warnings = normalize_saved_config(original)
    assert normalized == {"data": {}}
    assert original["data"]["dataset_class_name"] is None
    assert any("data.dataset_class_name" in warning for warning in warnings)


@pytest.mark.parametrize(
    "class_name",
    ["CTRATEDataset", "MerlinDataset", "CTRATEDatasetRetrievalWithImagePrompt1", "UnknownDataset"],
)
def test_obsolete_saved_dataset_class_is_ignored_even_if_conflicting(class_name):
    original = {
        "data": {
            "dataset_name": "ctrate",
            "answer_types": "report",
            "retrieval_json_path": "retrieval.json",
            "retrieval_use_image_in_prompt": True,
            "dataset_class_name": class_name,
        }
    }
    normalized, warnings = normalize_saved_config(original)
    assert "dataset_class_name" not in normalized["data"]
    assert DataConfig.model_validate(normalized["data"]).get_dataset_class().__name__ == (
        "CTRATEDatasetRetrievalWithImage"
    )
    assert any("data.dataset_class_name" in warning for warning in warnings)
    assert original["data"]["dataset_class_name"] == class_name


@pytest.mark.parametrize(
    "name, value, error",
    [
        ("model.language_model.base_model.weight", torch.ones(3, 2), "shape/dtype mismatch"),
        (
            "model.language_model.base_model.weight",
            torch.ones(2, 2, dtype=torch.int64),
            "shape/dtype mismatch",
        ),
        ("model.unknown.weight", torch.ones(2, 2), "does not exist"),
    ],
)
def test_invalid_tensors_are_rejected_before_loading(tmp_path, name, value, error):
    save_file({name: value}, tmp_path / "model.safetensors")
    model = CheckpointModel()
    before = model.model.language_model.base_model.weight.detach().clone()
    with pytest.raises(ValueError, match=error):
        load_model_checkpoint(model, tmp_path)
    assert torch.equal(model.model.language_model.base_model.weight, before)


def test_lower_precision_weights_are_cast_to_the_model_dtype(tmp_path):
    weight = torch.tensor([[1.5, -2.0], [0.25, 3.0]])
    save_file({"model.vision_adapter.weight": weight.bfloat16()}, tmp_path / "model.safetensors")
    model = CheckpointModel()
    load_model_checkpoint(model, tmp_path)
    assert model.model.vision_adapter.weight.dtype == torch.float32
    torch.testing.assert_close(model.model.vision_adapter.weight.detach(), weight)
