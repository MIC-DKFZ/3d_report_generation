import pytest
import torch
from torch.torch_version import TorchVersion


@pytest.fixture
def cudnn_sdp(monkeypatch):
    """Record the cuDNN SDP switches instead of changing the global state."""
    state = {"enabled": True, "calls": []}

    def enable_cudnn_sdp(enabled):
        state["calls"].append(enabled)
        state["enabled"] = enabled

    previous = torch.backends.cuda.cudnn_sdp_enabled()
    monkeypatch.setattr(torch.backends.cuda, "cudnn_sdp_enabled", lambda: state["enabled"])
    monkeypatch.setattr(torch.backends.cuda, "enable_cudnn_sdp", enable_cudnn_sdp)
    yield state
    monkeypatch.undo()
    torch.backends.cuda.enable_cudnn_sdp(previous)


def set_device(monkeypatch, version, capability, cuda_available=True):
    monkeypatch.setattr(torch, "__version__", TorchVersion(version))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args: capability)


@pytest.mark.parametrize("capability", [(9, 0), (10, 0)])
def test_cudnn_attention_is_disabled_on_hopper_before_torch_2_15(
    monkeypatch, cudnn_sdp, capability
):
    import train as train_script

    set_device(monkeypatch, "2.14.1+cu132", capability)
    train_script.disable_cudnn_attention_on_hopper_if_affected()
    assert cudnn_sdp["calls"] == [False]


@pytest.mark.parametrize(
    "version, capability, cuda_available, enabled",
    [
        ("2.14.1+cu132", (8, 6), True, True),  # Ampere
        ("2.15.0+cu132", (9, 0), True, True),  # fixed in 2.15
        ("2.14.1+cu132", (9, 0), False, True),  # no GPU
        ("2.14.1+cu132", (9, 0), True, False),  # already disabled
    ],
)
def test_cudnn_attention_is_left_unchanged_otherwise(
    monkeypatch, cudnn_sdp, version, capability, cuda_available, enabled
):
    import train as train_script

    cudnn_sdp["enabled"] = enabled
    set_device(monkeypatch, version, capability, cuda_available)
    train_script.disable_cudnn_attention_on_hopper_if_affected()
    assert cudnn_sdp["calls"] == []
