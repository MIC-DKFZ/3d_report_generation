from types import SimpleNamespace

import pytest

from vlm.models import abstract_vl_wrapper


@pytest.mark.parametrize("implementation", ["flash_attention_2", "flash_attention_3"])
@pytest.mark.parametrize("installed", [True, False])
def test_missing_flash_attention_falls_back_to_sdpa(monkeypatch, implementation, installed):
    monkeypatch.setitem(
        abstract_vl_wrapper.FLASH_ATTENTION_AVAILABLE, implementation, lambda: installed
    )
    model_cfg = SimpleNamespace(attn_implementation=implementation)
    abstract_vl_wrapper.fall_back_to_sdpa_if_flash_attention_is_missing(model_cfg)
    assert model_cfg.attn_implementation == (implementation if installed else "sdpa")


@pytest.mark.parametrize("implementation", ["sdpa", "eager"])
def test_sdpa_and_eager_are_never_changed(monkeypatch, implementation):
    for name in abstract_vl_wrapper.FLASH_ATTENTION_AVAILABLE:
        monkeypatch.setitem(abstract_vl_wrapper.FLASH_ATTENTION_AVAILABLE, name, lambda: False)
    model_cfg = SimpleNamespace(attn_implementation=implementation)
    abstract_vl_wrapper.fall_back_to_sdpa_if_flash_attention_is_missing(model_cfg)
    assert model_cfg.attn_implementation == implementation
