"""Evaluate report pairs with the optional RadGraph scorer."""

import importlib.metadata
import os
from pathlib import Path


def evaluate_reports(
    preds: list[str], gts: list[str], **kwargs
) -> dict[str, tuple[float, list[float]]]:
    from radgraph import F1RadGraph

    VLM_RADGRAPH_CACHE_DIR = os.getenv("VLM_RADGRAPH_CACHE_DIR")
    model_cache_dir = None
    if VLM_RADGRAPH_CACHE_DIR is not None:
        model_cache_dir = Path(VLM_RADGRAPH_CACHE_DIR, importlib.metadata.version("radgraph"))

    f1radgraph = F1RadGraph(
        reward_level="all", model_type="radgraph-xl", model_cache_dir=model_cache_dir
    )
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(
        hyps=preds, refs=gts
    )

    rg_e, rg_er, rg_bar_er = mean_reward
    _, rg_er_list, _ = reward_list

    return {
        "f1_radgraph": (rg_er.item(), rg_er_list)
    }
