import os
import importlib.metadata
from pathlib import Path
from radgraph import F1RadGraph


def eval_f1_radgraph(preds: list[str], gts: list[str], **kwargs) -> dict[str, list]:
    # batch size is hardcoded to 1, already pretty fast though, so no concerns here

    VLM_RADGRAPH_CACHE_DIR = os.getenv("VLM_RADGRAPH_CACHE_DIR")
    model_cache_dir = None
    if VLM_RADGRAPH_CACHE_DIR is not None:
        model_cache_dir = Path(VLM_RADGRAPH_CACHE_DIR, importlib.metadata.version("radgraph"))

    f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl", model_cache_dir=model_cache_dir)
    mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=preds, refs=gts)

    rg_e, rg_er, rg_bar_er = mean_reward

    return {
        "f1_radgraph": rg_er.item()
    }