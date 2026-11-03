import os
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from vlm.utils.ddp import gather_inference_outputs

FILES = [f"case_{i:02d}.nii.gz" for i in range(7)]
FAILED = {"case_03.nii.gz"}


def split_and_gather_worker(rank, world_size, rendezvous, output_path):
    torch.set_num_threads(1)
    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size))
    dist.init_process_group(
        "gloo",
        rank=rank,
        world_size=world_size,
        init_method=Path(rendezvous).as_uri(),
        timeout=timedelta(seconds=60),
    )
    try:
        # The split and gather of 'inference_nifti.main'.
        results, failed = [], []
        for name in FILES[rank::world_size]:
            record = {"image_id": name.removesuffix(".nii.gz"), "rank": rank}
            (failed if name in FAILED else results).append(record)
        gathered = gather_inference_outputs({"results": results, "failed": failed}, main_rank=0)
        if rank == 0:
            results = sorted(gathered["results"], key=lambda r: r["image_id"])
            failed = sorted(gathered["failed"], key=lambda f: f["image_id"])
            ids = [r["image_id"] for r in results] + [f["image_id"] for f in failed]
            Path(output_path).write_text("\n".join(ids))
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU DDP requires Gloo")
@pytest.mark.parametrize("world_size", [2, 3])
def test_every_volume_is_generated_once_across_ranks(tmp_path, world_size):
    output_path = tmp_path / "ids.txt"
    torch.multiprocessing.start_processes(
        split_and_gather_worker,
        args=(world_size, str(tmp_path / "rendezvous"), str(output_path)),
        nprocs=world_size,
        start_method="spawn",
    )
    ids = output_path.read_text().splitlines()
    expected = [name.removesuffix(".nii.gz") for name in FILES]
    successful = [i for i in expected if f"{i}.nii.gz" not in FAILED]
    assert ids[: len(successful)] == successful  # each once, sorted
    assert ids[len(successful) :] == ["case_03"]
