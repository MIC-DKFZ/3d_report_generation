import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import torch.distributed as dist

from vlm.utils import logger
from vlm.utils.serialization import load_json


def format_bytes(num_bytes: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def format_rate(num_bytes: int, seconds: float) -> str:
    if seconds <= 0:
        return "inf"
    return f"{format_bytes(int(num_bytes / seconds))}/s"


def get_local_staging_root() -> Path:
    hostname = subprocess.getoutput("hostname").lower()
    systemname = os.environ.get("SYSTEMNAME", "").lower()

    if "jureca" in hostname or "jureca" in systemname:
        return Path("/local/nvme")

    env_path = os.environ.get("VLM_LOCAL_DATASET_STAGING_DIR")
    if env_path:
        return Path(env_path)

    raise RuntimeError(
        "Could not determine local dataset staging directory. "
        "Set VLM_LOCAL_DATASET_STAGING_DIR or add a case for your system in get_local_staging_root()."
    )


def get_local_staging_workers() -> int:
    value = os.environ.get("VLM_LOCAL_DATASET_STAGING_WORKERS", "4")
    try:
        workers = int(value)
    except ValueError as exc:
        raise ValueError("VLM_LOCAL_DATASET_STAGING_WORKERS must be an integer") from exc
    if workers < 1:
        raise ValueError("VLM_LOCAL_DATASET_STAGING_WORKERS must be >= 1")
    return workers


def get_local_vision_input_path() -> Path:
    job_id = os.environ.get("SLURM_JOB_ID")
    suffix = job_id if job_id else f"manual_{os.getpid()}"
    user = os.environ.get("USER", "unknown_user")
    return get_local_staging_root() / user / suffix / "vision_input"


def regular_file_bytes(path: Path) -> tuple[int, int]:
    file_count = 0
    total_bytes = 0
    for dirpath, _, filenames in os.walk(path):
        file_count += len(filenames)
        for filename in filenames:
            total_bytes += Path(dirpath, filename).stat().st_size
    return file_count, total_bytes


def extract_tar(shard_path: Path, dst_dir: Path) -> int:
    subprocess.run(
        ["tar", "-C", str(dst_dir), "-xf", str(shard_path)],
        check=True,
    )
    return shard_path.stat().st_size


def get_shard_paths_to_extract(
    src: Path,
    image_ids: Iterable[str] | None,
    manifest_name: str,
) -> list[Path]:
    if image_ids is None:
        return sorted(src.rglob("*.tar"))

    manifest_path = src / manifest_name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Shard manifest does not exist: {manifest_path}")

    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise TypeError(f"Shard manifest must be a JSON object: {manifest_path}")

    missing_image_ids = sorted({image_id for image_id in image_ids if image_id not in manifest})
    if missing_image_ids:
        preview = missing_image_ids[:20]
        suffix = "" if len(missing_image_ids) <= len(preview) else " ..."
        raise KeyError(
            f"{len(missing_image_ids)} image IDs are missing from shard manifest "
            f"{manifest_path}: {preview}{suffix}"
        )

    shard_rel_paths = sorted({manifest[image_id] for image_id in image_ids})
    shard_paths = [src / shard_rel_path for shard_rel_path in shard_rel_paths]
    missing_shards = [path for path in shard_paths if not path.is_file()]
    if missing_shards:
        preview = [str(path) for path in missing_shards[:20]]
        suffix = "" if len(missing_shards) <= len(preview) else " ..."
        raise FileNotFoundError(f"{len(missing_shards)} manifest shards are missing: {preview}{suffix}")

    return shard_paths


def extract_sharded_vision_input(
    src: Path,
    dst: Path,
    workers: int,
    image_ids: Iterable[str] | None,
    manifest_name: str,
) -> tuple[int, int]:
    shard_paths = get_shard_paths_to_extract(
        src=src,
        image_ids=image_ids,
        manifest_name=manifest_name,
    )
    if not shard_paths:
        raise FileNotFoundError(f"No tar shards found under sharded vision input path: {src}")
    logger.info_main(f"  selected shards: {len(shard_paths)}")

    tasks: list[tuple[Path, Path]] = []
    for shard_path in shard_paths:
        relative_parent = shard_path.parent.relative_to(src)
        dst_dir = dst / relative_parent
        dst_dir.mkdir(parents=True, exist_ok=True)
        tasks.append((shard_path, dst_dir))

    total_shard_bytes = 0
    completed_shards = 0
    start = time.perf_counter()
    last_log = start
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(extract_tar, shard_path, dst_dir) for shard_path, dst_dir in tasks]
        for future in as_completed(futures):
            shard_bytes = future.result()
            total_shard_bytes += shard_bytes
            completed_shards += 1
            now = time.perf_counter()
            if completed_shards == len(tasks) or completed_shards % 25 == 0 or now - last_log >= 60:
                elapsed = now - start
                logger.info_main(
                    "  extracted shards: "
                    f"{completed_shards}/{len(tasks)}, "
                    f"source bytes {format_bytes(total_shard_bytes)}, "
                    f"time {elapsed:.2f}s, throughput {format_rate(total_shard_bytes, elapsed)}"
                )
                last_log = now

    return len(tasks), total_shard_bytes


def copy_one(src_root: Path, dst_root: Path, src_path: Path) -> int:
    dst_path = dst_root / src_path.relative_to(src_root)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, dst_path)
    return src_path.stat().st_size


def image_id_from_unsharded_path(path: Path) -> str:
    suffix = "_embedding.bl2"
    if path.name.endswith(suffix):
        return path.name[: -len(suffix)]
    return path.stem


def copy_unsharded_vision_input(
    src: Path,
    dst: Path,
    workers: int,
    image_ids: Iterable[str] | None,
) -> tuple[int, int]:
    files = [path for path in sorted(src.rglob("*")) if path.is_file()]
    if image_ids is not None:
        image_id_set = set(image_ids)
        files = [path for path in files if image_id_from_unsharded_path(path) in image_id_set]
    if not files:
        raise FileNotFoundError(f"No files found under vision input path: {src}")
    logger.info_main(f"  selected files: {len(files)}")

    copied_bytes = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for file_bytes in executor.map(lambda path: copy_one(src, dst, path), files):
            copied_bytes += file_bytes

    return len(files), copied_bytes


def stage_vision_input_to_local(
    vision_input_path: str | Path,
    vision_input_is_sharded: bool,
    image_ids: Iterable[str] | None = None,
    manifest_name: str = "shard_manifest.json",
    do_copy: bool = True,
) -> Path:
    src = Path(vision_input_path)
    dst = get_local_vision_input_path()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    requested_image_ids = set(image_ids) if image_ids is not None else None

    if do_copy and local_rank == 0:
        workers = get_local_staging_workers()
        if not src.is_dir():
            raise NotADirectoryError(f"Vision input source directory does not exist: {src}")
        if dst.exists():
            raise FileExistsError(f"Local vision input staging directory already exists: {dst}")

        dst.mkdir(parents=True, exist_ok=False)

        logger.info_main("Staging vision input to node-local storage")
        logger.info_main(f"  source: {src}")
        logger.info_main(f"  dest: {dst}")
        logger.info_main(f"  sharded: {vision_input_is_sharded}")
        logger.info_main(f"  workers: {workers}")
        if requested_image_ids is not None:
            logger.info_main(f"  requested image IDs: {len(requested_image_ids)}")

        start = time.perf_counter()
        if vision_input_is_sharded:
            item_count, source_bytes = extract_sharded_vision_input(
                src=src,
                dst=dst,
                workers=workers,
                image_ids=requested_image_ids,
                manifest_name=manifest_name,
            )
            item_label = "shards"
        else:
            item_count, source_bytes = copy_unsharded_vision_input(
                src=src,
                dst=dst,
                workers=workers,
                image_ids=requested_image_ids,
            )
            item_label = "files"
        seconds = time.perf_counter() - start

        staged_files, staged_bytes = regular_file_bytes(dst)
        logger.info_main(
            "Finished staging vision input: "
            f"{item_count} {item_label}, source bytes {format_bytes(source_bytes)}, "
            f"staged files {staged_files}, staged bytes {format_bytes(staged_bytes)}, "
            f"time {seconds:.2f}s, throughput {format_rate(source_bytes, seconds)}"
        )

    if do_copy and dist.is_available() and dist.is_initialized():
        dist.barrier()

    if not dst.is_dir():
        raise FileNotFoundError(f"Local staged vision input directory is missing after staging: {dst}")

    return dst
