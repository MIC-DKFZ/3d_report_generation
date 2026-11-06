"""
ROI masks of NIfTI volumes made with TotalSegmentator, for inference on volumes without masks.

The runs cropped their inputs to TotalSegmentator 'total' masks: the masks shipped with CT-RATE
('ts_seg/ts_total') and those made for Merlin, both with TotalSegmentator v2 (117 labels). The
masks made here use the same task, with its default settings (not '--fast') and all labels in
one file ('--ml'), so that they can be cropped to like the training masks.

TotalSegmentator is an optional dependency, 'pip install TotalSegmentator==2.18.0'; its weights
are downloaded on first use into 'TOTALSEG_HOME_DIR' (default '~/.totalsegmentator'), or
beforehand with 'totalseg_download_weights -t total' for machines without internet access.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from vlm.data.preprocessing.nifti_inputs import strip_nifti_suffix
from vlm.utils import logger

TASK = "total"
# The ROI masks that can be made here, as recorded in 'data.vision_input_preprocessing.roi_mask'.
SUPPORTED_ROI_PREFIX = "TotalSegmentator 'total'"


def can_generate(roi_mask: str | None) -> bool:
    return roi_mask is not None and roi_mask.startswith(SUPPORTED_ROI_PREFIX)


def settings() -> dict:
    """The TotalSegmentator settings of the masks made here, for the saved inference settings."""
    try:
        installed = version("TotalSegmentator")
    except PackageNotFoundError:
        installed = None
    return {"version": installed, "task": TASK, "fast": False, "ml": True}


def generate_missing_masks(
    input_dir: Path, files: list[Path], mask_dir: Path, device: str = "gpu"
) -> dict[str, str]:
    """
    Make the masks of the volumes that have none in 'mask_dir' (same relative paths as in
    'input_dir'), and return for each volume whether its mask was 'read' or 'generated'.
    Volumes that TotalSegmentator fails on are logged and left without a mask ('failed').
    """
    sources = {}
    missing = [path for path in files if not (mask_dir / path.relative_to(input_dir)).is_file()]
    for path in files:
        sources[strip_nifti_suffix(path.relative_to(input_dir))] = "read"
    if not missing:
        return sources

    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as error:
        raise ImportError(
            f"{len(missing)} of {len(files)} volumes have no mask in {mask_dir}, and making them "
            "needs TotalSegmentator: install it with 'pip install TotalSegmentator==2.18.0', or "
            "pass the masks of all volumes with '--mask_dir'."
        ) from error

    logger.info(f"Making {len(missing)} masks with TotalSegmentator '{TASK}' in {mask_dir}.")
    for path in missing:
        relative_path = path.relative_to(input_dir)
        image_id = strip_nifti_suffix(relative_path)
        mask_path = mask_dir / relative_path
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        # Written under another name first, so that an interrupted run leaves no partial mask
        # that a later run would read.
        partial_path = mask_path.with_name(f".partial_{mask_path.name}")
        try:
            totalsegmentator(path, partial_path, ml=True, task=TASK, device=device, quiet=True)
            partial_path.rename(mask_path)
            sources[image_id] = "generated"
        except Exception as error:  # Skip the volume instead of stopping the run.
            partial_path.unlink(missing_ok=True)
            logger.error(f"TotalSegmentator failed on {path}: {type(error).__name__}: {error}")
            sources[image_id] = "failed"
    return sources
