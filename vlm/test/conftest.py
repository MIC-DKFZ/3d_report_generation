"""Provide portable defaults for import-time configuration roots during tests."""

import os
from pathlib import Path

os.environ.setdefault(
    "VLM_CONFIG_DIR", str(Path(__file__).resolve().parents[2] / "paper_configs/RMR")
)
os.environ.setdefault("VLM_OUTPUT_DIR", "/tmp/vlm-test-results")
