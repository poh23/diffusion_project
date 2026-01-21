"""Compatibility wrapper for diffusion_sim.integrators.dop853."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.integrators.dop853 import run_dop853_chunked

__all__ = ["run_dop853_chunked"]
