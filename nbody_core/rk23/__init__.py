"""Compatibility wrapper for diffusion_sim.integrators.rk23."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.integrators.rk23 import run_rk23_dynamic

__all__ = ["run_rk23_dynamic"]
