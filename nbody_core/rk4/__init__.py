"""Compatibility wrapper for diffusion_sim.integrators.rk4."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.integrators.rk4 import rk4_step_numba, run_rk4_loop

__all__ = ["rk4_step_numba", "run_rk4_loop"]
