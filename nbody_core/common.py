"""Compatibility wrapper for diffusion_sim forces/metrics/init."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.forces import compute_velocity_overdamped
from diffusion_sim.metrics import compute_energy_numba, compute_std_numba
from diffusion_sim.init_conditions import init_positions_jittered_disk

__all__ = [
    "compute_velocity_overdamped",
    "compute_energy_numba",
    "compute_std_numba",
    "init_positions_jittered_disk",
]
