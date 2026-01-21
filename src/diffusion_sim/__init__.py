"""Top-level package for diffusion simulation."""

from .config import SimulationConfig, DEFAULT_R_FLOOR, load_config_file, validate_config
from .simulation import run_simulation
from .io.npz import load_npz, save_npz

__all__ = [
    "SimulationConfig",
    "DEFAULT_R_FLOOR",
    "load_config_file",
    "validate_config",
    "run_simulation",
    "load_npz",
    "save_npz",
]
