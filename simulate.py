"""Compatibility wrapper for the refactored diffusion_sim package."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.cli import main
from diffusion_sim.simulation import run_simulation
from diffusion_sim.io.npz import save_npz

__all__ = ["run_simulation", "save_npz", "main"]


if __name__ == "__main__":
    main()
