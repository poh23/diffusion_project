"""Compatibility wrapper for diffusion_sim.io.npz."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.io.npz import load_npz, save_npz

__all__ = ["load_npz", "save_npz"]
