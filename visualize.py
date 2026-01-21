"""Compatibility wrapper for diffusion_sim.plotting."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.plotting import (
    animate_mp4,
    embed_mp4,
    plot_density_vs_radius,
    plot_scaled_density_vs_radius,
    plot_energy,
    plot_std,
    save_mp4,
)
from diffusion_sim.io.npz import load_npz

__all__ = [
    "animate_mp4",
    "embed_mp4",
    "plot_density_vs_radius",
    "plot_scaled_density_vs_radius",
    "plot_energy",
    "plot_std",
    "save_mp4",
    "load_npz",
]


if __name__ == "__main__":
    sim = load_npz(r"data\31-12-2025_k3_soft0.1_dt1e_N_comparison\dop853_k3.0_N1000_steps500_dt0.001_softening0.1_seed1_with_density.npz")
    plot_density_vs_radius(sim, times=[0, sim["times"][1], sim["times"][2], sim["times"][3], sim["times"][-1]], window_frac=0.02)
