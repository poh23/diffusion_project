"""Compatibility wrapper for diffusion_sim.postprocess.density_voronoi."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.postprocess.density_voronoi import (
    compute_density_and_radius_series,
    compute_density_series,
    process_file,
    process_files,
    save_with_density,
    voronoi_density,
)

__all__ = [
    "compute_density_and_radius_series",
    "compute_density_series",
    "process_file",
    "process_files",
    "save_with_density",
    "voronoi_density",
]


if __name__ == "__main__":
    files = [
        r"data\31-12-2025_k3_soft0.1_dt1e_N_comparison\rk4_k3.0_N1000_steps500_dt0.001_softening0.1_seed1.npz",
    ]
    process_files(
        files,
        overwrite=False,
        suffix="_with_density",
        include_radii=True,
        print_every=50,
    )
