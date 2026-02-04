from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_R_FLOOR = 1e-12


@dataclass
class SimulationConfig:
    # system/physics
    n_particles: int = 10
    k: float = 1.0
    v0: float = 1.0
    l: float = 1.0
    r_floor: float = DEFAULT_R_FLOOR
    init_radius: float = 1.0
    # integration / time
    dt: float = 0.01
    steps: int = 200
    t0: float = 0.0
    method: str = "rk4"  # "rk4" or "rk2" or "rk23" or "dop853"
    seed: int = 0
    out_format: str = "npz"
    # progress / chunking
    chunk_steps: int = 0
    print_every_chunks: int = 1
    # dop853 tolerances
    rtol: float = 1e-6
    atol: float = 1e-6
    # rk23 adaptive settings
    first_step: float | None = None
    max_step_global: float = np.inf
    eta: float = 0.05
    recompute_every: int = 10
    rk23_sample_dt: float | None = None
    rk23_sample_count: int | None = None
    rk23_status_every_steps: int | None = 50
    rk23_status_every_sec: float | None = 1.0
    # stochastic diffusion (rk23 only)
    diffusion: bool = False
    diffusion_coeff: float = 0.0
    diffusion_seed: int | None = None
    diffusion_noise_var: float = 1.0
    # optional density computation (post-process, SciPy Voronoi)
    compute_density: bool = False
    density_print_every: int | None = None
    density_stride: int = 1  # compute every n steps when density is enabled


def load_config_file(config_path: Path) -> SimulationConfig:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "out_format" not in data and "out-format" in data:
        data["out_format"] = data["out-format"]

    allowed_keys = set(SimulationConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in allowed_keys}
    unknown = set(data.keys()) - allowed_keys
    if unknown:
        print(f"Warning: ignoring unknown config keys: {sorted(unknown)}")

    return SimulationConfig(**filtered)


def validate_config(config: SimulationConfig) -> None:
    errors = []

    if config.n_particles <= 0:
        errors.append("n_particles must be > 0")
    if config.dt <= 0.0:
        errors.append("dt must be > 0")
    if config.steps <= 0:
        errors.append("steps must be > 0")
    if config.r_floor < 0.0:
        errors.append("r_floor must be >= 0")
    # k=0 is supported (log potential), but keep other checks intact
    if config.init_radius <= 0.0:
        errors.append("init_radius must be > 0")
    if config.method not in ("rk2", "rk4", "rk23", "dop853"):
        errors.append("method must be one of: rk2, rk4, rk23, dop853")
    if config.out_format not in ("npz", "h5", "hdf5"):
        errors.append("out_format must be one of: npz, h5, hdf5")
    if config.density_stride <= 0:
        errors.append("density_stride must be >= 1")

    if config.rk23_sample_dt is not None and config.rk23_sample_dt <= 0.0:
        errors.append("rk23_sample_dt must be > 0 when set")
    if config.rk23_sample_count is not None and config.rk23_sample_count < 0:
        errors.append("rk23_sample_count must be >= 0 when set")
    if config.rk23_status_every_steps is not None and config.rk23_status_every_steps < 0:
        errors.append("rk23_status_every_steps must be >= 0 when set")
    if config.rk23_status_every_sec is not None and config.rk23_status_every_sec < 0.0:
        errors.append("rk23_status_every_sec must be >= 0 when set")
    if config.diffusion_coeff < 0.0:
        errors.append("diffusion_coeff must be >= 0")
    if config.diffusion_noise_var < 0.0:
        errors.append("diffusion_noise_var must be >= 0")
    if config.diffusion and config.method != "rk23":
        errors.append("diffusion is only supported with method='rk23'")

    if errors:
        raise ValueError("Invalid SimulationConfig: " + "; ".join(errors))
