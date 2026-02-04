import argparse
import time
from dataclasses import replace
from pathlib import Path

from .config import SimulationConfig, load_config_file
from .io.h5 import save_h5
from .io.npz import save_npz
from .simulation import run_simulation


def _parse_args():
    parser = argparse.ArgumentParser(description="Run diffusion simulation and save NPZ output.")
    parser.add_argument("--config", type=Path, help="JSON config file matching SimulationConfig fields.")
    parser.add_argument("--out", type=Path, help="Output .npz path (default: data/sim_<timestamp>.npz).")
    parser.add_argument(
        "--out-format",
        choices=["npz", "h5", "hdf5"],
        default=None,
        help="Output format for default naming (npz, h5, or hdf5).",
    )

    parser.add_argument("--n-particles", type=int)
    parser.add_argument("--k", type=float)
    parser.add_argument("--v0", type=float)
    parser.add_argument("--l", type=float)
    parser.add_argument("--r-floor", type=float, help="Hard distance floor for interactions.")
    parser.add_argument("--init-radius", type=float, help="Initial disk radius for particle placement.")
    parser.add_argument("--dt", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--t0", type=float)
    parser.add_argument("--method", choices=["rk2", "rk4", "rk23", "dop853"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chunk-steps", type=int)
    parser.add_argument("--print-every-chunks", type=int)
    parser.add_argument("--rtol", type=float, help="Relative tolerance (rk23/dop853).")
    parser.add_argument("--atol", type=float, help="Absolute tolerance (rk23/dop853).")
    parser.add_argument("--first-step", type=float, help="Initial step guess (rk23).")
    parser.add_argument("--max-step-global", type=float, help="Global max step size (rk23).")
    parser.add_argument("--eta", type=float, help="Safety factor for rk23 distance cap.")
    parser.add_argument("--recompute-every", type=int, help="Recompute rk23 distance cap every N steps.")
    parser.add_argument("--rk23-sample-dt", type=float, help="Sample interval for rk23 output (optional).")
    parser.add_argument("--rk23-sample-count", type=int, help="Number of samples for rk23 output (optional).")
    parser.add_argument("--rk23-status-every-steps", type=int, help="Update rk23 progress status every N accepted steps.")
    parser.add_argument("--rk23-status-every-sec", type=float, help="Update rk23 progress status every N seconds.")
    parser.add_argument(
        "--diffusion",
        action="store_true",
        default=None,
        help="Enable stochastic diffusion step (rk23 only).",
    )
    parser.add_argument("--diffusion-coeff", type=float, help="Diffusion constant D for rk23 diffusion.")
    parser.add_argument("--diffusion-seed", type=int, help="RNG seed for rk23 diffusion.")
    parser.add_argument("--diffusion-noise-var", type=float, help="Gaussian noise variance for rk23 diffusion.")
    parser.add_argument(
        "--compute-density",
        action="store_true",
        default=None,
        help="Compute Voronoi density/radii (post-process).",
    )
    parser.add_argument("--density-print-every", type=int, help="Print progress every N steps during density computation.")
    parser.add_argument("--density-stride", type=int, help="Compute density every N steps (default 1 = every step).")
    return parser.parse_args()


def _config_from_args(args) -> SimulationConfig:
    config = SimulationConfig()
    if args.config:
        config = load_config_file(args.config)

    overrides = {
        "n_particles": args.n_particles,
        "k": args.k,
        "v0": args.v0,
        "l": args.l,
        "r_floor": args.r_floor,
        "init_radius": args.init_radius,
        "dt": args.dt,
        "steps": args.steps,
        "t0": args.t0,
        "method": args.method,
        "seed": args.seed,
        "chunk_steps": args.chunk_steps,
        "print_every_chunks": args.print_every_chunks,
        "rtol": args.rtol,
        "atol": args.atol,
        "first_step": args.first_step,
        "max_step_global": args.max_step_global,
        "eta": args.eta,
        "recompute_every": args.recompute_every,
        "rk23_sample_dt": args.rk23_sample_dt,
        "rk23_sample_count": args.rk23_sample_count,
        "rk23_status_every_steps": args.rk23_status_every_steps,
        "rk23_status_every_sec": args.rk23_status_every_sec,
        "diffusion": args.diffusion,
        "diffusion_coeff": args.diffusion_coeff,
        "diffusion_seed": args.diffusion_seed,
        "diffusion_noise_var": args.diffusion_noise_var,
        "compute_density": args.compute_density,
        "density_print_every": args.density_print_every,
        "density_stride": args.density_stride,
        "out_format": args.out_format,
    }

    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        config = replace(config, **overrides)

    return config


def main():
    args = _parse_args()
    config = _config_from_args(args)

    sim = run_simulation(**config.__dict__)

    if args.out:
        out_path = args.out
    else:
        date_dir = time.strftime("%Y%m%d")
        diffusion_tag = ""
        if config.diffusion:
            diffusion_tag = f"_diff_D{config.diffusion_coeff}"
        stem = (
            f"{config.method}_N{config.n_particles}_steps{config.steps}_"
            f"dt{config.dt}_k{config.k}_rtol{config.rtol}_atol{config.atol}{diffusion_tag}"
        )
        suffix = ".npz" if config.out_format == "npz" else f".{config.out_format}"
        out_path = Path("data") / date_dir / f"{stem}{suffix}"

    if out_path.suffix.lower() in {".h5", ".hdf5"}:
        save_h5(out_path, sim)
    else:
        save_npz(out_path, sim)


if __name__ == "__main__":
    main()
