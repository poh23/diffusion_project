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
    parser.add_argument("--out", type=Path, help="Output .npz/.h5 path (default: data/YYYYMMDD/<auto>.npz).")
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
    parser.add_argument("--t0", type=float)
    parser.add_argument("--t-duration", type=float, help="Total integration time.")
    parser.add_argument(
        "--save-every",
        type=float,
        help="Sample interval for output (optional; required for dop853).",
    )
    parser.add_argument("--method", choices=["rk23", "dop853"])
    parser.add_argument("--seed", type=int)
    parser.add_argument("--rtol", type=float, help="Relative tolerance (rk23/dop853).")
    parser.add_argument("--atol", type=float, help="Absolute tolerance (rk23/dop853).")
    parser.add_argument("--first-step", type=float, help="Initial step guess (rk23).")
    parser.add_argument("--max-step-global", type=float, help="Global max step size (rk23).")
    parser.add_argument("--eta", type=float, help="Safety factor for rk23 distance cap.")
    parser.add_argument("--recompute-every", type=int, help="Recompute rk23 distance cap every N accepted steps.")
    parser.add_argument(
        "--diffusion",
        action="store_true",
        default=None,
        help="Enable stochastic diffusion step (rk23 only).",
    )
    parser.add_argument("--diffusion-coeff", type=float, help="Diffusion constant D for rk23 diffusion.")
    parser.add_argument("--diffusion-seed", type=int, help="RNG seed for rk23 diffusion.")
    parser.add_argument("--diffusion-noise-var", type=float, help="Gaussian noise variance for rk23 diffusion.")
    parser.add_argument("--batch-every", type=float, help="Sim-time interval between batch flushes.")
    parser.add_argument("--target-batch-mb", type=float, help="Target batch size in MB.")
    parser.add_argument("--max-wall-time", type=float, help="Stop after this many seconds (checkpoint and exit).")
    parser.add_argument("--resume-from", type=Path, help="Resume from an existing HDF5 file.")
    parser.add_argument(
        "--resume-force",
        action="store_true",
        default=None,
        help="Resume even if config mismatch.",
    )
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
        "t0": args.t0,
        "t_duration": args.t_duration,
        "save_every": args.save_every,
        "method": args.method,
        "seed": args.seed,
        "rtol": args.rtol,
        "atol": args.atol,
        "first_step": args.first_step,
        "max_step_global": args.max_step_global,
        "eta": args.eta,
        "recompute_every": args.recompute_every,
        "diffusion": args.diffusion,
        "diffusion_coeff": args.diffusion_coeff,
        "diffusion_seed": args.diffusion_seed,
        "diffusion_noise_var": args.diffusion_noise_var,
        "batch_every": args.batch_every,
        "target_batch_mb": args.target_batch_mb,
        "max_wall_time": args.max_wall_time,
        "resume_from": str(args.resume_from) if args.resume_from is not None else None,
        "resume_force": args.resume_force,
        "out_format": args.out_format,
    }

    overrides = {k: v for k, v in overrides.items() if v is not None}
    if overrides:
        config = replace(config, **overrides)

    if args.out_format is None:
        if args.out is not None:
            suffix = args.out.suffix.lower()
            if suffix in {".h5", ".hdf5"}:
                config = replace(config, out_format="h5")
        elif args.resume_from is not None:
            suffix = args.resume_from.suffix.lower()
            if suffix in {".h5", ".hdf5"}:
                config = replace(config, out_format="h5")

    return config


def main():
    args = _parse_args()
    config = _config_from_args(args)

    if config.resume_from is not None:
        out_path = Path(config.resume_from)
    elif args.out:
        out_path = args.out
    else:
        date_dir = time.strftime("%Y%m%d")
        diffusion_tag = ""
        if config.diffusion:
            diffusion_tag = f"_diff_D{config.diffusion_coeff}"
        save_tag = ""
        if config.save_every is not None:
            save_tag = f"_save{config.save_every}"
        stem = (
            f"{config.method}_N{config.n_particles}_t{config.t_duration}_"
            f"k{config.k}_rtol{config.rtol}_atol{config.atol}{save_tag}{diffusion_tag}"
        )
        suffix = ".npz" if config.out_format == "npz" else f".{config.out_format}"
        out_path = Path("data") / date_dir / f"{stem}{suffix}"

    sim = run_simulation(**config.__dict__, out_path=out_path)
    if sim.get("saved_path"):
        return

    if out_path.suffix.lower() in {".h5", ".hdf5"}:
        save_h5(out_path, sim)
    else:
        save_npz(out_path, sim)


if __name__ == "__main__":
    main()
