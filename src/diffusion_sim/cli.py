import argparse
import ast
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

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
    parser.add_argument(
        "--charge-values",
        nargs=2,
        type=float,
        metavar=("Q1", "Q2"),
        help="Charge values for two populations (positive).",
    )
    parser.add_argument(
        "--charge-counts",
        nargs=2,
        type=int,
        metavar=("N1", "N2"),
        help="Particle counts for the two charge populations (sum to n_particles).",
    )
    parser.add_argument("--t0", type=float)
    parser.add_argument("--t-duration", type=float, help="Total integration time.")
    parser.add_argument(
        "--save-every",
        type=float,
        help="Sample interval for output (optional; required for dop853).",
    )
    parser.add_argument(
        "--save-every-steps",
        type=int,
        help="For rk23: record every N accepted steps (mutually exclusive with --save-every).",
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
        "--no-interpolation",
        dest="interpolate_sampling",
        action="store_false",
        default=None,
        help="For rk23 with --save-every: disable interpolation and record at accepted steps only.",
    )
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
    parser.add_argument(
        "--sweep",
        action="append",
        default=None,
        metavar="PARAM=V1,V2,...",
        help=(
            "Run a sequential zipped parameter sweep using repeated flags "
            "(max 2), e.g. --sweep k=0,1 --sweep diffusion_coeff=0.1,1.0"
        ),
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
        "charge_values": args.charge_values,
        "charge_counts": args.charge_counts,
        "t0": args.t0,
        "t_duration": args.t_duration,
        "save_every": args.save_every,
        "save_every_steps": args.save_every_steps,
        "method": args.method,
        "seed": args.seed,
        "rtol": args.rtol,
        "atol": args.atol,
        "first_step": args.first_step,
        "max_step_global": args.max_step_global,
        "eta": args.eta,
        "recompute_every": args.recompute_every,
        "interpolate_sampling": args.interpolate_sampling,
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


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {raw!r}")


def _coerce_sweep_value(field: str, raw: str, current_value: Any) -> Any:
    if isinstance(current_value, bool):
        return _parse_bool(raw)
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw)
    if isinstance(current_value, float):
        return float(raw)
    if isinstance(current_value, dict):
        raise ValueError(f"parameter {field!r} is not supported in --sweep")
    if isinstance(current_value, (tuple, list)):
        raise ValueError(
            f"internal error: tuple/list sweep values for {field!r} should be parsed separately"
        )
    if current_value is None:
        lowered = raw.strip().lower()
        if lowered in {"true", "false", "1", "0", "yes", "no", "on", "off"}:
            return _parse_bool(raw)
        try:
            return int(raw)
        except ValueError:
            try:
                return float(raw)
            except ValueError:
                return raw
    return raw


def _coerce_nested_value(raw: Any, template: Any, field: str) -> Any:
    if isinstance(template, bool):
        if isinstance(raw, bool):
            return raw
        return _parse_bool(str(raw))
    if isinstance(template, int) and not isinstance(template, bool):
        return int(raw)
    if isinstance(template, float):
        return float(raw)
    if template is None:
        return raw
    return type(template)(raw)


def _parse_composite_sweep_values(field: str, raw_values: str, current_value: Any) -> list[Any]:
    try:
        parsed = ast.literal_eval(f"[{raw_values}]")
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            f"invalid composite --sweep values for {field!r}; expected e.g. "
            f"{field}=[a,b],[c,d]"
        ) from exc

    if not isinstance(parsed, list) or not parsed:
        raise ValueError(f"--sweep for {field!r} must include at least one value")

    if isinstance(current_value, tuple):
        expected_len = len(current_value)
        out: list[Any] = []
        for item in parsed:
            if not isinstance(item, (list, tuple)):
                raise ValueError(
                    f"each sweep value for {field!r} must be a list/tuple, got {type(item).__name__}"
                )
            if len(item) != expected_len:
                raise ValueError(
                    f"each sweep value for {field!r} must have length {expected_len}"
                )
            coerced = tuple(
                _coerce_nested_value(v, tmpl, field)
                for v, tmpl in zip(item, current_value)
            )
            out.append(coerced)
        return out

    raise ValueError(f"parameter {field!r} is not supported in composite --sweep")


def _parse_sweep_specs(args, base_config: SimulationConfig) -> list[tuple[str, list[Any]]]:
    if not args.sweep:
        return []
    if len(args.sweep) > 2:
        raise ValueError("--sweep supports at most 2 parameters")
    if args.out is not None:
        raise ValueError("--out cannot be used with --sweep (would overwrite the same file)")
    if args.resume_from is not None:
        raise ValueError("--resume-from cannot be used with --sweep")

    valid_fields = set(SimulationConfig.__dataclass_fields__.keys())
    specs: list[tuple[str, list[Any]]] = []
    seen: set[str] = set()

    for item in args.sweep:
        if "=" not in item:
            raise ValueError(f"invalid --sweep spec {item!r} (expected PARAM=V1,V2,...)")
        field, raw_values = item.split("=", 1)
        field = field.strip().replace("-", "_")
        if field not in valid_fields:
            raise ValueError(f"unknown sweep parameter: {field!r}")
        if field in {"resume_from", "resume_force", "batch_every", "target_batch_mb", "max_wall_time"}:
            raise ValueError(f"sweeping {field!r} is not supported")
        if field in seen:
            raise ValueError(f"duplicate sweep parameter: {field!r}")
        seen.add(field)

        current_value = getattr(base_config, field, None)
        if isinstance(current_value, tuple):
            values = _parse_composite_sweep_values(field, raw_values, current_value)
        else:
            tokens = [token.strip() for token in raw_values.split(",")]
            if any(token == "" for token in tokens):
                raise ValueError(f"empty value in --sweep spec for {field!r}")
            values = [_coerce_sweep_value(field, token, current_value) for token in tokens]
        if not values:
            raise ValueError(f"--sweep for {field!r} must include at least one value")
        specs.append((field, values))

    if len(specs) == 2 and len(specs[0][1]) != len(specs[1][1]):
        f1, v1 = specs[0]
        f2, v2 = specs[1]
        raise ValueError(
            f"zipped --sweep values must have equal lengths: {f1} has {len(v1)}, {f2} has {len(v2)}"
        )

    return specs


def _build_sweep_configs(
    base_config: SimulationConfig, sweep_specs: list[tuple[str, list[Any]]]
) -> list[SimulationConfig]:
    if not sweep_specs:
        return [base_config]

    num_runs = len(sweep_specs[0][1])
    runs: list[SimulationConfig] = []
    for i in range(num_runs):
        overrides = {field: values[i] for field, values in sweep_specs}
        runs.append(replace(base_config, **overrides))
    return runs


def _resolve_out_path(config: SimulationConfig, args) -> Path:
    if config.resume_from is not None:
        return Path(config.resume_from)
    if args.out:
        return args.out

    date_dir = time.strftime("%Y%m%d")
    diffusion_tag = ""
    if config.diffusion:
        diffusion_tag = f"_diff_D{config.diffusion_coeff}"
    charge_tag = ""
    if config.charge_values is not None and config.charge_counts is not None:
        q1, q2 = config.charge_values
        n1, n2 = config.charge_counts
        charge_tag = f"_q{q1}-{q2}_n{n1}-{n2}"
    save_tag = ""
    if config.save_every is not None:
        save_tag = f"_save{config.save_every}"
    elif config.save_every_steps is not None:
        save_tag = f"_saveSteps{config.save_every_steps}"
    stem = (
        f"{config.method}_N{config.n_particles}_t{config.t_duration}_"
        f"k{config.k}_rtol{config.rtol}_atol{config.atol}"
        f"{charge_tag}{save_tag}{diffusion_tag}"
    )
    suffix = ".npz" if config.out_format == "npz" else f".{config.out_format}"
    return Path("data") / date_dir / f"{stem}{suffix}"


def _run_single(config: SimulationConfig, out_path: Path) -> None:
    sim = run_simulation(**config.__dict__, out_path=out_path)
    if sim.get("saved_path"):
        return

    if out_path.suffix.lower() in {".h5", ".hdf5"}:
        save_h5(out_path, sim)
    else:
        save_npz(out_path, sim)


def main():
    args = _parse_args()
    config = _config_from_args(args)
    sweep_specs = _parse_sweep_specs(args, config)
    run_configs = _build_sweep_configs(config, sweep_specs)

    if len(run_configs) == 1 and not sweep_specs:
        out_path = _resolve_out_path(run_configs[0], args)
        _run_single(run_configs[0], out_path)
        return 0

    failures: list[tuple[int, Path, str]] = []
    total = len(run_configs)
    planned_paths = [_resolve_out_path(run_config, args) for run_config in run_configs]
    if len({str(p) for p in planned_paths}) != len(planned_paths):
        raise ValueError(
            "sweep would produce duplicate output paths with current auto naming; "
            "choose sweep parameters included in the filename or run separately"
        )
    print(f"Starting sweep: {total} run(s)")
    for idx, run_config in enumerate(run_configs, start=1):
        out_path = planned_paths[idx - 1]
        sweep_desc = ", ".join(
            f"{field}={getattr(run_config, field)!r}" for field, _ in sweep_specs
        )
        print(f"[{idx}/{total}] {sweep_desc} -> {out_path}")
        try:
            _run_single(run_config, out_path)
        except Exception as exc:
            failures.append((idx, out_path, f"{type(exc).__name__}: {exc}"))
            print(f"[{idx}/{total}] FAILED: {exc}")
            continue

    if failures:
        print("")
        print(f"Sweep completed with {len(failures)} failure(s) out of {total} runs.")
        for idx, out_path, msg in failures:
            print(f"  - run {idx}: {out_path} :: {msg}")
        return 1

    print(f"Sweep completed successfully: {total}/{total} runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
