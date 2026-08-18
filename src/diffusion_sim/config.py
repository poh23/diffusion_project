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
    init_radii: dict | None = None
    charge_values: tuple[float, float] | None = None
    charge_counts: tuple[int, int] | None = None
    # integration / time
    t0: float = 0.0
    t_duration: float = 1.0
    save_every: float | None = None
    save_every_steps: int | None = None
    method: str = "rk23"  # "rk23" or "dop853"
    seed: int = 0
    out_format: str = "npz"
    # progress / chunking
    chunk_steps: int = 0
    # batching / resume
    batch_every: float | None = None
    target_batch_mb: float | None = None
    max_wall_time: float | None = None
    resume_from: str | None = None
    resume_force: bool = False
    # dop853 tolerances
    rtol: float = 1e-6
    atol: float = 1e-6
    # rk23 adaptive settings
    first_step: float | None = None
    max_step_global: float = np.inf
    eta: float = 0.05
    recompute_every: int = 10
    interpolate_sampling: bool = True
    # stochastic diffusion (rk23 only)
    diffusion: bool = False
    diffusion_coeff: float = 0.0
    diffusion_seed: int | None = None
    diffusion_noise_var: float = 1.0
    external_potential: str | None = None
    external_potential_params: dict | None = None


def _coerce_init_radius_value(value):
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return float(value)


def load_config_file(config_path: Path) -> SimulationConfig:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "out_format" not in data and "out-format" in data:
        data["out_format"] = data["out-format"]
    if "resume_from" not in data and "resume-from" in data:
        data["resume_from"] = data["resume-from"]
    if "resume_force" not in data and "resume-force" in data:
        data["resume_force"] = data["resume-force"]
    if "batch_every" not in data and "batch-every" in data:
        data["batch_every"] = data["batch-every"]
    if "target_batch_mb" not in data and "target-batch-mb" in data:
        data["target_batch_mb"] = data["target-batch-mb"]
    if "init_radii" not in data and "init-radii" in data:
        data["init_radii"] = data["init-radii"]
    if "max_wall_time" not in data and "max-wall-time" in data:
        data["max_wall_time"] = data["max-wall-time"]
    if "charge_values" not in data and "charge-values" in data:
        data["charge_values"] = data["charge-values"]
    if "charge_counts" not in data and "charge-counts" in data:
        data["charge_counts"] = data["charge-counts"]
    if "external_potential" not in data and "external-potential" in data:
        data["external_potential"] = data["external-potential"]
    if "external_potential_params" not in data and "external-potential-params" in data:
        data["external_potential_params"] = data["external-potential-params"]

    if "save_every" not in data and "rk23_sample_dt" in data:
        data["save_every"] = data["rk23_sample_dt"]
    if "save_every_steps" not in data and "save-every-steps" in data:
        data["save_every_steps"] = data["save-every-steps"]
    if "interpolate_sampling" not in data and "interpolate-sampling" in data:
        data["interpolate_sampling"] = data["interpolate-sampling"]
    if "save_every" not in data and "dt" in data:
        data["save_every"] = data["dt"]

    if "t_duration" not in data:
        if "t_end" in data:
            t0 = data.get("t0", 0.0)
            data["t_duration"] = data["t_end"] - t0
        elif "dt" in data and "steps" in data:
            data["t_duration"] = data["dt"] * data["steps"]

    for legacy_key in (
        "out-format",
        "resume-from",
        "resume-force",
        "batch-every",
        "target-batch-mb",
        "init-radii",
        "max-wall-time",
        "charge-values",
        "charge-counts",
        "external-potential",
        "external-potential-params",
        "rk23_sample_dt",
        "rk23_sample_count",
        "save-every-steps",
        "interpolate-sampling",
        "rk23_status_every_steps",
        "rk23_status_every_sec",
        "dt",
        "steps",
        "t_end",
        "compute_density",
        "density_print_every",
        "density_stride",
        "print_every_chunks",
    ):
        data.pop(legacy_key, None)

    if "charge_values" in data and data["charge_values"] is not None:
        data["charge_values"] = tuple(float(v) for v in data["charge_values"])
    if "charge_counts" in data and data["charge_counts"] is not None:
        data["charge_counts"] = tuple(data["charge_counts"])
    if "init_radii" in data and data["init_radii"] is not None:
        init_radii = data["init_radii"]
        if isinstance(init_radii, dict):
            data["init_radii"] = {str(k): _coerce_init_radius_value(v) for k, v in init_radii.items()}

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
    if config.r_floor < 0.0:
        errors.append("r_floor must be >= 0")
    # k=0 is supported (log potential), but keep other checks intact
    if config.init_radius <= 0.0:
        errors.append("init_radius must be > 0")
    if config.init_radii is not None:
        if not isinstance(config.init_radii, dict):
            errors.append("init_radii must be a JSON object / dict")
        else:
            keys = set(config.init_radii.keys())
            if keys != {"low", "high"}:
                errors.append("init_radii must contain exactly 'low' and 'high'")
            else:
                for key in ("low", "high"):
                    value = config.init_radii[key]
                    if isinstance(value, (list, tuple)):
                        if len(value) != 2:
                            errors.append(f"init_radii['{key}'] range must have length 2")
                            break
                        try:
                            inner = float(value[0])
                            outer = float(value[1])
                        except (TypeError, ValueError):
                            errors.append(f"init_radii['{key}'] range values must be numbers")
                            break
                        if not np.isfinite(inner) or not np.isfinite(outer):
                            errors.append(f"init_radii['{key}'] range values must be finite")
                            break
                        if inner < 0.0:
                            errors.append(f"init_radii['{key}'] inner radius must be >= 0")
                            break
                        if outer <= inner:
                            errors.append(f"init_radii['{key}'] outer radius must be > inner radius")
                            break
                    else:
                        try:
                            radius = float(value)
                        except (TypeError, ValueError):
                            errors.append(f"init_radii['{key}'] must be a number or length-2 range")
                            break
                        if not np.isfinite(radius) or radius <= 0.0:
                            errors.append(f"init_radii['{key}'] must be finite and > 0")
                            break
        if config.charge_values is None or config.charge_counts is None:
            errors.append("init_radii requires charge_values and charge_counts")
    if config.method not in ("rk23", "dop853"):
        errors.append("method must be one of: rk23, dop853")
    if config.t_duration <= 0.0:
        errors.append("t_duration must be > 0")
    if config.out_format not in ("npz", "h5", "hdf5"):
        errors.append("out_format must be one of: npz, h5, hdf5")
    if config.save_every is not None and config.save_every <= 0.0:
        errors.append("save_every must be > 0 when set")
    if config.save_every_steps is not None:
        try:
            save_every_steps_val = float(config.save_every_steps)
        except (TypeError, ValueError):
            errors.append("save_every_steps must be a positive integer when set")
        else:
            if not save_every_steps_val.is_integer() or int(save_every_steps_val) <= 0:
                errors.append("save_every_steps must be a positive integer when set")
    if config.save_every is not None and config.save_every_steps is not None:
        errors.append("save_every and save_every_steps are mutually exclusive")
    if config.batch_every is not None and config.batch_every <= 0.0:
        errors.append("batch_every must be > 0 when set")
    if config.target_batch_mb is not None and config.target_batch_mb <= 0.0:
        errors.append("target_batch_mb must be > 0 when set")
    if config.max_wall_time is not None and config.max_wall_time <= 0.0:
        errors.append("max_wall_time must be > 0 when set")
    if config.diffusion_coeff < 0.0:
        errors.append("diffusion_coeff must be >= 0")
    if config.diffusion_noise_var < 0.0:
        errors.append("diffusion_noise_var must be >= 0")
    if config.diffusion and config.method != "rk23":
        errors.append("diffusion is only supported with method='rk23'")
    if config.method == "dop853" and config.save_every is None:
        errors.append("save_every must be set for method='dop853'")
    if config.method == "dop853" and config.save_every_steps is not None:
        errors.append("save_every_steps is only supported for method='rk23'")
    if (
        config.batch_every is not None
        or config.target_batch_mb is not None
        or config.max_wall_time is not None
        or config.resume_from is not None
    ):
        if config.out_format not in ("h5", "hdf5"):
            errors.append("batching/resume requires out_format 'h5' or 'hdf5'")
    if (config.charge_values is None) ^ (config.charge_counts is None):
        errors.append("charge_values and charge_counts must be set together")
    if config.charge_values is not None:
        if len(config.charge_values) != 2:
            errors.append("charge_values must have length 2")
        else:
            for value in config.charge_values:
                if value <= 0.0:
                    errors.append("charge_values must be positive")
                    break
    if config.charge_counts is not None:
        if len(config.charge_counts) != 2:
            errors.append("charge_counts must have length 2")
        else:
            total = 0
            for count in config.charge_counts:
                try:
                    count_val = float(count)
                except (TypeError, ValueError):
                    errors.append("charge_counts must be integers")
                    break
                if not count_val.is_integer():
                    errors.append("charge_counts must be integers")
                    break
                count_int = int(count_val)
                if count_int <= 0:
                    errors.append("charge_counts must be positive")
                    break
                total += count_int
            if total != config.n_particles:
                errors.append("charge_counts must sum to n_particles")
    if config.external_potential is not None:
        if config.external_potential != "harmonic":
            errors.append("external_potential must be 'harmonic' when set")
        params = config.external_potential_params
        if params is None:
            errors.append("external_potential_params must be set when external_potential is used")
        elif not isinstance(params, dict):
            errors.append("external_potential_params must be a JSON object / dict")
        else:
            omega = params.get("omega")
            if omega is None:
                errors.append("external_potential_params must include 'omega' for harmonic potential")
            else:
                try:
                    float(omega)
                except (TypeError, ValueError):
                    errors.append("external_potential_params['omega'] must be a number")
            if "center" in params:
                center = params["center"]
                if not isinstance(center, (list, tuple)) or len(center) != 2:
                    errors.append("external_potential_params['center'] must be a length-2 list or tuple")
                else:
                    for value in center:
                        try:
                            float(value)
                        except (TypeError, ValueError):
                            errors.append("external_potential_params['center'] values must be numbers")
                            break

    if errors:
        raise ValueError("Invalid SimulationConfig: " + "; ".join(errors))
