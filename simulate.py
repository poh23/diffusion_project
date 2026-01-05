import argparse
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from nbody_core import (
    init_positions_jittered_disk,
    run_rk2_loop,
    run_rk2_loop_with_noise,
    run_rk4_loop,
    run_dop853_chunked,
)
from density_voronoi import compute_density_and_radius_series

DEFAULT_SOFTENING = 1e-1


@dataclass
class SimulationConfig:
    # system/physics
    n_particles: int = 10
    k: float = 1.0
    v0: float = 1.0
    l: float = 1.0
    softening: float = DEFAULT_SOFTENING
    # integration / time
    dt: float = 0.01
    steps: int = 200
    t0: float = 0.0
    method: str = "rk4"  # "rk4" or "rk2" or "dop853"
    # diffusion (rk2 only)
    D: float = 0.0
    var_chi: float = 1.0
    seed: int = 0
    # progress / chunking
    chunk_steps: int = 0
    print_every_chunks: int = 1
    # dop853 tolerances
    rtol: float = 1e-6
    atol: float = 1e-6
    # optional density computation (post-process, SciPy Voronoi)
    compute_density: bool = False
    density_print_every: int | None = None
    density_stride: int = 1  # compute every n steps when density is enabled

# ----------------------------
# Fixed-step (Numba) RK2/RK4 with progress via chunking
# ----------------------------
def _run_fixedstep_integrator(config: SimulationConfig, r0):
    return run_fixedstep_chunked(
        r0,
        config.k,
        config.v0,
        config.l,
        config.softening,
        config.dt,
        config.steps,
        config.t0,
        method=config.method,
        D=config.D,
        var_chi=config.var_chi,
        seed=config.seed,
        chunk_steps=config.chunk_steps,
        print_every_chunks=config.print_every_chunks,
    )


def run_fixedstep_chunked(
    r0,
    k,
    v0,
    l,
    softening,
    dt,
    steps,
    t0,
    method="rk4",
    D=0.0,          # diffusion constant
    var_chi=1.0,    # Var(chi) per coordinate (usually 1)
    seed=0,
    chunk_steps=5000,
    print_every_chunks=1,
):
    """
    Runs rk4/rk2 in chunks and prints progress between chunks.

    NOTE on seeds:
    - seed controls reproducibility overall.
    - for rk2 random-walk, we pass a per-chunk seed = seed + idx so runs are reproducible.
    """
    n = r0.shape[0]
    positions = np.empty((steps, n, 2), dtype=np.float64)
    times = np.empty(steps, dtype=np.float64)
    energy = np.empty(steps, dtype=np.float64)
    std = np.empty(steps, dtype=np.float64)

    r = r0.copy()
    t = t0
    idx = 0
    show_progress = print_every_chunks is None or print_every_chunks > 0
    pbar = tqdm(total=steps, desc=method, unit="step", disable=not show_progress)

    while idx < steps:
        m = min(chunk_steps, steps - idx)

        if method == "rk4":
            r_final, r_hist, pe_hist, std_hist, t_hist = run_rk4_loop(
                r, k, v0, l, softening, dt, m, t
            )
        elif method == "rk2":
            if D > 0.0:
                # dx = sqrt(2*D*dt) * chi, with Var(chi)=var_chi
                rng = np.random.default_rng((seed + idx) & 0xFFFFFFFF)

                chi = rng.standard_normal(size=(m, n, 2)).astype(np.float64)
                if var_chi != 1.0:
                    chi *= np.sqrt(var_chi)

                noise = (np.sqrt(2.0 * D * dt) * chi).astype(np.float64)

                r_final, r_hist, pe_hist, std_hist, t_hist = run_rk2_loop_with_noise(
                    r, k, v0, l, softening, dt, m, t, noise
                )
            else:
                # no diffusion
                seed_numba = (seed + idx) & 0xFFFFFFFF
                r_final, r_hist, pe_hist, std_hist, t_hist = run_rk2_loop(
                    r, k, v0, l, softening, dt, m, t,
                    random_walk_std=0.0, seed=seed_numba
                )

        else:
            raise ValueError("method must be 'rk4' or 'rk2'")

        positions[idx : idx + m] = r_hist
        times[idx : idx + m] = t_hist
        energy[idx : idx + m] = pe_hist
        std[idx : idx + m] = std_hist

        r = r_final
        t += m * dt
        idx += m

        if show_progress:
            pbar.update(m)

    if show_progress:
        pbar.close()
    return r, positions, energy, std, times


def _auto_chunk_steps(steps: int, *, target_updates: int = 100, min_chunk: int = 100, max_chunk: int = 5000) -> int:
    """
    Choose a chunk size to get roughly target_updates progress updates without making chunks too small.
    """
    if steps <= 0:
        return min_chunk
    chunk = max(min_chunk, steps // max(1, target_updates))
    return int(max(1, min(chunk, max_chunk)))


# ----------------------------
# Public API
# ----------------------------
def run_simulation(
    *,
    n_particles=10,
    k=1.0,
    v0=1.0,
    l=1.0,
    dt=0.01,
    steps=200,
    method="rk4",  # "rk4" or "rk2" or "dop853"
    D=0.0,        # diffusion constant (used for rk2)
    var_chi=1.0,  # Var(chi) per coordinate
    seed=0,
    softening=DEFAULT_SOFTENING,
    t0=0.0,
    # progress control
    chunk_steps=5000,
    print_every_chunks=1,
    # dop853 tolerances
    rtol=1e-6,
    atol=1e-6,
    compute_density=False,
    density_print_every=None,
    density_stride=1,
):
    config = SimulationConfig(
        n_particles=n_particles,
        k=k,
        v0=v0,
        l=l,
        softening=softening,
        dt=dt,
        steps=steps,
        t0=t0,
        method=method,
        D=D,
        var_chi=var_chi,
        seed=seed,
        chunk_steps=chunk_steps,
        print_every_chunks=print_every_chunks,
        rtol=rtol,
        atol=atol,
        compute_density=compute_density,
        density_print_every=density_print_every,
        density_stride=density_stride,
    )

    r0 = init_positions_jittered_disk(config.n_particles, seed=config.seed)

    # If chunk_steps <= 0, auto-select for nicer tqdm updates
    if config.chunk_steps <= 0:
        if config.method == "dop853":
            config = replace(config, chunk_steps=_auto_chunk_steps(config.steps, target_updates=80, min_chunk=500, max_chunk=10000))
        else:
            config = replace(config, chunk_steps=_auto_chunk_steps(config.steps, target_updates=100, min_chunk=100, max_chunk=5000))

    t_start = time.perf_counter()

    if config.method in ("rk4", "rk2"):
        r_final, r_hist, pe_hist, std_hist, t_hist = _run_fixedstep_integrator(config, r0)
    elif config.method == "dop853":
        r_final, r_hist, pe_hist, std_hist, t_hist = run_dop853_chunked(
            r0,
            config.k,
            config.v0,
            config.l,
            config.softening,
            config.dt,
            config.steps,
            config.t0,
            rtol=config.rtol,
            atol=config.atol,
            chunk_steps=config.chunk_steps,
            print_every_chunks=config.print_every_chunks,
        )
    else:
        raise ValueError("method must be 'rk4', 'rk2', or 'dop853'")

    elapsed = time.perf_counter() - t_start

    density = None
    radii = None
    if config.compute_density:
        # stride > 1: compute densities on subsampled steps to save time
        if config.density_stride > 1:
            r_hist_for_density = r_hist[:: config.density_stride]
        else:
            r_hist_for_density = r_hist

        density_raw, radii_raw = compute_density_and_radius_series(
            r_hist_for_density,
            print_every=config.density_print_every,
        )

        if config.density_stride > 1:
            # expand back to full length with NaNs for skipped steps to keep alignment
            steps = len(r_hist)
            n = r_hist.shape[1]
            density = np.full((steps, n), np.nan, dtype=np.float64)
            radii = np.full((steps, n), np.nan, dtype=np.float64)
            density[:: config.density_stride] = density_raw
            radii[:: config.density_stride] = radii_raw
        else:
            density = density_raw
            radii = radii_raw

    meta = dict(
        n_particles=config.n_particles,
        k=config.k,
        v0=config.v0,
        l=config.l,
        dt=config.dt,
        steps=config.steps,
        method=config.method,
        D=config.D if config.method == "rk2" else None,
        var_chi=config.var_chi if config.method == "rk2" else None,
        seed=config.seed,
        softening=config.softening,
        t0=config.t0,
        rtol=config.rtol if config.method == "dop853" else None,
        atol=config.atol if config.method == "dop853" else None,
        chunk_steps=config.chunk_steps,
        density_method="voronoi_2d" if config.compute_density else None,
        density_stride=1 if config.compute_density else None,
        elapsed_sec=elapsed,
        steps_per_sec=(config.steps / elapsed) if elapsed > 0 else None,
    )

    return dict(
        positions=r_hist,
        times=t_hist,
        energy=pe_hist,
        std=std_hist,
        final_positions=r_final,
        density=density,
        radii=radii,
        meta=meta,
    )


def save_npz(out_path, sim_dict):
    meta_json = json.dumps(sim_dict["meta"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "positions": sim_dict["positions"],
        "times": sim_dict["times"],
        "energy": sim_dict["energy"],
        "std": sim_dict["std"],
        "final_positions": sim_dict["final_positions"],
        "meta_json": np.array(meta_json, dtype=object),
    }

    # optional
    if sim_dict.get("density") is not None:
        arrays["density"] = sim_dict["density"]
    if sim_dict.get("radii") is not None:
        arrays["radii"] = sim_dict["radii"]

    np.savez_compressed(
        out_path,
        **arrays,
    )
    print(f"Saved: {out_path}")
    print(f"Meta: {sim_dict['meta']}")


def _load_config_file(config_path: Path) -> SimulationConfig:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    allowed_keys = set(SimulationConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in data.items() if k in allowed_keys}
    unknown = set(data.keys()) - allowed_keys
    if unknown:
        print(f"Warning: ignoring unknown config keys: {sorted(unknown)}")

    return SimulationConfig(**filtered)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run diffusion simulation and save NPZ output.")
    parser.add_argument("--config", type=Path, help="JSON config file matching SimulationConfig fields.")
    parser.add_argument("--out", type=Path, help="Output .npz path (default: data/sim_<timestamp>.npz).")

    parser.add_argument("--n-particles", type=int)
    parser.add_argument("--k", type=float)
    parser.add_argument("--v0", type=float)
    parser.add_argument("--l", type=float)
    parser.add_argument("--softening", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--t0", type=float)
    parser.add_argument("--method", choices=["rk2", "rk4", "dop853"])
    parser.add_argument("--D", type=float, help="Diffusion constant (rk2 only).")
    parser.add_argument("--var-chi", type=float, help="Variance of chi per coordinate (rk2 only).")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chunk-steps", type=int)
    parser.add_argument("--print-every-chunks", type=int)
    parser.add_argument("--rtol", type=float, help="dop853 relative tolerance.")
    parser.add_argument("--atol", type=float, help="dop853 absolute tolerance.")
    parser.add_argument("--compute-density", action="store_true", help="Compute Voronoi density/radii (post-process).")
    parser.add_argument("--density-print-every", type=int, help="Print progress every N steps during density computation.")
    parser.add_argument("--density-stride", type=int, help="Compute density every N steps (default 1 = every step).")
    return parser.parse_args()


def _config_from_args(args) -> SimulationConfig:
    config = SimulationConfig()
    if args.config:
        config = _load_config_file(args.config)

    overrides = {
        "n_particles": args.n_particles,
        "k": args.k,
        "v0": args.v0,
        "l": args.l,
        "softening": args.softening,
        "dt": args.dt,
        "steps": args.steps,
        "t0": args.t0,
        "method": args.method,
        "D": args.D,
        "var_chi": args.var_chi,
        "seed": args.seed,
        "chunk_steps": args.chunk_steps,
        "print_every_chunks": args.print_every_chunks,
        "rtol": args.rtol,
        "atol": args.atol,
        "compute_density": args.compute_density,
        "density_print_every": args.density_print_every,
        "density_stride": args.density_stride,
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
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        out_path = Path("data") / f"sim_{timestamp}.npz"

    save_npz(out_path, sim)


if __name__ == "__main__":
    main()
