import argparse
import os
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
import imageio_ffmpeg

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if SRC_PATH.exists():
    sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.simulation import run_simulation
from diffusion_sim.io.npz import load_npz, save_npz

# Notebook embedding
from IPython.display import Video

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# Core math: normalized divergence
# ============================================================

def compute_norm_dists(sim_a, sim_b):
    """
    Returns: times (T,), norm_dists (T,N)
    norm_dists = ||r_a - r_b|| / spacing
    spacing estimated from mean radius on sim_a: 2*pi*R/N
    """
    r_a = np.asarray(sim_a["positions"])
    r_b = np.asarray(sim_b["positions"])
    t_a = np.asarray(sim_a["times"])
    t_b = np.asarray(sim_b["times"])

    min_len = min(len(r_a), len(r_b), len(t_a), len(t_b))
    r_a = r_a[:min_len]
    r_b = r_b[:min_len]
    times = t_a[:min_len]
    N = r_a.shape[1]

    dists = np.sqrt(np.sum((r_a - r_b) ** 2, axis=2))  # (T, N)

    mean_radius = np.mean(np.linalg.norm(r_a, axis=2))
    spacing = (2.0 * np.pi * mean_radius) / N
    norm_dists = dists / spacing

    return times, norm_dists


# ============================================================
# Plotting
# ============================================================

def plot_divergence_pair(sim_a, sim_b, *, title="Solver Error Relative to Spacing", t_max=None,
                         save_path=None, show=True, alpha_band=0.15, label="Mean"):
    times, norm_dists = compute_norm_dists(sim_a, sim_b)

    if t_max is not None:
        mask = times <= t_max
        times = times[mask]
        norm_dists = norm_dists[mask]

    mean_dist = np.mean(norm_dists, axis=1)
    min_dist = np.min(norm_dists, axis=1)
    max_dist = np.max(norm_dists, axis=1)
    N = np.asarray(sim_a["positions"]).shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(times, mean_dist, linewidth=2, label=label)
    ax.fill_between(times, min_dist, max_dist, alpha=alpha_band, label="Min-Max")

    ax.axhline(y=0.01, linestyle="--", label="1% Spacing (Safe)")
    ax.axhline(y=1.00, linestyle="--", label="100% Spacing (Failure)")

    ax.set_title(f"{title} (N={N})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Error / Spacing")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_divergence_sweep(experiments, *, title="RK23 vs DOP853 (sweep)",
                          t_max=None, save_path=None, show=True, alpha_band=0.15,
                          save_fresh=False, out_dir="data"):
    """
    experiments: list of dicts.
      mode="fresh": runs the simulation twice (solver_a + solver_b)
      mode="files": loads two npz files
    """
    if save_fresh:
        os.makedirs(out_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(y=0.01, linestyle="--", label="1% Spacing (Safe)")
    ax.axhline(y=1.00, linestyle="--", label="100% Spacing (Failure)")

    for idx, exp in enumerate(experiments, start=1):
        mode = exp.get("mode", "fresh")
        label = exp.get("label", f"Exp {idx}")

        if mode == "files":
            sim_a = load_npz(exp.get("solver_a_file", exp.get("rk4_file")))
            sim_b = load_npz(exp.get("solver_b_file", exp.get("dop_file")))

        elif mode == "fresh":
            fresh_kwargs = dict(exp)
            sim_a, sim_b = run_comparison_fresh(
                **fresh_kwargs,
                save_fresh=save_fresh,
                out_dir=out_dir,
            )

        else:
            raise ValueError(f"Unknown experiment mode: {mode}")

        times, norm_dists = compute_norm_dists(sim_a, sim_b)

        if t_max is not None:
            mask = times <= t_max
            times = times[mask]
            norm_dists = norm_dists[mask]
            if len(times) < 2:
                print(f"Warning: '{label}' has <2 points after t_max clipping; skipping.")
                continue

        mean_dist = np.mean(norm_dists, axis=1)
        min_dist = np.min(norm_dists, axis=1)
        max_dist = np.max(norm_dists, axis=1)

        (line,) = ax.plot(times, mean_dist, linewidth=2, label=label)
        ax.fill_between(times, min_dist, max_dist, alpha=alpha_band, color=line.get_color())

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Error / Spacing")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=200)
        print(f"Saved plot: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def _stem_from_exp(exp):
    # for file naming; keep it simple and readable
    N = exp["N"]
    dt = exp["dt"]
    steps = exp["steps"]
    k = exp.get("k", 1.0)
    r_floor = exp.get("r_floor", 1e-1)
    seed = exp.get("seed_init", 1)
    solver_a = exp.get("solver_a", "rk4")
    solver_b = exp.get("solver_b", "dop853")
    return f"k{k}_N{N}_steps{steps}_dt{dt}_r_floor{r_floor}_{solver_a}_vs_{solver_b}_seed{seed}"


# ============================================================
# Video + notebook embedding
# ============================================================

def create_comparison_video(sim_a, sim_b, filename="solver_comparison.mp4", step=10,
                            label_a="RK23", label_b="DOP853"):
    r_a = np.asarray(sim_a["positions"])[::step]
    r_b = np.asarray(sim_b["positions"])[::step]

    min_len = min(len(r_a), len(r_b))
    r_a = r_a[:min_len]
    r_b = r_b[:min_len]

    all_pos = np.concatenate([r_a, r_b], axis=0)
    max_range = np.max(np.abs(all_pos)) * 1.1

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal")
    ax.set_title(f"Solver Divergence: {label_a} (Blue) vs {label_b} (Red)")
    ax.grid(True, alpha=0.3)

    lines = LineCollection([], colors="black", linewidths=0.8, alpha=0.5)
    ax.add_collection(lines)

    scat_a = ax.scatter(r_a[0, :, 0], r_a[0, :, 1],
                        color="blue", s=40, alpha=0.85, edgecolors="white",
                        label=label_a)
    scat_b = ax.scatter(r_b[0, :, 0], r_b[0, :, 1],
                        color="red", s=40, alpha=0.85, edgecolors="white",
                        label=label_b)
    ax.legend(loc="upper right")

    def update(frame):
        pa = r_a[frame]
        pb = r_b[frame]
        scat_a.set_offsets(pa)
        scat_b.set_offsets(pb)
        segments = np.stack((pa, pb), axis=1)  # (N,2,2)
        lines.set_segments(segments)
        return scat_a, scat_b, lines

    print(f"Rendering video: {filename} ({min_len} frames, step={step})")
    ani = FuncAnimation(fig, update, frames=min_len, blit=True)
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=30, bitrate=3000)
    ani.save(filename, writer=writer, dpi=120)
    plt.close(fig)
    print(f"Saved video: {filename}")
    return filename


def embed_mp4(path, width=600, embed=True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Video(path, width=width, embed=embed)


def animate_comparison_mp4(sim_a, sim_b, filename="solver_comparison.mp4", step=10,
                           label_a="RK23", label_b="DOP853", embed=True, width=600):
    path = create_comparison_video(sim_a, sim_b, filename=filename, step=step,
                                   label_a=label_a, label_b=label_b)
    return embed_mp4(path, width=width, embed=embed)


# ============================================================
# Fresh run API (calls simulation)
# ============================================================

def run_comparison_fresh(
    *,
    N, dt, steps,
    k=1.0, v0=1.0, l=1.0, r_floor=1e-1,
    seed_init=1,
    chunk_steps=5000,
    rtol=1e-6,
    atol=1e-6,
    solver_a="rk23",
    solver_b="dop853",
    mode="fresh",
    label=None,
    save_fresh=False,
    out_dir="data",
):
    t_duration = dt * steps
    print(f"Running {solver_a}: N={N}, dt={dt}, steps={steps}, k={k}, r_floor={r_floor}")
    sim_a = run_simulation(
        n_particles=N, k=k, v0=v0, l=l,
        t_duration=t_duration, save_every=dt,
        method=solver_a,
        seed=seed_init,
        r_floor=r_floor,
        chunk_steps=chunk_steps,
    )

    if save_fresh:
        stem = _stem_from_exp(dict(
            N=N, dt=dt, steps=steps, k=k, r_floor=r_floor, seed_init=seed_init,
            solver_a=solver_a, solver_b=solver_b
        ))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_npz(out_dir / f"{solver_a}_{stem}.npz", sim_a)

    print(f"Running {solver_b}: N={N}, dt={dt}, steps={steps}, k={k}, r_floor={r_floor}")
    sim_b = run_simulation(
        n_particles=N, k=k, v0=v0, l=l,
        t_duration=t_duration, save_every=dt,
        method=solver_b,
        seed=seed_init,
        r_floor=r_floor,
        chunk_steps=chunk_steps,
        rtol=rtol,
        atol=atol,
    )

    if save_fresh:
        stem = _stem_from_exp(dict(
            N=N, dt=dt, steps=steps, k=k, r_floor=r_floor, seed_init=seed_init,
            solver_a=solver_a, solver_b=solver_b
        ))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_npz(out_dir / f"{solver_b}_{stem}.npz", sim_b)

    return sim_a, sim_b


def _load_experiments_config(path: Path):
    import json

    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Config file must contain a JSON list of experiment dicts.")
    return data


def _parse_args():
    parser = argparse.ArgumentParser(description="Compare solvers and optionally plot/save results.")
    parser.add_argument("--config", type=Path, required=True, help="JSON file with list of experiment dicts.")
    parser.add_argument("--title", default="RK23 vs DOP853 (sweep)", help="Plot title.")
    parser.add_argument("--t-max", type=float, help="Clip times to this value.")
    parser.add_argument("--save-path", type=Path, help="Path to save the plot PNG.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="Directory to save fresh runs when enabled.")
    parser.add_argument("--save-fresh", action="store_true", help="Save fresh runs to disk.")
    parser.add_argument("--no-show", action="store_true", help="Do not display the plot.")
    return parser.parse_args()


def main():
    args = _parse_args()
    experiments = _load_experiments_config(args.config)

    plot_divergence_sweep(
        experiments,
        title=args.title,
        t_max=args.t_max,
        save_path=args.save_path,
        show=not args.no_show,
        save_fresh=args.save_fresh,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
