import os
import re
import json
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.collections import LineCollection
import imageio_ffmpeg

from simulate import run_simulation, save_npz

# Notebook embedding
from IPython.display import Video

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


# ============================================================
# IO: load npz (old + new compatible)
# ============================================================

def _infer_meta_from_filename_and_data(path, positions, times):
    fname = os.path.basename(path)
    meta = {}

    if positions is not None:
        meta["n_particles"] = int(positions.shape[1])
        meta["steps"] = int(positions.shape[0])

    if times is not None and len(times) >= 2:
        meta["dt"] = float(np.median(np.diff(times)))
        meta["t0"] = float(times[0])
    else:
        meta["dt"] = None
        meta["t0"] = None

    mN = re.search(r"N(\d+)", fname)
    msteps = re.search(r"steps(\d+)", fname)
    mdt = re.search(r"dt([0-9.]+(?:e[-+]?\d+)?)", fname, flags=re.IGNORECASE)
    if mN:
        meta["n_particles"] = int(mN.group(1))
    if msteps:
        meta["steps"] = int(msteps.group(1))
    if mdt:
        try:
            meta["dt"] = float(mdt.group(1))
        except ValueError:
            pass

    for method in ("dop853", "rk4", "rk2"):
        if method in fname.lower():
            meta["method"] = method
            break
    meta.setdefault("method", "unknown")

    # common params (may be absent in old files)
    meta.setdefault("k", None)
    meta.setdefault("v0", None)
    meta.setdefault("l", None)
    meta.setdefault("softening", None)
    meta.setdefault("random_walk_std", 0.0)

    return meta


def load_npz_compat(path):
    data = np.load(path, allow_pickle=True)

    def pick(*names):
        for n in names:
            if n in data.files:
                return data[n]
        return None

    positions = pick("positions", "r_history", "r_hist")
    times = pick("times", "t", "t_hist")
    energy = pick("energy", "pe", "pe_history", "pe_hist")
    std = pick("std", "std_history", "std_hist")
    final_positions = pick("final_positions", "r_final")

    if positions is None:
        raise KeyError(f"{path}: couldn't find positions. Keys={data.files}")

    if times is None:
        times = np.arange(positions.shape[0], dtype=float)

    if energy is None:
        energy = np.full(positions.shape[0], np.nan)

    if std is None:
        std = np.full(positions.shape[0], np.nan)

    if final_positions is None:
        final_positions = positions[-1]

    meta = None
    if "meta_json" in data.files:
        try:
            meta = json.loads(str(data["meta_json"].item()))
        except Exception:
            meta = None
    if meta is None:
        meta = _infer_meta_from_filename_and_data(path, positions, times)

    return {
        "positions": positions,
        "times": times,
        "energy": energy,
        "std": std,
        "final_positions": final_positions,
        "meta": meta,
        "source_path": path,
    }


# ============================================================
# Core math: normalized divergence
# ============================================================

def compute_norm_dists(sim_a, sim_b):
    """
    Returns: times (T,), norm_dists (T,N)
    norm_dists = ||r_a - r_b|| / spacing
    spacing estimated from mean radius on sim_a: 2π⟨R⟩/N
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
    ax.fill_between(times, min_dist, max_dist, alpha=alpha_band, label="Min–Max")

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


def plot_divergence_sweep(experiments, *, title="RK4 vs DOP853 (sweep)",
                          t_max=None, save_path=None, show=True, alpha_band=0.15,
                          save_fresh=False, out_dir="data"):
    """
    experiments: list of dicts.
      mode="fresh": runs simulate.py twice (rk4 + dop853)
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
            sim_rk4 = load_npz_compat(exp["rk4_file"])
            sim_dop = load_npz_compat(exp["dop_file"])

        elif mode == "fresh":
            sim_rk4, sim_dop = run_comparison_fresh(**exp)

            if save_fresh:
                stem = _stem_from_exp(exp)
                save_npz(os.path.join(out_dir, f"rk4_{stem}.npz"), sim_rk4)
                save_npz(os.path.join(out_dir, f"dop853_{stem}.npz"), sim_dop)

        else:
            raise ValueError(f"Unknown experiment mode: {mode}")

        times, norm_dists = compute_norm_dists(sim_rk4, sim_dop)

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
    soft = exp.get("softening", 1e-1)
    seed = exp.get("seed_init", 1)
    return f"k{k}_N{N}_steps{steps}_dt{dt}_softening{soft}_seed{seed}"


# ============================================================
# Video + notebook embedding
# ============================================================

def create_comparison_video(sim_a, sim_b, filename="solver_comparison.mp4", step=10,
                            label_a="RK4", label_b="DOP853"):
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
                           label_a="RK4", label_b="DOP853", embed=True, width=600):
    path = create_comparison_video(sim_a, sim_b, filename=filename, step=step,
                                   label_a=label_a, label_b=label_b)
    return embed_mp4(path, width=width, embed=embed)


# ============================================================
# Fresh run API (calls simulate.py)
# ============================================================

def run_comparison_fresh(
    *,
    N, dt, steps,
    k=1.0, v0=1.0, l=1.0, softening=1e-1,
    seed_init=1,
    chunk_steps=5000,
    print_every_chunks=1,
    rtol=1e-6,
    atol=1e-6,
    mode="fresh",
    label=None,
):
    sim_rk4 = run_simulation(
        n_particles=N, k=k, v0=v0, l=l,
        dt=dt, steps=steps,
        method="rk4",
        seed=seed_init,
        softening=softening,
        chunk_steps=chunk_steps,
        print_every_chunks=print_every_chunks,
    )

    sim_dop = run_simulation(
        n_particles=N, k=k, v0=v0, l=l,
        dt=dt, steps=steps,
        method="dop853",
        seed=seed_init,
        softening=softening,
        chunk_steps=chunk_steps,
        print_every_chunks=print_every_chunks,
        rtol=rtol,
        atol=atol,
    )

    return sim_rk4, sim_dop


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    experiments = [
        dict(
            mode="files",
            label="N=100",
            rk4_file=r"data\31-12-2025_k3_soft0.1_dt1e_N_comparison\rk4_k3.0_N100_steps500_dt0.001_softening0.1_seed1.npz",
            dop_file=r"data\31-12-2025_k3_soft0.1_dt1e_N_comparison\dop853_k3.0_N100_steps500_dt0.001_softening0.1_seed1.npz",
        ),
        dict(
            mode="files",
            label="dt=1e-3",
            rk4_file=r"data\31-12-2025_k3_soft0.1_dt1e_N_comparison\rk4_k3.0_N1000_steps500_dt0.001_softening0.1_seed1.npz",
            dop_file=r"data\31-12-2025_k3_soft0.1_dt1e_N_comparison\dop853_k3.0_N1000_steps500_dt0.001_softening0.1_seed1.npz",
        ),
        dict(mode="fresh", label="dt=1e-3, soft=1e-1", 
             N=5000, dt=1e-3, steps=500, seed_init=1,
             k=3.0, v0=1.0, l=1.0, softening=1e-1,
             chunk_steps=100, print_every_chunks=1),
    ]

    plot_divergence_sweep(
        experiments,
        title="RK4 vs DOP853 (sweep, clipped)",
        t_max=5,
        save_path=r"graphs\multi_divergence_sweep_N_comparison.png",
        save_fresh=True,
        out_dir=r"data\31-12-2025_k3_soft0.1_dt1e_N_comparison",
    )
