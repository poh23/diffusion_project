import os
import numpy as np
import matplotlib.pyplot as plt

import imageio_ffmpeg
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Only needed for notebook embedding
from IPython.display import Video

from npz_io import load_npz

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def plot_energy(sim):
    t = np.asarray(sim["times"])
    e = np.asarray(sim["energy"])

    plt.figure(figsize=(8, 4))
    plt.plot(t, e, linewidth=2)
    plt.xlabel("Time")
    plt.ylabel("Potential Energy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_std(sim, show_theory=True, k=None, v0=None, l=None):
    meta = sim.get("meta", {}) or {}

    # Try: explicit args > meta > None
    k = k if k is not None else meta.get("k", None)
    v0 = v0 if v0 is not None else meta.get("v0", None)
    l = l if l is not None else meta.get("l", None)

    dt = meta.get("dt", None)
    rw = meta.get("random_walk_std", 0.0)

    t = np.asarray(sim["times"])
    s = np.asarray(sim["std"])

    # log axes can't show 0
    m = t > 0
    t = t[m]
    s = s[m]

    plt.figure(figsize=(8, 4))
    plt.plot(t, s, linewidth=2, label="Simulation")

    # Only show theory if we actually have the parameters
    can_theory = (k is not None) and (v0 is not None) and (l is not None)
    if show_theory and can_theory:
        gamma = 1.0 / (k + 2.0)
        theo = (v0 * (l ** (k + 1)) * t) ** gamma
        plt.plot(t, theo, "--", linewidth=1, label="Theory (no diffusion)")

        # diffusion-only line requires dt and rw
        if (rw is not None) and (rw > 0) and (dt is not None):
            diff_only = rw * np.sqrt(t / dt)  # per-step random walk std assumption
            plt.plot(t, diff_only, "--", linewidth=1, label="Random-walk only")
    elif show_theory and not can_theory:
        print("Note: k/v0/l not found in file metadata -> skipping theory curves.")
        print("      You can pass them manually: plot_std(sim, k=..., v0=..., l=...)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Time")
    plt.ylabel("Std(r)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_mp4(sim, out_path="simulation.mp4", fps=30, dpi=120, step=10, marker_size=64):
    """
    Save an MP4 of particle motion (no trails) and return the output path.
    """
    r_hist = np.asarray(sim["positions"])
    n = r_hist.shape[1]

    r_view = r_hist[::step]
    if len(r_view) < 2:
        raise ValueError("Not enough frames to animate (try smaller step or more steps).")

    max_range = np.max(np.abs(r_hist)) * 1.1

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-max_range, max_range)
    ax.set_ylim(-max_range, max_range)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # init with first frame so scatter has correct size
    colors = plt.cm.jet(np.linspace(0, 1, n))
    particles = ax.scatter(r_view[0, :, 0], r_view[0, :, 1],
                           s=marker_size, c=colors)

    def update(frame_idx):
        particles.set_offsets(r_view[frame_idx])
        return (particles,)

    ani = FuncAnimation(fig, update, frames=len(r_view), blit=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    writer = FFMpegWriter(fps=fps, metadata={"artist": "you"}, bitrate=1800)
    ani.save(out_path, writer=writer, dpi=dpi)

    plt.close(fig)
    print(f"Saved video: {out_path}")
    return out_path


def embed_mp4(path, width=600, embed=True):
    """
    Return an IPython Video object for Jupyter display.
    - embed=True will base64-embed the video into the notebook output (portable, but can be large).
    - embed=False will link to the file (smaller output, but path must remain accessible).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Video(path, width=width, embed=embed)


def animate_mp4(sim, out_path="simulation.mp4", fps=30, dpi=120, step=10,
                marker_size=64, width=600, embed=True):
    """
    Convenience: save mp4 then return a Jupyter-embeddable Video object.
    """
    path = save_mp4(sim, out_path=out_path, fps=fps, dpi=dpi, step=step, marker_size=marker_size)
    return embed_mp4(path, width=width, embed=embed)


if __name__ == "__main__":
    sim = load_npz(r"data\dop853_N100_steps5000_dt0.001.npy.npz")
    plot_energy(sim)
    plot_std(sim, show_theory=True, k=1.0, v0=1.0, l=1.0)
    save_mp4(sim, out_path="sim_output.mp4", step=10)
