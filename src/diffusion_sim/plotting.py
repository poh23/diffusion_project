import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

import imageio_ffmpeg
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Only needed for notebook embedding
from IPython.display import Video

from .io.npz import load_npz

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


def plot_density_vs_radius(sim, times, window_frac=0.05, min_window=5, ax=None, show=True):
    """
    Plot density vs radius for selected times using a smoothed curve.
      sim: simulation dict with keys "times", "density" (steps,n), "positions" or "radii" (steps,n)
      times: iterable of times (float); nearest sample is used
      window_frac: fraction of points used as smoothing window for Gaussian filter
      min_window: minimum window size (int)
    """
    if "density" not in sim:
        raise KeyError("Simulation dict must contain 'density' array.")

    density = np.asarray(sim["density"])
    sim_times = np.asarray(sim["times"])

    if "radii" in sim:
        radii = np.asarray(sim["radii"])
    else:
        radii = np.linalg.norm(np.asarray(sim["positions"]), axis=2)

    if density.shape != radii.shape:
        raise ValueError(f"density shape {density.shape} and radii shape {radii.shape} differ.")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        created_fig = True
    else:
        fig = ax.figure

    times = np.atleast_1d(times)

    for t in times:
        idx = int(np.abs(sim_times - t).argmin())
        r = radii[idx]
        d = density[idx]

        order = np.argsort(r)
        r_sorted = r[order]
        d_sorted = d[order]

        window = max(min_window, int(len(d_sorted) * window_frac))
        if window % 2 == 0:
            window += 1  # ensure odd for nicer symmetry
        sigma = window / 6.0  # approx converts window to stddev

        d_smooth = gaussian_filter1d(d_sorted, sigma=sigma, mode="nearest")
        ax.plot(r_sorted, d_smooth, label=f"t~{sim_times[idx]:.3g}")

    ax.set_xlabel("Radius")
    ax.set_ylabel("Density (1 / Voronoi area)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        # Only close if we created the figure; otherwise leave it open for caller.
        plt.close(fig)
    return fig, ax
