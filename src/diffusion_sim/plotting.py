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
        theo = (50* t) ** gamma
        plt.plot(t, theo, "--", linewidth=1, label="Theory (no diffusion)")
        if meta.get("diffusion", True):
            D = meta.get("diffusion_coeff", 0.0)
            theo_diff = (2.0 * D * t) ** 0.5
            plt.plot(t, theo_diff, ":",
                     linewidth=1, label="Diffusion theory")

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


def plot_density_vs_radius(
    sim,
    times,
    window_frac=0.05,
    min_window=5,
    ax=None,
    show=True,
    drop_zeros=False,
    min_points=10,
    snap_to_stride=True,
):
    """
    Plot density vs radius for selected times using a smoothed curve.
      sim: simulation dict with keys "times", "density" (steps,n), "positions" or "radii" (steps,n)
      times: iterable of times (float); nearest sample is used
      window_frac: fraction of points used as smoothing window for Gaussian filter
      min_window: minimum window size (int)
      drop_zeros: drop zero-density points (unbounded Voronoi cells)
      min_points: minimum finite points required to plot a curve
      snap_to_stride: snap requested times to the nearest valid density step when stride is used
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

    stride = None
    if snap_to_stride:
        stride = sim.get("meta", {}).get("density_stride")
        if not stride or stride <= 1:
            stride = None

    for t in times:
        idx = int(np.abs(sim_times - t).argmin())
        if stride is not None:
            idx = int(round(idx / stride) * stride)
            idx = max(0, min(idx, len(sim_times) - 1))
        r = radii[idx]
        d = density[idx]

        finite = np.isfinite(r) & np.isfinite(d)
        if drop_zeros:
            finite &= d > 0.0
        if finite.sum() < min_points:
            print(f"Warning: insufficient finite density points at t~{sim_times[idx]:.3g}; skipping.")
            continue
        r = r[finite]
        d = d[finite]

        order = np.argsort(r)
        r_sorted = r[order]
        d_sorted = d[order]

        if window_frac and window_frac > 0.0:
            window = max(min_window, int(len(d_sorted) * window_frac))
            window = min(window, len(d_sorted))
            if window < 1:
                window = 1
            if window % 2 == 0 and window > 1:
                window -= 1  # ensure odd for nicer symmetry
            sigma = window / 6.0  # approx converts window to stddev

            d_plot = gaussian_filter1d(d_sorted, sigma=sigma, mode="nearest")
        else:
            d_plot = d_sorted

        ax.plot(r_sorted, d_plot, label=f"t~{sim_times[idx]:.3g}")

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


def plot_scaled_density_vs_radius(
    sim,
    times,
    *,
    k=None,
    window_frac=0.05,
    min_window=5,
    ax_pair=None,
    show=True,
    drop_zeros=False,
    min_points=10,
    snap_to_stride=True,
):
    """
    Plot scaled density for selected times:
      1) density * t^(2/(k+2)) vs r / t^(1/(k+2))
      2) (density * t^(2/(k+2)))^(3/2) vs r^2 / t^(2/(k+2))
    """
    if "density" not in sim:
        raise KeyError("Simulation dict must contain 'density' array.")

    meta = sim.get("meta", {}) or {}
    if k is None:
        k = meta.get("k", None)
    if k is None:
        raise ValueError("k is required (pass k=... or include it in sim['meta']).")

    density = np.asarray(sim["density"])
    sim_times = np.asarray(sim["times"])
    if "radii" in sim:
        radii = np.asarray(sim["radii"])
    else:
        radii = np.linalg.norm(np.asarray(sim["positions"]), axis=2)

    if density.shape != radii.shape:
        raise ValueError(f"density shape {density.shape} and radii shape {radii.shape} differ.")

    created_fig = False
    if ax_pair is None:
        fig, (ax1, ax2) = plt.subplots(figsize=(12, 5), ncols=2)
        created_fig = True
    else:
        ax1, ax2 = ax_pair
        fig = ax1.figure

    gamma = 1.0 / (k + 2.0)
    times = np.atleast_1d(times)

    stride = None
    if snap_to_stride:
        stride = meta.get("density_stride")
        if not stride or stride <= 1:
            stride = None

    for t in times:
        idx = int(np.abs(sim_times - t).argmin())
        if stride is not None:
            idx = int(round(idx / stride) * stride)
            idx = max(0, min(idx, len(sim_times) - 1))

        t_val = float(sim_times[idx])
        if t_val <= 0.0:
            print(f"Warning: t={t_val:.3g} is not positive; skipping.")
            continue

        r = radii[idx]
        d = density[idx]

        finite = np.isfinite(r) & np.isfinite(d)
        if drop_zeros:
            finite &= d > 0.0
        if finite.sum() < min_points:
            print(f"Warning: insufficient finite density points at t~{t_val:.3g}; skipping.")
            continue

        r = r[finite]
        d = d[finite]

        t_scale = t_val ** gamma
        r_scaled = r / t_scale
        d_scaled = d * (t_val ** (2.0 * gamma))

        order = np.argsort(r_scaled)
        r_sorted = r_scaled[order]
        d_sorted = d_scaled[order]

        if window_frac and window_frac > 0.0:
            window = max(min_window, int(len(d_sorted) * window_frac))
            window = min(window, len(d_sorted))
            if window < 1:
                window = 1
            if window % 2 == 0 and window > 1:
                window -= 1
            sigma = window / 6.0
            d_plot = gaussian_filter1d(d_sorted, sigma=sigma, mode="nearest")
        else:
            d_plot = d_sorted

        ax1.plot(r_sorted, d_plot, label=f"t~{t_val:.3g}")

        r2_scaled = (r * r) / (t_val ** (2.0 * gamma))
        y2 = d_scaled ** 1.5
        order2 = np.argsort(r2_scaled)
        x2 = r2_scaled[order2]
        y2 = y2[order2]
        ax2.plot(x2, y2, label=f"t~{t_val:.3g}")

    ax1.set_xlabel(r"$r / t^{\frac{1}{k+2}}$")
    ax1.set_ylabel(r"$\rho t^{\frac{2}{k+2}}$")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_xlabel(r"$r^2 / t^{\frac{2}{k+2}}$")
    ax2.set_ylabel(r"$(\rho t^{\frac{2}{k+2}})^{3/2}$")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    fig.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    return fig, (ax1, ax2)
