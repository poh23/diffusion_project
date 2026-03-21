import os
from pathlib import Path

import imageio_ffmpeg
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video
from matplotlib.animation import FFMpegWriter, FuncAnimation

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def save_mp4(
    sim,
    out_path="simulation.mp4",
    fps=30,
    dpi=120,
    step=10,
    marker_size=64,
    axis_smoothing=0.2,
    axis_padding_frac=0.05,
    t_start=None,
    t_end=None,
    scale=False,
    k=None,
):
    r_hist = np.asarray(sim["positions"])
    times = np.asarray(sim.get("times", []))
    scale_gamma = None

    if t_start is not None or t_end is not None:
        if times.size == 0:
            raise ValueError("t_start/t_end require sim['times'].")
        if times.shape[0] != r_hist.shape[0]:
            raise ValueError("sim['times'] length must match sim['positions'] frames.")
        t_min = float(np.min(times))
        t_max = float(np.max(times))
        start_idx = 0 if t_start is None else int(np.abs(times - float(np.clip(float(t_start), t_min, t_max))).argmin())
        end_idx = times.shape[0] - 1 if t_end is None else int(np.abs(times - float(np.clip(float(t_end), t_min, t_max))).argmin())
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        r_hist = r_hist[start_idx:end_idx + 1]
        times = times[start_idx:end_idx + 1]

    if scale:
        if times.size == 0:
            raise ValueError("scale=True requires sim['times'].")
        if times.shape[0] != r_hist.shape[0]:
            raise ValueError("sim['times'] length must match sim['positions'] frames when scale=True.")
        meta = sim.get("meta", {}) or {}
        k_val = k if k is not None else meta.get("k", None)
        if k_val is None:
            raise ValueError("k is required for scale=True (pass k=... or include sim['meta']['k']).")
        if float(k_val) == -2.0:
            raise ValueError("k=-2 is invalid for self-similar scaling (division by zero in exponent).")
        scale_gamma = 1.0 / (float(k_val) + 2.0)
        positive = times > 0.0
        if not np.any(positive):
            raise ValueError("scale=True dropped all frames because all times are <= 0.")
        r_hist = r_hist[positive] / (times[positive] ** scale_gamma)[:, None, None]
        times = times[positive]

    n = r_hist.shape[1]
    r_view = r_hist[::step]
    if len(r_view) < 2:
        raise ValueError("Not enough frames to animate (try smaller step or more steps).")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    colors = None
    charges = sim.get("charges")
    if charges is not None:
        charges = np.asarray(charges)
        if charges.shape[0] == n:
            unique = np.unique(charges)
            palette = ["#1f77b4", "#d62728"]
            if unique.size <= len(palette):
                color_map = {val: palette[i] for i, val in enumerate(unique)}
                colors = np.array([color_map[val] for val in charges], dtype=object)
    if colors is None:
        colors = plt.cm.jet(np.linspace(0, 1, n))

    particles = ax.scatter(r_view[0, :, 0], r_view[0, :, 1], s=marker_size, c=colors)
    t_view = times[::step] if times.size else None
    t_scaled_view = t_view ** scale_gamma if scale and t_view is not None else None
    time_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top")

    x0 = r_view[0, :, 0]
    y0 = r_view[0, :, 1]
    smoothed_cx = 0.5 * (float(np.min(x0)) + float(np.max(x0)))
    smoothed_cy = 0.5 * (float(np.min(y0)) + float(np.max(y0)))
    smoothed_half = max(0.5 * (float(np.max(x0)) - float(np.min(x0))), 0.5 * (float(np.max(y0)) - float(np.min(y0))), 1e-9) * (1.0 + axis_padding_frac)
    ax.set_xlim(smoothed_cx - smoothed_half, smoothed_cx + smoothed_half)
    ax.set_ylim(smoothed_cy - smoothed_half, smoothed_cy + smoothed_half)
    alpha = float(np.clip(axis_smoothing, 0.0, 1.0))

    def update(frame_idx):
        nonlocal smoothed_cx, smoothed_cy, smoothed_half
        frame = r_view[frame_idx]
        particles.set_offsets(frame)
        x = frame[:, 0]
        y = frame[:, 1]
        cx_f = 0.5 * (float(np.min(x)) + float(np.max(x)))
        cy_f = 0.5 * (float(np.min(y)) + float(np.max(y)))
        half_f = max(0.5 * (float(np.max(x)) - float(np.min(x))), 0.5 * (float(np.max(y)) - float(np.min(y))), 1e-9) * (1.0 + axis_padding_frac)
        smoothed_cx = (1.0 - alpha) * smoothed_cx + alpha * cx_f
        smoothed_cy = (1.0 - alpha) * smoothed_cy + alpha * cy_f
        smoothed_half = (1.0 - alpha) * smoothed_half + alpha * half_f
        ax.set_xlim(smoothed_cx - smoothed_half, smoothed_cx + smoothed_half)
        ax.set_ylim(smoothed_cy - smoothed_half, smoothed_cy + smoothed_half)
        if t_view is not None and frame_idx < len(t_view):
            time_text.set_text(
                f"t = {t_view[frame_idx]:.5f}, t_scaled = {t_scaled_view[frame_idx]:.5f}"
                if t_scaled_view is not None else f"t = {t_view[frame_idx]:.5f}"
            )
        else:
            time_text.set_text("")
        return (particles, time_text)

    ani = FuncAnimation(fig, update, frames=len(r_view), blit=True)
    out_path_obj = Path(out_path)
    if out_path_obj.suffix.lower() not in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
        out_path_obj = out_path_obj.with_suffix(".mp4")
    out_path = str(out_path_obj)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    ani.save(out_path, writer=FFMpegWriter(fps=fps, metadata={"artist": "you"}, bitrate=1800), dpi=dpi)
    plt.close(fig)
    print(f"Saved video: {out_path}")
    return out_path


def embed_mp4(path, width=600, embed=True):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return Video(path, width=width, embed=embed)


def animate_mp4(
    sim,
    out_path="simulation.mp4",
    fps=30,
    dpi=120,
    step=10,
    marker_size=64,
    width=600,
    embed=True,
    t_start=None,
    t_end=None,
    scale=False,
    k=None,
):
    path = save_mp4(
        sim,
        out_path=out_path,
        fps=fps,
        dpi=dpi,
        step=step,
        marker_size=marker_size,
        t_start=t_start,
        t_end=t_end,
        scale=scale,
        k=k,
    )
    return embed_mp4(path, width=width, embed=embed)


__all__ = ["animate_mp4", "embed_mp4", "save_mp4"]
