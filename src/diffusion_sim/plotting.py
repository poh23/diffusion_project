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
):
    """
    Save an MP4 of particle motion (no trails) and return the output path.
    Optional t_start/t_end select an animation window in simulation time.
    Requested bounds are clamped to available times and snapped to nearest samples.
    """
    r_hist = np.asarray(sim["positions"])
    times = np.asarray(sim.get("times", []))

    if t_start is not None or t_end is not None:
        if times.size == 0:
            raise ValueError("t_start/t_end require sim['times'].")
        if times.shape[0] != r_hist.shape[0]:
            raise ValueError("sim['times'] length must match sim['positions'] frames.")

        t_min = float(np.min(times))
        t_max = float(np.max(times))

        if t_start is None:
            start_idx = 0
        else:
            t_start_clamped = float(np.clip(float(t_start), t_min, t_max))
            start_idx = int(np.abs(times - t_start_clamped).argmin())

        if t_end is None:
            end_idx = times.shape[0] - 1
        else:
            t_end_clamped = float(np.clip(float(t_end), t_min, t_max))
            end_idx = int(np.abs(times - t_end_clamped).argmin())

        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx

        r_hist = r_hist[start_idx:end_idx + 1]
        times = times[start_idx:end_idx + 1]

    n = r_hist.shape[1]

    r_view = r_hist[::step]
    if len(r_view) < 2:
        raise ValueError("Not enough frames to animate (try smaller step or more steps).")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # init with first frame so scatter has correct size
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
            else:
                c_min = float(np.min(charges))
                c_max = float(np.max(charges))
                if c_max > c_min:
                    norm = (charges - c_min) / (c_max - c_min)
                    colors = plt.cm.jet(norm)

    if colors is None:
        colors = plt.cm.jet(np.linspace(0, 1, n))

    particles = ax.scatter(r_view[0, :, 0], r_view[0, :, 1],
                           s=marker_size, c=colors)
    t_view = times[::step] if times.size else None
    time_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    # Initialize dynamic limits from first frame.
    x0 = r_view[0, :, 0]
    y0 = r_view[0, :, 1]
    x_min = float(np.min(x0))
    x_max = float(np.max(x0))
    y_min = float(np.min(y0))
    y_max = float(np.max(y0))
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    half_x = max(0.5 * (x_max - x_min), 1e-9)
    half_y = max(0.5 * (y_max - y_min), 1e-9)
    half = max(half_x, half_y) * (1.0 + axis_padding_frac)
    smoothed_cx = cx
    smoothed_cy = cy
    smoothed_half = half
    ax.set_xlim(smoothed_cx - smoothed_half, smoothed_cx + smoothed_half)
    ax.set_ylim(smoothed_cy - smoothed_half, smoothed_cy + smoothed_half)

    alpha = float(np.clip(axis_smoothing, 0.0, 1.0))

    def update(frame_idx):
        nonlocal smoothed_cx, smoothed_cy, smoothed_half
        frame = r_view[frame_idx]
        particles.set_offsets(frame)

        x = frame[:, 0]
        y = frame[:, 1]
        x_min_f = float(np.min(x))
        x_max_f = float(np.max(x))
        y_min_f = float(np.min(y))
        y_max_f = float(np.max(y))
        cx_f = 0.5 * (x_min_f + x_max_f)
        cy_f = 0.5 * (y_min_f + y_max_f)
        half_x_f = max(0.5 * (x_max_f - x_min_f), 1e-9)
        half_y_f = max(0.5 * (y_max_f - y_min_f), 1e-9)
        half_f = max(half_x_f, half_y_f) * (1.0 + axis_padding_frac)

        smoothed_cx = (1.0 - alpha) * smoothed_cx + alpha * cx_f
        smoothed_cy = (1.0 - alpha) * smoothed_cy + alpha * cy_f
        smoothed_half = (1.0 - alpha) * smoothed_half + alpha * half_f
        ax.set_xlim(smoothed_cx - smoothed_half, smoothed_cx + smoothed_half)
        ax.set_ylim(smoothed_cy - smoothed_half, smoothed_cy + smoothed_half)

        if t_view is not None and frame_idx < len(t_view):
            time_text.set_text(f"t = {t_view[frame_idx]:.5f}")
        else:
            time_text.set_text("")
        return (particles, time_text)

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
                marker_size=64, width=600, embed=True, t_start=None, t_end=None):
    """
    Convenience: save mp4 then return a Jupyter-embeddable Video object.
    """
    path = save_mp4(
        sim,
        out_path=out_path,
        fps=fps,
        dpi=dpi,
        step=step,
        marker_size=marker_size,
        t_start=t_start,
        t_end=t_end,
    )
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
    split_by_charge=True,
    charge_value=None,
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
      split_by_charge: if True and charges exist, plot a separate curve per charge population
      charge_value: if provided, plot only this charge population (requires sim["charges"])
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
    n = density.shape[1]

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

    charges = sim.get("charges")
    charge_groups: list[tuple[str, np.ndarray]] = [("all", np.ones(n, dtype=bool))]
    if charges is not None:
        charges = np.asarray(charges)
        if charges.shape == (n,):
            unique_charges = np.unique(charges)
            if charge_value is not None:
                mask = np.isclose(charges, charge_value)
                if not np.any(mask):
                    raise ValueError(
                        f"Requested charge_value={charge_value} not found. "
                        f"Available charges: {unique_charges.tolist()}"
                    )
                charge_groups = [(f"q={float(charge_value):g}", mask)]
            elif split_by_charge and unique_charges.size > 0:
                charge_groups = [(f"q={float(q):g}", np.isclose(charges, q)) for q in unique_charges]
    elif charge_value is not None:
        raise ValueError("charge_value was set but sim['charges'] is missing.")

    for t in times:
        idx = int(np.abs(sim_times - t).argmin())
        if stride is not None:
            idx = int(round(idx / stride) * stride)
            idx = max(0, min(idx, len(sim_times) - 1))
        r = radii[idx]
        d = density[idx]

        for charge_label, charge_mask in charge_groups:
            finite = charge_mask & np.isfinite(r) & np.isfinite(d)
            if drop_zeros:
                finite &= d > 0.0
            if finite.sum() < min_points:
                print(
                    f"Warning: insufficient finite density points at t~{sim_times[idx]:.3g}, "
                    f"{charge_label}; skipping."
                )
                continue
            r_group = r[finite]
            d_group = d[finite]

            order = np.argsort(r_group)
            r_sorted = r_group[order]
            d_sorted = d_group[order]

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

            if charge_label == "all":
                label = f"t~{sim_times[idx]:.3g}"
            else:
                label = f"{charge_label}, t~{sim_times[idx]:.3g}"
            ax.plot(r_sorted, d_plot, label=label)

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
    split_by_charge=True,
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
    n = density.shape[1]

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

    charge_groups: list[tuple[str, np.ndarray]] = [("all", np.ones(n, dtype=bool))]
    if split_by_charge:
        charges = sim.get("charges")
        if charges is not None:
            charges = np.asarray(charges)
            if charges.shape == (n,):
                unique_charges = np.unique(charges)
                if unique_charges.size > 0:
                    charge_groups = [(f"q={float(q):g}", np.isclose(charges, q)) for q in unique_charges]

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

        for charge_label, charge_mask in charge_groups:
            finite = charge_mask & np.isfinite(r) & np.isfinite(d)
            if drop_zeros:
                finite &= d > 0.0
            if finite.sum() < min_points:
                print(
                    f"Warning: insufficient finite density points at t~{t_val:.3g}, "
                    f"{charge_label}; skipping."
                )
                continue

            r_group = r[finite]
            d_group = d[finite]

            t_scale = t_val ** gamma
            r_scaled = r_group / t_scale
            d_scaled = d_group * (t_val ** (2.0 * gamma))

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

            if charge_label == "all":
                label = f"t~{t_val:.3g}"
            else:
                label = f"{charge_label}, t~{t_val:.3g}"
            ax1.plot(r_sorted, d_plot, label=label)

            r2_scaled = (r_group * r_group) / (t_val ** (2.0 * gamma))
            y2 = d_scaled ** 1.5
            order2 = np.argsort(r2_scaled)
            x2 = r2_scaled[order2]
            y2 = y2[order2]
            ax2.plot(x2, y2, label=label)

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


def plot_msd_by_charge(
    sim,
    particle_indices_by_charge=None,
    origin_window=None,
    same_plot=False,
    ax_list=None,
    show=True,
):
    """
    Plot mean-square displacement (MSD) vs time for one representative particle
    from each charge population.

    MSD is computed as |r(t) - r(0)|^2 by default. If `origin_window` is set,
    MSD is averaged over multiple initial time origins in the window:
        mean_{o in origins} |r(t+o) - r(o)|^2

    Args:
      sim: simulation dict (as returned by load_npz/load_h5), requires
           "positions", "times", and "charges".
      particle_indices_by_charge: optional dict mapping charge value -> global
           particle index to use for that charge. Defaults to first particle
           index found for each charge.
      origin_window: optional averaging window for time-origin averaging.
           - None: single origin at t=0.
           - int >= 1: number of initial frames used as origins.
           - float > 0: duration in simulation time; uses frames with
              times <= times[0] + origin_window.
      same_plot: if True, plot all charge curves on a single axis.
      ax_list: optional matplotlib axis/axes. If same_plot is True, provide
           one axis; otherwise provide one axis per charge.
      show: whether to call plt.show().
    """
    positions = np.asarray(sim["positions"])
    times = np.asarray(sim["times"])
    charges = sim.get("charges")

    if charges is None:
        raise KeyError("Simulation dict must contain 'charges' for charge-wise MSD.")
    charges = np.asarray(charges)

    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(
            f"Expected positions shape (steps, n_particles, 2), got {positions.shape}."
        )
    if times.ndim != 1 or times.shape[0] != positions.shape[0]:
        raise ValueError("sim['times'] must be 1D with length equal to positions steps.")
    if charges.shape != (positions.shape[1],):
        raise ValueError("sim['charges'] must have length equal to n_particles.")

    unique_charges = np.unique(charges)
    if unique_charges.size == 0:
        raise ValueError("No charge groups found in sim['charges'].")

    selected_indices = {}
    for q in unique_charges:
        members = np.flatnonzero(np.isclose(charges, q))
        if members.size == 0:
            continue
        if particle_indices_by_charge is None or q not in particle_indices_by_charge:
            idx = int(members[0])
        else:
            idx = int(particle_indices_by_charge[q])
            if idx < 0 or idx >= positions.shape[1]:
                raise ValueError(f"Index {idx} for charge {q} is out of range.")
            if not np.isclose(charges[idx], q):
                raise ValueError(
                    f"Index {idx} has charge {charges[idx]}, expected charge {q}."
                )
        selected_indices[q] = idx

    if not selected_indices:
        raise ValueError("Could not select representative particles for charge groups.")

    n_charges = len(unique_charges)
    created_fig = False
    if ax_list is None:
        if same_plot:
            fig, ax = plt.subplots(figsize=(8, 4))
            axes = np.asarray([ax])
        else:
            fig, axes = plt.subplots(
                nrows=n_charges,
                ncols=1,
                figsize=(8, max(3, 3 * n_charges)),
                squeeze=False,
                sharex=True,
            )
            axes = axes[:, 0]
        created_fig = True
    else:
        axes = np.asarray(ax_list).reshape(-1)
        expected_axes = 1 if same_plot else n_charges
        if axes.shape[0] != expected_axes:
            raise ValueError(f"ax_list has {axes.shape[0]} axes, expected {expected_axes}.")
        fig = axes[0].figure

    if origin_window is None:
        n_origins = 1
    elif isinstance(origin_window, (int, np.integer)):
        n_origins = int(origin_window)
    elif isinstance(origin_window, (float, np.floating)):
        if origin_window <= 0:
            raise ValueError("origin_window duration must be > 0.")
        t_limit = float(times[0]) + float(origin_window)
        n_origins = int(np.count_nonzero(times <= t_limit))
    else:
        raise TypeError("origin_window must be None, int, or float.")

    if n_origins < 1:
        raise ValueError("origin_window selected zero origins; increase the window.")
    if n_origins > times.shape[0]:
        n_origins = times.shape[0]

    for i, q in enumerate(unique_charges):
        ax = axes[0] if same_plot else axes[i]
        idx = selected_indices[q]
        traj = positions[:, idx, :]  # (steps, 2)

        if n_origins == 1:
            disp = traj - traj[0]
            msd = np.sum(disp * disp, axis=1)
            t_plot = times - times[0]
        else:
            # Average over early origins and align by lag.
            max_lag = times.shape[0] - n_origins
            msd_accum = np.zeros(max_lag + 1, dtype=float)
            for origin in range(n_origins):
                disp = traj[origin:origin + max_lag + 1] - traj[origin]
                msd_accum += np.sum(disp * disp, axis=1)
            msd = msd_accum / float(n_origins)
            t_plot = times[: max_lag + 1] - times[0]

        label = f"q={float(q):g}, idx={idx}"
        if n_origins > 1:
            label += f", origins={n_origins}"
        ax.plot(t_plot, msd, linewidth=2, label=label)
        ax.set_ylabel("MSD")
        if not same_plot:
            ax.set_title(f"Charge q={float(q):g}")
        ax.grid(True, alpha=0.3)

    if same_plot:
        axes[0].set_title("MSD by charge")
        axes[0].set_xlabel("Time")
        axes[0].legend()
    else:
        for ax in axes:
            ax.legend()
        axes[-1].set_xlabel("Time")
    fig.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    return fig, axes
