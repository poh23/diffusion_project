import os
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

import imageio_ffmpeg
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Only needed for notebook embedding
from IPython.display import Video

from .io.h5 import load_h5
from .io.npz import load_npz
from .postprocess.energy_components import compute_energy_component_series, compute_population_energy_series

plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def plot_energy(sim, scale_time=False):
    t_raw = np.asarray(sim["times"])
    e = np.asarray(sim["energy"])

    if scale_time:
        meta = sim.get("meta", {}) or {}
        k = meta.get("k", None)
        if k is not None:
            gamma = 1.0 / (k + 2.0)
            t = t_raw ** gamma
            if k == 0:
                charges = sim.get("charges", None)
                if charges is not None:
                    charges = np.asarray(charges, dtype=np.float64)
                    pair_charge_sum = 0.5 * (
                        np.sum(charges) ** 2 - np.sum(charges * charges)
                    )
                else:
                    charge_values = meta.get("charge_values", None)
                    charge_counts = meta.get("charge_counts", None)
                    if charge_values is None or charge_counts is None:
                        print("Warning: charge data not found, cannot apply k=0 energy shift.")
                        pair_charge_sum = None
                    else:
                        charge_values = np.asarray(charge_values, dtype=np.float64)
                        charge_counts = np.asarray(charge_counts, dtype=np.float64)
                        total_charge = np.sum(charge_values * charge_counts)
                        total_charge_sq = np.sum((charge_values ** 2) * charge_counts)
                        pair_charge_sum = 0.5 * (total_charge ** 2 - total_charge_sq)

                if pair_charge_sum is not None:
                    coupling = meta.get("v0", None)
                    if coupling is not None:
                        l = meta.get("l", None)
                        if l is None:
                            print("Warning: l not found in sim['meta'], cannot apply k=0 energy shift.")
                        else:
                            coupling = coupling * l
                            e = e.astype(np.float64, copy=True)
                            positive = t_raw > 0.0
                            e[positive] = (
                                e[positive]
                                + 0.5 * coupling * np.log(t_raw[positive]) * pair_charge_sum
                            )
                            if not np.all(positive):
                                e[~positive] = np.nan
                    else:
                        print("Warning: v0 not found in sim['meta'], cannot apply k=0 energy shift.")
            else:
                e = e * t_raw ** (k * gamma)  # Scale energy by t^(k/(k+2)) using the unscaled time.
        else:
            print("Warning: k not found in sim['meta'], cannot scale time.")
            t = t_raw
    else:
        t = t_raw

    plt.figure(figsize=(8, 4))
    plt.plot(t, e, linewidth=2)
    if scale_time and k is not None:
        plt.xlabel(f"Time (scaled by t^(1/(k+2)) with k={k})")
        plt.ylabel(f"Potential Energy (scaled with k={k})")
    else:
        plt.xlabel("Time")
        plt.ylabel("Potential Energy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _energy_components_from_sim(sim):
    energy_aa = sim.get("energy_aa")
    energy_ab = sim.get("energy_ab")
    energy_bb = sim.get("energy_bb")
    if energy_aa is not None and energy_ab is not None and energy_bb is not None:
        return (
            np.asarray(energy_aa, dtype=np.float64),
            np.asarray(energy_ab, dtype=np.float64),
            np.asarray(energy_bb, dtype=np.float64),
        )

    meta = sim.get("meta", {}) or {}
    charges = sim.get("charges")
    if charges is None:
        raise KeyError("Simulation dict must contain 'charges' to plot per-population energy.")

    missing = [name for name in ("k", "v0", "l", "r_floor") if meta.get(name) is None]
    if missing:
        raise KeyError(f"Simulation metadata is missing required fields: {', '.join(missing)}")

    _, energy_aa, energy_ab, energy_bb = compute_energy_component_series(
        np.asarray(sim["positions"]),
        k=meta["k"],
        v0=meta["v0"],
        l=meta["l"],
        r_floor=meta["r_floor"],
        charges=np.asarray(charges),
        population_values=meta.get("charge_values"),
    )
    return energy_aa, energy_ab, energy_bb


def _population_charge_pair_sums(sim):
    charges = sim.get("charges")
    meta = sim.get("meta", {}) or {}
    charge_values = meta.get("charge_values")
    charge_counts = meta.get("charge_counts")

    if charges is not None:
        charges = np.asarray(charges, dtype=np.float64)
        if charge_values is not None and len(charge_values) >= 2:
            charge_a = float(charge_values[0])
            charge_b = float(charge_values[1])
            mask_a = np.isclose(charges, charge_a)
            mask_b = np.isclose(charges, charge_b)
        else:
            unique = np.unique(charges)
            if unique.size < 2:
                raise ValueError("Need exactly two charge populations to scale per-population energy.")
            charge_a = float(unique[0])
            charge_b = float(unique[1])
            mask_a = np.isclose(charges, charge_a)
            mask_b = np.isclose(charges, charge_b)

        count_a = int(np.count_nonzero(mask_a))
        count_b = int(np.count_nonzero(mask_b))
    else:
        if charge_values is None or charge_counts is None or len(charge_values) < 2 or len(charge_counts) < 2:
            raise KeyError("Need charges or charge_values/charge_counts to scale per-population energy.")
        charge_a = float(charge_values[0])
        charge_b = float(charge_values[1])
        count_a = int(charge_counts[0])
        count_b = int(charge_counts[1])

    pair_sum_aa = 0.5 * count_a * max(0, count_a - 1) * (charge_a ** 2)
    pair_sum_bb = 0.5 * count_b * max(0, count_b - 1) * (charge_b ** 2)
    pair_sum_ab = count_a * count_b * charge_a * charge_b
    return pair_sum_aa, pair_sum_ab, pair_sum_bb


def plot_energy_by_charge(sim, scale_time=False):
    t_raw = np.asarray(sim["times"])
    energy_aa, energy_ab, energy_bb = _energy_components_from_sim(sim)
    energy_a, energy_b = compute_population_energy_series(energy_aa, energy_ab, energy_bb)

    meta = sim.get("meta", {}) or {}
    charge_values = meta.get("charge_values")
    if charge_values is not None and len(charge_values) >= 2:
        label_a = f"Population {charge_values[0]}"
        label_b = f"Population {charge_values[1]}"
    else:
        label_a = "Population A"
        label_b = "Population B"

    t = t_raw
    if scale_time:
        k = meta.get("k", None)
        if k is None:
            print("Warning: k not found in sim['meta'], cannot scale time.")
        else:
            gamma = 1.0 / (k + 2.0)
            t = t_raw ** gamma
            if k == 0:
                v0 = meta.get("v0", None)
                l = meta.get("l", None)
                if v0 is None or l is None:
                    print("Warning: v0/l not found in sim['meta'], cannot apply k=0 energy shift.")
                else:
                    pair_sum_aa, pair_sum_ab, pair_sum_bb = _population_charge_pair_sums(sim)
                    pair_sum_a = pair_sum_aa + 0.5 * pair_sum_ab
                    pair_sum_b = pair_sum_bb + 0.5 * pair_sum_ab
                    coupling = v0 * l
                    energy_a = energy_a.astype(np.float64, copy=True)
                    energy_b = energy_b.astype(np.float64, copy=True)
                    positive = t_raw > 0.0
                    energy_a[positive] = (
                        energy_a[positive]
                        + 0.5 * coupling * np.log(t_raw[positive]) * pair_sum_a
                    )
                    energy_b[positive] = (
                        energy_b[positive]
                        + 0.5 * coupling * np.log(t_raw[positive]) * pair_sum_b
                    )
                    if not np.all(positive):
                        energy_a[~positive] = np.nan
                        energy_b[~positive] = np.nan
            else:
                scale = t_raw ** (k * gamma)
                energy_a = energy_a * scale
                energy_b = energy_b * scale

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(t, energy_a, linewidth=2, label=label_a)
    axes[1].plot(t, energy_b, linewidth=2, label=label_b)

    axes[0].set_ylabel(label_a)
    axes[1].set_ylabel(label_b)
    if scale_time and meta.get("k", None) is not None:
        axes[1].set_xlabel(f"Time (scaled by t^(1/(k+2)) with k={meta['k']})")
    else:
        axes[1].set_xlabel("Time")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

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


def _density_frame_index(sim_times, t, stride=None):
    idx = int(np.abs(sim_times - t).argmin())
    if stride is not None:
        idx = int(round(idx / stride) * stride)
        idx = max(0, min(idx, len(sim_times) - 1))
    return idx


def _load_sim_auto(path):
    path = Path(path)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_h5(path)
    if path.suffix.lower() == ".npz":
        return load_npz(path)
    raise ValueError(f"Unsupported simulation file type: {path}")


def _get_high_charge_fraction(charges):
    charges = np.asarray(charges)
    unique_charges = np.unique(charges)
    if unique_charges.size != 2:
        raise ValueError(
            f"Expected exactly two charge populations, found {unique_charges.size}: "
            f"{unique_charges.tolist()}"
        )
    high_charge = unique_charges[-1]
    return float(np.count_nonzero(np.isclose(charges, high_charge))) / float(charges.size)


def _get_charge_value_ratio(charges):
    charges = np.asarray(charges)
    unique_charges = np.unique(charges)
    if unique_charges.size != 2:
        raise ValueError(
            f"Expected exactly two charge populations, found {unique_charges.size}: "
            f"{unique_charges.tolist()}"
        )

    abs_values = np.sort(np.abs(unique_charges))
    if abs_values[0] <= 0.0:
        raise ValueError("Charge-value ratio is undefined when one charge magnitude is zero.")
    return float(abs_values[1] / abs_values[0])


def _parse_k_directory_name(path):
    name = Path(path).name
    if not name.startswith("k") or len(name) == 1:
        raise ValueError(f"Expected subdirectory name like 'k0' or 'k1.5', got: {name}")
    try:
        return float(name[1:])
    except ValueError as exc:
        raise ValueError(f"Expected subdirectory name like 'k0' or 'k1.5', got: {name}") from exc


def _radial_profile_from_density(radii, density, mask, bin_edges, drop_zeros=True, min_points=10):
    finite = mask & np.isfinite(radii) & np.isfinite(density)
    if drop_zeros:
        finite &= density > 0.0

    if np.count_nonzero(finite) < min_points:
        return None

    r_sel = radii[finite]
    d_sel = density[finite]

    counts, _ = np.histogram(r_sel, bins=bin_edges)
    density_sum, _ = np.histogram(r_sel, bins=bin_edges, weights=d_sel)

    rho = np.zeros(len(bin_edges) - 1, dtype=np.float64)
    valid = counts > 0
    rho[valid] = density_sum[valid] / counts[valid]
    return rho


def compute_signed_radial_wasserstein(
    sim,
    times,
    k=None,
    n_bins=100,
    drop_zeros=True,
    min_points=10,
    snap_to_stride=True,
    scaled=False,
):
    """
    Compute a signed radial separation score between two charge populations.

    The magnitude is the 1D Wasserstein distance between the two radial
    distributions inferred from binned density-vs-radius curves. The sign is
    positive when the larger charge-value population has the larger mean radius.

    If scaled=True, the comparison is performed in the self-similar radial
    coordinate r / t^(1/(k+2)), so the returned score is also in scaled-radius
    units.

    Returns:
      actual_times: (m,) array of sampled times after nearest-frame snapping
      score:        (m,) array of signed Wasserstein distances
    """
    if "density" not in sim:
        raise KeyError("Simulation dict must contain 'density' array.")
    if "charges" not in sim:
        raise KeyError("Simulation dict must contain 'charges' array.")

    density_all = np.asarray(sim["density"])
    sim_times = np.asarray(sim["times"])
    charges = np.asarray(sim["charges"])

    if "radii" in sim:
        radii_all = np.asarray(sim["radii"])
    else:
        radii_all = np.linalg.norm(np.asarray(sim["positions"]), axis=2)

    if density_all.shape != radii_all.shape:
        raise ValueError(f"density shape {density_all.shape} and radii shape {radii_all.shape} differ.")
    if density_all.ndim != 2:
        raise ValueError(f"Expected density with shape (steps, n), got {density_all.shape}.")
    if sim_times.ndim != 1 or sim_times.shape[0] != density_all.shape[0]:
        raise ValueError("sim['times'] must be 1D with length equal to density steps.")
    if charges.shape != (density_all.shape[1],):
        raise ValueError("sim['charges'] must have length equal to n_particles.")
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2.")

    meta = sim.get("meta", {}) or {}
    if scaled:
        if k is None:
            k = meta.get("k", None)
        if k is None:
            raise ValueError("k is required when scaled=True (pass k=... or include it in sim['meta']).")
        gamma = 1.0 / (k + 2.0)

    unique_charges = np.unique(charges)
    if unique_charges.size != 2:
        raise ValueError(
            f"Expected exactly two charge populations, found {unique_charges.size}: "
            f"{unique_charges.tolist()}"
        )

    stride = None
    if snap_to_stride:
        stride = sim.get("meta", {}).get("density_stride")
        if not stride or stride <= 1:
            stride = None

    charge_lo = unique_charges[0]
    charge_hi = unique_charges[1]
    mask_lo = np.isclose(charges, charge_lo)
    mask_hi = np.isclose(charges, charge_hi)

    requested_times = np.atleast_1d(times)
    actual_times = np.empty(requested_times.shape[0], dtype=np.float64)
    scores = np.full(requested_times.shape[0], np.nan, dtype=np.float64)

    for i, t in enumerate(requested_times):
        idx = _density_frame_index(sim_times, t, stride=stride)
        actual_times[i] = float(sim_times[idx])

        r = radii_all[idx]
        d = density_all[idx]
        t_val = actual_times[i]

        if scaled:
            if t_val <= 0.0:
                print(f"Warning: t={t_val:.3g} is not positive; skipping.")
                continue
            t_scale = t_val ** gamma
            r = r / t_scale
            d = d * (t_val ** (2.0 * gamma))

        finite_all = np.isfinite(r) & np.isfinite(d)
        if drop_zeros:
            finite_all &= d > 0.0
        if np.count_nonzero(finite_all) < 2:
            print(f"Warning: insufficient finite density points at t~{actual_times[i]:.3g}; skipping.")
            continue

        r_valid = r[finite_all]
        r_min = float(np.min(r_valid))
        r_max = float(np.max(r_valid))
        if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= r_min:
            print(f"Warning: invalid radial range at t~{actual_times[i]:.3g}; skipping.")
            continue

        bin_edges = np.linspace(r_min, r_max, int(n_bins) + 1, dtype=np.float64)
        bin_widths = np.diff(bin_edges)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        rho_lo = _radial_profile_from_density(
            r,
            d,
            mask_lo,
            bin_edges,
            drop_zeros=drop_zeros,
            min_points=min_points,
        )
        rho_hi = _radial_profile_from_density(
            r,
            d,
            mask_hi,
            bin_edges,
            drop_zeros=drop_zeros,
            min_points=min_points,
        )

        if rho_lo is None or rho_hi is None:
            print(
                f"Warning: insufficient points per charge group at t~{actual_times[i]:.3g}; skipping."
            )
            continue

        mass_lo = bin_centers * rho_lo * bin_widths
        mass_hi = bin_centers * rho_hi * bin_widths

        mass_lo = np.where(np.isfinite(mass_lo) & (mass_lo > 0.0), mass_lo, 0.0)
        mass_hi = np.where(np.isfinite(mass_hi) & (mass_hi > 0.0), mass_hi, 0.0)

        total_lo = float(np.sum(mass_lo))
        total_hi = float(np.sum(mass_hi))
        if total_lo <= 0.0 or total_hi <= 0.0:
            print(f"Warning: zero radial mass at t~{actual_times[i]:.3g}; skipping.")
            continue

        mass_lo /= total_lo
        mass_hi /= total_hi

        cdf_lo = np.cumsum(mass_lo)
        cdf_hi = np.cumsum(mass_hi)
        w1 = float(np.sum(np.abs(cdf_hi - cdf_lo) * bin_widths))

        mean_lo = float(np.sum(bin_centers * mass_lo))
        mean_hi = float(np.sum(bin_centers * mass_hi))
        scores[i] = float(np.sign(mean_hi - mean_lo) * w1)

    return actual_times, scores


def compute_signed_mean_radius_difference(
    sim,
    times,
    k=None,
    snap_to_stride=True,
    scaled=False,
):
    """
    Compute the signed mean-radius difference between two charge populations.

    The score is:
      mean_radius(high-charge population) - mean_radius(low-charge population)

    If scaled=True, the comparison is performed in the self-similar radial
    coordinate r / t^(1/(k+2)).

    Returns:
      actual_times: (m,) array of sampled times after nearest-frame snapping
      score:        (m,) array of signed mean-radius differences
    """
    if "charges" not in sim:
        raise KeyError("Simulation dict must contain 'charges' array.")

    sim_times = np.asarray(sim["times"])
    charges = np.asarray(sim["charges"])

    if "radii" in sim:
        radii_all = np.asarray(sim["radii"])
    else:
        radii_all = np.linalg.norm(np.asarray(sim["positions"]), axis=2)

    if radii_all.ndim != 2:
        raise ValueError(f"Expected radii with shape (steps, n), got {radii_all.shape}.")
    if sim_times.ndim != 1 or sim_times.shape[0] != radii_all.shape[0]:
        raise ValueError("sim['times'] must be 1D with length equal to radii steps.")
    if charges.shape != (radii_all.shape[1],):
        raise ValueError("sim['charges'] must have length equal to n_particles.")

    meta = sim.get("meta", {}) or {}
    if scaled:
        if k is None:
            k = meta.get("k", None)
        if k is None:
            raise ValueError("k is required when scaled=True (pass k=... or include it in sim['meta']).")
        gamma = 1.0 / (k + 2.0)

    unique_charges = np.unique(charges)
    if unique_charges.size != 2:
        raise ValueError(
            f"Expected exactly two charge populations, found {unique_charges.size}: "
            f"{unique_charges.tolist()}"
        )

    stride = None
    if snap_to_stride:
        stride = sim.get("meta", {}).get("density_stride")
        if not stride or stride <= 1:
            stride = None

    mask_lo = np.isclose(charges, unique_charges[0])
    mask_hi = np.isclose(charges, unique_charges[1])

    requested_times = np.atleast_1d(times)
    actual_times = np.empty(requested_times.shape[0], dtype=np.float64)
    scores = np.full(requested_times.shape[0], np.nan, dtype=np.float64)

    for i, t in enumerate(requested_times):
        idx = _density_frame_index(sim_times, t, stride=stride)
        t_val = float(sim_times[idx])
        actual_times[i] = t_val

        r = radii_all[idx]
        if scaled:
            if t_val <= 0.0:
                print(f"Warning: t={t_val:.3g} is not positive; skipping.")
                continue
            r = r / (t_val ** gamma)

        r_lo = r[mask_lo & np.isfinite(r)]
        r_hi = r[mask_hi & np.isfinite(r)]
        if r_lo.size == 0 or r_hi.size == 0:
            print(f"Warning: insufficient finite radii per charge group at t~{t_val:.3g}; skipping.")
            continue

        scores[i] = float(np.mean(r_hi) - np.mean(r_lo))

    return actual_times, scores


def plot_signed_radial_wasserstein(
    sim,
    times,
    k=None,
    n_bins=100,
    drop_zeros=True,
    min_points=10,
    snap_to_stride=True,
    scaled=False,
    ax=None,
    show=True,
):
    """
    Plot the signed radial Wasserstein separation score vs time.

    Positive values mean the larger charge-value population is farther out
    on average. Negative values mean the smaller charge-value population is
    farther out.
    """
    actual_times, scores = compute_signed_radial_wasserstein(
        sim,
        times,
        k=k,
        n_bins=n_bins,
        drop_zeros=drop_zeros,
        min_points=min_points,
        snap_to_stride=snap_to_stride,
        scaled=scaled,
    )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure

    charges = np.asarray(sim["charges"])
    unique_charges = np.unique(charges)
    ax.plot(actual_times, scores, marker="o", linewidth=2)
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    if scaled:
        ax.set_ylabel("Scaled Signed Radial Wasserstein")
    else:
        ax.set_ylabel("Signed Radial Wasserstein")
    ax.set_title(
        f"Positive: q={float(unique_charges[1]):g} farther out, "
        f"Negative: q={float(unique_charges[0]):g} farther out"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    return fig, ax, actual_times, scores


def plot_signed_mean_radius_difference(
    sim,
    times,
    k=None,
    snap_to_stride=True,
    scaled=False,
    ax=None,
    show=True,
):
    """
    Plot the signed mean-radius difference vs time.

    Positive values mean the larger charge-value population is farther out
    on average. Negative values mean the smaller charge-value population is
    farther out.
    """
    actual_times, scores = compute_signed_mean_radius_difference(
        sim,
        times,
        k=k,
        snap_to_stride=snap_to_stride,
        scaled=scaled,
    )

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure

    charges = np.asarray(sim["charges"])
    unique_charges = np.unique(charges)
    ax.plot(actual_times, scores, marker="o", linewidth=2)
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    if scaled:
        ax.set_ylabel("Scaled Signed Mean Radius Difference")
    else:
        ax.set_ylabel("Signed Mean Radius Difference")
    ax.set_title(
        f"Positive: q={float(unique_charges[1]):g} farther out, "
        f"Negative: q={float(unique_charges[0]):g} farther out"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    return fig, ax, actual_times, scores


def plot_signed_radial_wasserstein_vs_ratio(
    directory,
    times,
    k=None,
    n_bins=100,
    drop_zeros=True,
    min_points=10,
    snap_to_stride=True,
    scaled=False,
    ax=None,
    show=True,
):
    """
    Plot the signed radial Wasserstein score at one or more selected times
    across all supported simulation files in a directory.

    X-axis: fraction of the larger-charge population, n_high / N
    Y-axis: signed radial Wasserstein score at the nearest saved frame(s)
    to the requested time(s)

    If scaled=True, the score is computed in the self-similar radial coordinate
    r / t^(1/(k+2)).
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")

    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".npz", ".h5", ".hdf5"}
    )
    if not paths:
        raise ValueError(f"No supported simulation files found in directory: {directory}")

    requested_times = np.atleast_1d(times).astype(np.float64)
    n_times = requested_times.shape[0]

    ratios = []
    score_rows = []
    actual_time_rows = []
    labels = []

    print(f"[plot] computing signed radial Wasserstein for {len(paths)} file(s)")
    for index, path in enumerate(paths, start=1):
        print(f"[plot] processing file {index}/{len(paths)}: {path}")
        sim = _load_sim_auto(path)
        if "charges" not in sim:
            print(f"Warning: missing charges in {path}; skipping.")
            continue
        try:
            ratio = _get_high_charge_fraction(sim["charges"])
            times_used, score = compute_signed_radial_wasserstein(
                sim,
                times=requested_times,
                k=k,
                n_bins=n_bins,
                drop_zeros=drop_zeros,
                min_points=min_points,
                snap_to_stride=snap_to_stride,
                scaled=scaled,
            )
        except ValueError as exc:
            print(f"Warning: {path}: {exc}; skipping.")
            continue

        if np.any(~np.isfinite(score)):
            print(f"Warning: could not compute a finite score for {path}; skipping.")
            continue

        ratios.append(ratio)
        score_rows.append(np.asarray(score, dtype=np.float64))
        actual_time_rows.append(np.asarray(times_used, dtype=np.float64))
        labels.append(path.name)

    if not ratios:
        raise ValueError("No valid files produced a finite signed radial Wasserstein score.")

    order = np.argsort(ratios)
    ratios_arr = np.asarray(ratios, dtype=np.float64)[order]
    scores_arr = np.asarray(score_rows, dtype=np.float64)[order]
    actual_times_arr = np.asarray(actual_time_rows, dtype=np.float64)[order]
    labels_arr = np.asarray(labels, dtype=object)[order]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure

    for time_index in range(n_times):
        label = f"t~{float(np.mean(actual_times_arr[:, time_index])):.3g}"
        ax.plot(ratios_arr, scores_arr[:, time_index], marker="o", linewidth=2, label=label)
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel(r"High-Charge Fraction $n_{\mathrm{high}} / N$")
    if scaled:
        ax.set_ylabel("Scaled Signed Radial Wasserstein")
        ax.set_title("Scaled Signed Radial Wasserstein vs Charge Ratio")
    else:
        ax.set_ylabel("Signed Radial Wasserstein")
        ax.set_title("Signed Radial Wasserstein vs Charge Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    if n_times == 1:
        return fig, ax, ratios_arr, scores_arr[:, 0], actual_times_arr[:, 0], labels_arr
    return fig, ax, ratios_arr, scores_arr.T, actual_times_arr.T, labels_arr


def plot_signed_mean_radius_difference_vs_ratio(
    directory,
    times,
    k=None,
    snap_to_stride=True,
    scaled=False,
    ax=None,
    show=True,
):
    """
    Plot the signed mean-radius difference at one or more selected times
    across all supported simulation files in a directory.

    X-axis: fraction of the larger-charge population, n_high / N
    Y-axis: signed mean-radius difference at the nearest saved frame(s)
    to the requested time(s)
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")

    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".npz", ".h5", ".hdf5"}
    )
    if not paths:
        raise ValueError(f"No supported simulation files found in directory: {directory}")

    requested_times = np.atleast_1d(times).astype(np.float64)
    n_times = requested_times.shape[0]

    ratios = []
    score_rows = []
    actual_time_rows = []
    labels = []

    print(f"[plot] computing signed mean radius difference for {len(paths)} file(s)")
    for index, path in enumerate(paths, start=1):
        print(f"[plot] processing file {index}/{len(paths)}: {path}")
        sim = _load_sim_auto(path)
        if "charges" not in sim:
            print(f"Warning: missing charges in {path}; skipping.")
            continue
        try:
            ratio = _get_high_charge_fraction(sim["charges"])
            times_used, score = compute_signed_mean_radius_difference(
                sim,
                times=requested_times,
                k=k,
                snap_to_stride=snap_to_stride,
                scaled=scaled,
            )
        except ValueError as exc:
            print(f"Warning: {path}: {exc}; skipping.")
            continue

        if np.any(~np.isfinite(score)):
            print(f"Warning: could not compute a finite score for {path}; skipping.")
            continue

        ratios.append(ratio)
        score_rows.append(np.asarray(score, dtype=np.float64))
        actual_time_rows.append(np.asarray(times_used, dtype=np.float64))
        labels.append(path.name)

    if not ratios:
        raise ValueError("No valid files produced a finite signed mean-radius difference.")

    order = np.argsort(ratios)
    ratios_arr = np.asarray(ratios, dtype=np.float64)[order]
    scores_arr = np.asarray(score_rows, dtype=np.float64)[order]
    actual_times_arr = np.asarray(actual_time_rows, dtype=np.float64)[order]
    labels_arr = np.asarray(labels, dtype=object)[order]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure

    for time_index in range(n_times):
        label = f"t~{float(np.mean(actual_times_arr[:, time_index])):.3g}"
        ax.plot(ratios_arr, scores_arr[:, time_index], marker="o", linewidth=2, label=label)
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel(r"High-Charge Fraction $n_{\mathrm{high}} / N$")
    if scaled:
        ax.set_ylabel("Scaled Signed Mean Radius Difference")
        ax.set_title("Scaled Signed Mean Radius Difference vs Charge Ratio")
    else:
        ax.set_ylabel("Signed Mean Radius Difference")
        ax.set_title("Signed Mean Radius Difference vs Charge Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    if n_times == 1:
        return fig, ax, ratios_arr, scores_arr[:, 0], actual_times_arr[:, 0], labels_arr
    return fig, ax, ratios_arr, scores_arr.T, actual_times_arr.T, labels_arr


def plot_signed_mean_radius_difference_vs_charge_value_ratio(
    directory,
    times,
    k=None,
    snap_to_stride=True,
    scaled=False,
    ax=None,
    show=True,
):
    """
    Plot the signed mean-radius difference at one or more selected times
    across all supported simulation files in a directory.

    X-axis: ratio of charge magnitudes, max(|q|) / min(|q|)
    Y-axis: signed mean-radius difference at the nearest saved frame(s)
    to the requested time(s)
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")

    paths = sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".npz", ".h5", ".hdf5"}
    )
    if not paths:
        raise ValueError(f"No supported simulation files found in directory: {directory}")

    requested_times = np.atleast_1d(times).astype(np.float64)
    n_times = requested_times.shape[0]

    x_values = []
    score_rows = []
    actual_time_rows = []
    labels = []

    print(f"[plot] computing signed mean radius difference for {len(paths)} file(s)")
    for index, path in enumerate(paths, start=1):
        print(f"[plot] processing file {index}/{len(paths)}: {path}")
        sim = _load_sim_auto(path)
        if "charges" not in sim:
            print(f"Warning: missing charges in {path}; skipping.")
            continue
        try:
            x_value = _get_charge_value_ratio(sim["charges"])
            times_used, score = compute_signed_mean_radius_difference(
                sim,
                times=requested_times,
                k=k,
                snap_to_stride=snap_to_stride,
                scaled=scaled,
            )
        except ValueError as exc:
            print(f"Warning: {path}: {exc}; skipping.")
            continue

        if np.any(~np.isfinite(score)):
            print(f"Warning: could not compute a finite score for {path}; skipping.")
            continue

        x_values.append(x_value)
        score_rows.append(np.asarray(score, dtype=np.float64))
        actual_time_rows.append(np.asarray(times_used, dtype=np.float64))
        labels.append(path.name)

    if not x_values:
        raise ValueError("No valid files produced a finite signed mean-radius difference.")

    order = np.argsort(x_values)
    x_arr = np.asarray(x_values, dtype=np.float64)[order]
    scores_arr = np.asarray(score_rows, dtype=np.float64)[order]
    actual_times_arr = np.asarray(actual_time_rows, dtype=np.float64)[order]
    labels_arr = np.asarray(labels, dtype=object)[order]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure

    for time_index in range(n_times):
        label = f"t~{float(np.mean(actual_times_arr[:, time_index])):.3g}"
        ax.plot(x_arr, scores_arr[:, time_index], marker="o", linewidth=2, label=label)
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Charge-Magnitude Ratio $\max(|q|) / \min(|q|)$")
    if scaled:
        ax.set_ylabel("Scaled Signed Mean Radius Difference")
        ax.set_title("Scaled Signed Mean Radius Difference vs Charge-Value Ratio")
    else:
        ax.set_ylabel("Signed Mean Radius Difference")
        ax.set_title("Signed Mean Radius Difference vs Charge-Value Ratio")
    ax.grid(True, alpha=0.3)
    #ax.loglog()
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    if n_times == 1:
        return fig, ax, x_arr, scores_arr[:, 0], actual_times_arr[:, 0], labels_arr
    return fig, ax, x_arr, scores_arr.T, actual_times_arr.T, labels_arr


def plot_scaled_signed_radial_wasserstein_vs_ratio_by_k(
    directory,
    time,
    n_bins=100,
    drop_zeros=True,
    min_points=10,
    snap_to_stride=True,
    ax=None,
    show=True,
):
    """
    Plot scaled signed radial Wasserstein vs n_high / N for multiple k values.

    Expects `directory` to contain subdirectories named like `k0`, `k1`, `k1.5`.
    Each such subdirectory is treated as a separate family of runs with fixed k.
    One line is drawn per k value on the same axes.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")

    k_dirs = []
    for path in sorted(child for child in directory.iterdir() if child.is_dir()):
        try:
            k_value = _parse_k_directory_name(path)
        except ValueError:
            continue
        k_dirs.append((k_value, path))

    if not k_dirs:
        raise ValueError(f"No k* subdirectories found in directory: {directory}")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure

    series = []
    print(f"[plot] computing scaled signed radial Wasserstein across {len(k_dirs)} k directory(ies)")
    for index, (k_value, k_dir) in enumerate(sorted(k_dirs, key=lambda item: item[0]), start=1):
        print(f"[plot] processing k directory {index}/{len(k_dirs)}: {k_dir}")
        _, _, ratios, scores, actual_times, labels = plot_signed_radial_wasserstein_vs_ratio(
            k_dir,
            times=time,
            k=k_value,
            n_bins=n_bins,
            drop_zeros=drop_zeros,
            min_points=min_points,
            snap_to_stride=snap_to_stride,
            scaled=True,
            show=False,
        )
        series.append((k_value, ratios, scores, actual_times, labels))

    for k_value, ratios, scores, _, _ in series:
        ax.plot(ratios, scores, marker="o", linewidth=2, label=f"k={k_value:g}")

    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel(r"High-Charge Fraction $n_{\mathrm{high}} / N$")
    ax.set_ylabel("Scaled Signed Radial Wasserstein")
    ax.set_title(f"Scaled Signed Radial Wasserstein vs Charge Ratio (target t={float(time):.3g})")
    ax.loglog()
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    return fig, ax, series


def plot_scaled_signed_mean_radius_difference_vs_ratio_by_k(
    directory,
    time,
    snap_to_stride=True,
    ax=None,
    show=True,
):
    """
    Plot scaled signed mean-radius difference vs n_high / N for multiple k values.

    Expects `directory` to contain subdirectories named like `k0`, `k1`, `k1.5`.
    Each such subdirectory is treated as a separate family of runs with fixed k.
    One line is drawn per k value on the same axes.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")

    k_dirs = []
    for path in sorted(child for child in directory.iterdir() if child.is_dir()):
        try:
            k_value = _parse_k_directory_name(path)
        except ValueError:
            continue
        k_dirs.append((k_value, path))

    if not k_dirs:
        raise ValueError(f"No k* subdirectories found in directory: {directory}")

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure

    series = []
    print(f"[plot] computing scaled signed mean radius difference across {len(k_dirs)} k directory(ies)")
    for index, (k_value, k_dir) in enumerate(sorted(k_dirs, key=lambda item: item[0]), start=1):
        print(f"[plot] processing k directory {index}/{len(k_dirs)}: {k_dir}")
        _, _, ratios, scores, actual_times, labels = plot_signed_mean_radius_difference_vs_ratio(
            k_dir,
            times=time,
            k=k_value,
            snap_to_stride=snap_to_stride,
            scaled=True,
            show=False,
        )
        series.append((k_value, ratios, scores, actual_times, labels))

    for k_value, ratios, scores, _, _ in series:
        ax.plot(ratios, scores, marker="o", linewidth=2, label=f"k={k_value:g}")
        # Fit N=700 (k=-1)                                                                                                                                     
        m, b = np.polyfit(ratios, scores, 1)                                                                                  
        xfit = np.geomspace(np.min(ratios), np.max(ratios), 200)                                                                                  
        yfit = b + m * xfit                                                                                                           
        ax.plot(xfit, yfit, "--", linewidth=2, label=f"k={k_value:g} fit: y={b:.3g} + {m:.3g} x")   

    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel(r"High-Charge Fraction $n_{\mathrm{high}} / N$")
    ax.set_ylabel("Scaled Signed Mean Radius Difference")
    ax.set_title(f"Scaled Signed Mean Radius Difference vs Charge Ratio (target t={float(time):.3g})")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.legend()
    fig.tight_layout()

    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    return fig, ax, series


def plot_density_vs_radius(
    sim,
    times,
    k=None,
    window_frac=0.05,
    min_window=5,
    ax=None,
    show=True,
    drop_zeros=False,
    min_points=10,
    snap_to_stride=True,
    split_by_charge=True,
    charge_value=None,
    scaled=False,
):
    """
    Plot density vs radius for selected times using a smoothed curve.
      sim: simulation dict with keys "times", "density" (steps,n), "positions" or "radii" (steps,n)
      times: iterable of times (float); nearest sample is used
      k: interaction exponent, required when scaled=True unless present in sim["meta"]
      window_frac: fraction of points used as smoothing window for Gaussian filter
      min_window: minimum window size (int)
      drop_zeros: drop zero-density points (unbounded Voronoi cells)
      min_points: minimum finite points required to plot a curve
      snap_to_stride: snap requested times to the nearest valid density step when stride is used
      split_by_charge: if True and charges exist, plot a separate curve per charge population
      charge_value: if provided, plot only this charge population (requires sim["charges"])
      scaled: if True, plot rho * t^(2/(k+2)) vs r / t^(1/(k+2))
    """
    if "density" not in sim:
        raise KeyError("Simulation dict must contain 'density' array.")

    meta = sim.get("meta", {}) or {}
    if scaled:
        if k is None:
            k = meta.get("k", None)
        if k is None:
            raise ValueError("k is required when scaled=True (pass k=... or include it in sim['meta']).")
        gamma = 1.0 / (k + 2.0)

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
        t_val = float(sim_times[idx])
        r = radii[idx]
        d = density[idx]

        if scaled and t_val <= 0.0:
            print(f"Warning: t={t_val:.3g} is not positive; skipping.")
            continue

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

            if scaled:
                t_scale = t_val ** gamma
                r_plot = r_sorted / t_scale
                d_plot = d_plot * (t_val ** (2.0 * gamma))
            else:
                r_plot = r_sorted

            if charge_label == "all":
                label = f"t~{t_val:.3g}"
            else:
                label = f"{charge_label}, t~{t_val:.3g}"
            ax.plot(r_plot, d_plot, label=label)

    if scaled:
        ax.set_xlabel(r"$r / t^{\frac{1}{k+2}}$")
        ax.set_ylabel(r"$\rho t^{\frac{2}{k+2}}$")
    else:
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
