from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..shared_helpers import (
    _density_frame_index,
    _get_high_charge_fraction,
    _load_sim_auto,
    _parse_k_directory_name,
    _radial_profile_from_density,
)


def compute_signed_radial_wasserstein(sim, times, k=None, n_bins=100, drop_zeros=True, min_points=10, snap_to_stride=True, scaled=False):
    if "density" not in sim:
        raise KeyError("Simulation dict must contain 'density' array.")
    if "charges" not in sim:
        raise KeyError("Simulation dict must contain 'charges' array.")

    density_all = np.asarray(sim["density"])
    sim_times = np.asarray(sim["times"])
    charges = np.asarray(sim["charges"])
    radii_all = np.asarray(sim["radii"]) if "radii" in sim else np.linalg.norm(np.asarray(sim["positions"]), axis=2)
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
        raise ValueError(f"Expected exactly two charge populations, found {unique_charges.size}: {unique_charges.tolist()}")

    stride = sim.get("meta", {}).get("density_stride") if snap_to_stride else None
    if not stride or stride <= 1:
        stride = None
    mask_lo = np.isclose(charges, unique_charges[0])
    mask_hi = np.isclose(charges, unique_charges[1])

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
            r = r / (t_val ** gamma)
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
        rho_lo = _radial_profile_from_density(r, d, mask_lo, bin_edges, drop_zeros=drop_zeros, min_points=min_points)
        rho_hi = _radial_profile_from_density(r, d, mask_hi, bin_edges, drop_zeros=drop_zeros, min_points=min_points)
        if rho_lo is None or rho_hi is None:
            print(f"Warning: insufficient points per charge group at t~{actual_times[i]:.3g}; skipping.")
            continue
        mass_lo = np.where(np.isfinite(bin_centers * rho_lo * bin_widths) & (bin_centers * rho_lo * bin_widths > 0.0), bin_centers * rho_lo * bin_widths, 0.0)
        mass_hi = np.where(np.isfinite(bin_centers * rho_hi * bin_widths) & (bin_centers * rho_hi * bin_widths > 0.0), bin_centers * rho_hi * bin_widths, 0.0)
        total_lo = float(np.sum(mass_lo))
        total_hi = float(np.sum(mass_hi))
        if total_lo <= 0.0 or total_hi <= 0.0:
            print(f"Warning: zero radial mass at t~{actual_times[i]:.3g}; skipping.")
            continue
        mass_lo /= total_lo
        mass_hi /= total_hi
        w1 = float(np.sum(np.abs(np.cumsum(mass_hi) - np.cumsum(mass_lo)) * bin_widths))
        scores[i] = float(np.sign(np.sum(bin_centers * mass_hi) - np.sum(bin_centers * mass_lo)) * w1)

    return actual_times, scores


def plot_signed_radial_wasserstein(sim, times, k=None, n_bins=100, drop_zeros=True, min_points=10, snap_to_stride=True, scaled=False, ax=None, show=True):
    actual_times, scores = compute_signed_radial_wasserstein(sim, times, k=k, n_bins=n_bins, drop_zeros=drop_zeros, min_points=min_points, snap_to_stride=snap_to_stride, scaled=scaled)
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure
    unique_charges = np.unique(np.asarray(sim["charges"]))
    ax.plot(actual_times, scores, marker="o", linewidth=2)
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel("Time")
    ax.set_ylabel("Scaled Signed Radial Wasserstein" if scaled else "Signed Radial Wasserstein")
    ax.set_title(f"Positive: q={float(unique_charges[1]):g} farther out, Negative: q={float(unique_charges[0]):g} farther out")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    return fig, ax, actual_times, scores


def plot_signed_radial_wasserstein_vs_ratio(directory, times, k=None, n_bins=100, drop_zeros=True, min_points=10, snap_to_stride=True, scaled=False, ax=None, show=True):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".npz", ".h5", ".hdf5"})
    if not paths:
        raise ValueError(f"No supported simulation files found in directory: {directory}")

    requested_times = np.atleast_1d(times).astype(np.float64)
    ratios, score_rows, actual_time_rows, labels = [], [], [], []
    print(f"[plot] computing signed radial Wasserstein for {len(paths)} file(s)")
    for index, path in enumerate(paths, start=1):
        print(f"[plot] processing file {index}/{len(paths)}: {path}")
        sim = _load_sim_auto(path)
        if "charges" not in sim:
            print(f"Warning: missing charges in {path}; skipping.")
            continue
        try:
            ratio = _get_high_charge_fraction(sim["charges"])
            times_used, score = compute_signed_radial_wasserstein(sim, times=requested_times, k=k, n_bins=n_bins, drop_zeros=drop_zeros, min_points=min_points, snap_to_stride=snap_to_stride, scaled=scaled)
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
    for time_index in range(requested_times.shape[0]):
        ax.plot(ratios_arr, scores_arr[:, time_index], marker="o", linewidth=2, label=f"t~{float(np.mean(actual_times_arr[:, time_index])):.3g}")
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel(r"High-Charge Fraction $n_{\mathrm{high}} / N$")
    ax.set_ylabel("Scaled Signed Radial Wasserstein" if scaled else "Signed Radial Wasserstein")
    ax.set_title("Scaled Signed Radial Wasserstein vs Charge Ratio" if scaled else "Signed Radial Wasserstein vs Charge Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    if requested_times.shape[0] == 1:
        return fig, ax, ratios_arr, scores_arr[:, 0], actual_times_arr[:, 0], labels_arr
    return fig, ax, ratios_arr, scores_arr.T, actual_times_arr.T, labels_arr


def plot_scaled_signed_radial_wasserstein_vs_ratio_by_k(directory, time, n_bins=100, drop_zeros=True, min_points=10, snap_to_stride=True, ax=None, show=True):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")
    k_dirs = []
    for path in sorted(child for child in directory.iterdir() if child.is_dir()):
        try:
            k_dirs.append((_parse_k_directory_name(path), path))
        except ValueError:
            continue
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
        _, _, ratios, scores, actual_times, labels = plot_signed_radial_wasserstein_vs_ratio(k_dir, times=time, k=k_value, n_bins=n_bins, drop_zeros=drop_zeros, min_points=min_points, snap_to_stride=snap_to_stride, scaled=True, show=False)
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


__all__ = [
    "compute_signed_radial_wasserstein",
    "plot_scaled_signed_radial_wasserstein_vs_ratio_by_k",
    "plot_signed_radial_wasserstein",
    "plot_signed_radial_wasserstein_vs_ratio",
]
