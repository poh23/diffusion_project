from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..shared_helpers import (
    _density_frame_index,
    _get_charge_value_ratio,
    _get_high_charge_fraction,
    _load_sim_auto,
    _parse_k_directory_name,
)


def compute_signed_mean_radius_difference(sim, times, k=None, snap_to_stride=True, scaled=False):
    if "charges" not in sim:
        raise KeyError("Simulation dict must contain 'charges' array.")

    sim_times = np.asarray(sim["times"])
    charges = np.asarray(sim["charges"])
    radii_all = np.asarray(sim["radii"]) if "radii" in sim else np.linalg.norm(np.asarray(sim["positions"]), axis=2)
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


def plot_signed_mean_radius_difference(sim, times, k=None, snap_to_stride=True, scaled=False, ax=None, show=True):
    actual_times, scores = compute_signed_mean_radius_difference(sim, times, k=k, snap_to_stride=snap_to_stride, scaled=scaled)
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
    ax.set_ylabel("Scaled Signed Mean Radius Difference" if scaled else "Signed Mean Radius Difference")
    ax.set_title(f"Positive: q={float(unique_charges[1]):g} farther out, Negative: q={float(unique_charges[0]):g} farther out")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    return fig, ax, actual_times, scores


def plot_signed_mean_radius_difference_vs_ratio(directory, times, k=None, snap_to_stride=True, scaled=False, ax=None, show=True):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".npz", ".h5", ".hdf5"})
    if not paths:
        raise ValueError(f"No supported simulation files found in directory: {directory}")

    requested_times = np.atleast_1d(times).astype(np.float64)
    ratios, score_rows, actual_time_rows, labels = [], [], [], []
    print(f"[plot] computing signed mean radius difference for {len(paths)} file(s)")
    for index, path in enumerate(paths, start=1):
        print(f"[plot] processing file {index}/{len(paths)}: {path}")
        sim = _load_sim_auto(path)
        if "charges" not in sim:
            print(f"Warning: missing charges in {path}; skipping.")
            continue
        try:
            ratio = _get_high_charge_fraction(sim["charges"])
            times_used, score = compute_signed_mean_radius_difference(sim, times=requested_times, k=k, snap_to_stride=snap_to_stride, scaled=scaled)
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
    for time_index in range(requested_times.shape[0]):
        ax.plot(ratios_arr, scores_arr[:, time_index], marker="o", linewidth=2, label=f"t~{float(np.mean(actual_times_arr[:, time_index])):.3g}")
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel(r"High-Charge Fraction $n_{\mathrm{high}} / N$")
    ax.set_ylabel("Scaled Signed Mean Radius Difference" if scaled else "Signed Mean Radius Difference")
    ax.set_title("Scaled Signed Mean Radius Difference vs Charge Ratio" if scaled else "Signed Mean Radius Difference vs Charge Ratio")
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


def plot_signed_mean_radius_difference_vs_charge_value_ratio(directory, times, k=None, snap_to_stride=True, scaled=False, ax=None, show=True):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".npz", ".h5", ".hdf5"})
    if not paths:
        raise ValueError(f"No supported simulation files found in directory: {directory}")

    requested_times = np.atleast_1d(times).astype(np.float64)
    x_values, score_rows, actual_time_rows, labels = [], [], [], []
    print(f"[plot] computing signed mean radius difference for {len(paths)} file(s)")
    for index, path in enumerate(paths, start=1):
        print(f"[plot] processing file {index}/{len(paths)}: {path}")
        sim = _load_sim_auto(path)
        if "charges" not in sim:
            print(f"Warning: missing charges in {path}; skipping.")
            continue
        try:
            x_value = _get_charge_value_ratio(sim["charges"])
            times_used, score = compute_signed_mean_radius_difference(sim, times=requested_times, k=k, snap_to_stride=snap_to_stride, scaled=scaled)
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
    for time_index in range(requested_times.shape[0]):
        ax.plot(x_arr, scores_arr[:, time_index], marker="o", linewidth=2, label=f"t~{float(np.mean(actual_times_arr[:, time_index])):.3g}")
    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=1)
    ax.set_xlabel(r"Charge-Magnitude Ratio $\max(|q|) / \min(|q|)$")
    ax.set_ylabel("Scaled Signed Mean Radius Difference" if scaled else "Signed Mean Radius Difference")
    ax.set_title("Scaled Signed Mean Radius Difference vs Charge-Value Ratio" if scaled else "Signed Mean Radius Difference vs Charge-Value Ratio")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    if requested_times.shape[0] == 1:
        return fig, ax, x_arr, scores_arr[:, 0], actual_times_arr[:, 0], labels_arr
    return fig, ax, x_arr, scores_arr.T, actual_times_arr.T, labels_arr


def plot_scaled_signed_mean_radius_difference_vs_ratio_by_k(directory, time, snap_to_stride=True, ax=None, show=True):
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
    print(f"[plot] computing scaled signed mean radius difference across {len(k_dirs)} k directory(ies)")
    for index, (k_value, k_dir) in enumerate(sorted(k_dirs, key=lambda item: item[0]), start=1):
        print(f"[plot] processing k directory {index}/{len(k_dirs)}: {k_dir}")
        _, _, ratios, scores, actual_times, labels = plot_signed_mean_radius_difference_vs_ratio(k_dir, times=time, k=k_value, snap_to_stride=snap_to_stride, scaled=True, show=False)
        series.append((k_value, ratios, scores, actual_times, labels))
    for k_value, ratios, scores, _, _ in series:
        ax.plot(ratios, scores, marker="o", linewidth=2, label=f"k={k_value:g}")
        m, b = np.polyfit(ratios, scores, 1)
        xfit = np.geomspace(np.min(ratios), np.max(ratios), 200)
        ax.plot(xfit, b + m * xfit, "--", linewidth=2, label=f"k={k_value:g} fit: y={b:.3g} + {m:.3g} x")
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


__all__ = [
    "compute_signed_mean_radius_difference",
    "plot_scaled_signed_mean_radius_difference_vs_ratio_by_k",
    "plot_signed_mean_radius_difference",
    "plot_signed_mean_radius_difference_vs_charge_value_ratio",
    "plot_signed_mean_radius_difference_vs_ratio",
]
