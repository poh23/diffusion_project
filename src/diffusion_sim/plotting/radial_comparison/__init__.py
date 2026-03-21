import matplotlib.pyplot as plt
import numpy as np

from ...forces import compute_velocity_overdamped
from ..shared_helpers import _compute_low_and_high_inner_radii, _get_charge_value_ratio


def _infer_rescaled_from_meta(meta: dict) -> bool:
    if bool(meta.get("positions_rescaled", False)):
        return True
    return str(meta.get("coordinate_frame", "")).lower() in {"rescaled", "self_similar", "self-similar", "y"}


def _resolve_use_rescaled(mode: str, meta: dict) -> bool:
    mode = mode.lower()
    if mode == "yes":
        return True
    if mode == "no":
        return False
    if mode != "auto":
        raise ValueError("--use-rescaled-coords must be one of: auto, yes, no")
    return _infer_rescaled_from_meta(meta)


def _resolve_frame_indices(n_frames: int, frame: int | None, last_n_frames: int) -> np.ndarray:
    if n_frames <= 0:
        raise ValueError("No frames in simulation output.")
    if frame is not None:
        idx = int(frame)
        if idx < 0:
            idx = n_frames + idx
        if idx < 0 or idx >= n_frames:
            raise IndexError(f"--frame {frame} is out of range for {n_frames} frames.")
        return np.asarray([idx], dtype=np.int64)
    n = max(1, int(last_n_frames))
    return np.arange(max(0, n_frames - n), n_frames, dtype=np.int64)


def _radial_bin_stats(r: np.ndarray, f_rad: np.ndarray, edges: np.ndarray):
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(centers.shape, np.nan, dtype=np.float64)
    sem = np.full(centers.shape, np.nan, dtype=np.float64)
    counts = np.zeros(centers.shape, dtype=np.int64)
    for i in range(len(centers)):
        left = edges[i]
        right = edges[i + 1]
        mask = (r >= left) & (r <= right) if i == len(centers) - 1 else (r >= left) & (r < right)
        vals = f_rad[mask]
        n = int(vals.size)
        counts[i] = n
        if n == 0:
            continue
        means[i] = float(np.mean(vals))
        sem[i] = 0.0 if n == 1 else float(np.std(vals, ddof=1) / np.sqrt(n))
    return centers, means, sem, counts


def _crossings_with_reference(x: np.ndarray, y: np.ndarray, y_ref: np.ndarray) -> list[float]:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(y_ref)
    xv = x[valid]
    dv = y[valid] - y_ref[valid]
    if xv.size < 2:
        return []
    out = []
    for i in range(xv.size - 1):
        d0 = dv[i]
        d1 = dv[i + 1]
        x0 = xv[i]
        x1 = xv[i + 1]
        if d0 == 0.0:
            out.append(float(x0))
            continue
        if d0 * d1 < 0.0:
            out.append(float(x0 - d0 * (x1 - x0) / (d1 - d0)))
    return out


def plot_inner_radius_ratio_power_vs_charge_value_ratio(directory, time, low_percentile=95.0, high_inner_percentile=5.0, k=None, ax=None, show=True):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Expected a directory path, got: {directory}")
    paths = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".npz", ".h5", ".hdf5"})
    if not paths:
        raise ValueError(f"No supported simulation files found in directory: {directory}")

    x_values, y_values, actual_times, labels = [], [], [], []
    print(f"[plot] computing ((r2/r1)^(k+2)) for {len(paths)} file(s)")
    for index, path in enumerate(paths, start=1):
        print(f"[plot] processing file {index}/{len(paths)}: {path}")
        sim = _load_sim_auto(path)
        if "charges" not in sim:
            print(f"Warning: missing charges in {path}; skipping.")
            continue
        try:
            x_value = _get_charge_value_ratio(sim["charges"])
            t_used, r1, r2 = _compute_low_and_high_inner_radii(sim, time=time, low_percentile=low_percentile, high_inner_percentile=high_inner_percentile)
            k_value = float(k if k is not None else (sim.get("meta", {}) or {}).get("k"))
            if not np.isfinite(k_value):
                raise ValueError("k is missing or non-finite.")
            if np.isclose(k_value, -2.0):
                raise ValueError("k=-2 is invalid (division by zero in exponent).")
            if r1 <= 0.0:
                raise ValueError("r1 is non-positive, cannot form r2/r1.")
            y_value = float(r2 / r1)
        except ValueError as exc:
            print(f"Warning: {path}: {exc}; skipping.")
            continue
        if not np.isfinite(y_value):
            print(f"Warning: non-finite transformed ratio for {path}; skipping.")
            continue
        x_values.append(float(x_value))
        y_values.append(y_value)
        actual_times.append(t_used)
        labels.append(path.name)

    if not x_values:
        raise ValueError("No valid files produced a finite transformed radius ratio.")
    order = np.argsort(x_values)
    x_arr = np.asarray(x_values, dtype=np.float64)[order]
    y_arr = np.asarray(y_values, dtype=np.float64)[order]
    t_arr = np.asarray(actual_times, dtype=np.float64)[order]
    labels_arr = np.asarray(labels, dtype=object)[order]
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))
        created_fig = True
    else:
        fig = ax.figure
    ax.plot(x_arr, y_arr, marker="o", linewidth=2)
    ax.set_xlabel(r"Charge-Magnitude Ratio $\max(|q|) / \min(|q|)$")
    ax.set_ylabel(r"$\left(r_2/r_1\right)$")
    ax.set_title("Transformed Inner-Radius Ratio vs Charge-Value Ratio\n" f"(target t={float(time):.3g}, low p={low_percentile:g}, high-inner p={high_inner_percentile:g})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)
    return fig, ax, x_arr, y_arr, t_arr, labels_arr


def plot_radial_force_balance(sim, frame=None, last_n_frames=20, bins=60, use_rescaled_coords="auto", k=None, ax=None, show=True):
    meta = sim.get("meta", {}) or {}
    positions = np.asarray(sim["positions"], dtype=np.float64)
    times = np.asarray(sim["times"], dtype=np.float64)
    charges = np.asarray(sim.get("charges"), dtype=np.float64)
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(f"Expected positions shape (steps, n, 2), got {positions.shape}")
    if times.ndim != 1 or times.shape[0] != positions.shape[0]:
        raise ValueError("sim['times'] must be 1D with length equal to positions steps.")
    if charges.ndim != 1 or charges.shape[0] != positions.shape[1]:
        raise ValueError("sim['charges'] must be present and match the particle count.")

    unique = np.unique(charges)
    if unique.size != 2:
        raise ValueError(f"Expected exactly two populations, found charges={unique.tolist()}")
    q1, q2 = float(unique[0]), float(unique[1])
    mask1 = np.isclose(charges, q1)
    mask2 = np.isclose(charges, q2)
    k_value = float(meta.get("k") if k is None else k)
    if not np.isfinite(k_value):
        raise ValueError("k is missing or invalid; pass k explicitly.")
    if abs(k_value) > 1e-12:
        raise ValueError(f"This analysis targets k=0. Got k={k_value}.")
    v0 = float(meta.get("v0", 1.0))
    l = float(meta.get("l", 1.0))
    r_floor = float(meta.get("r_floor", 1e-12))
    use_rescaled = _resolve_use_rescaled(use_rescaled_coords, meta)
    gamma = 1.0 / (k_value + 2.0)
    frame_idx = _resolve_frame_indices(len(times), frame, last_n_frames)
    r1_all, f1_all, r2_all, f2_all = [], [], [], []
    used_frames = 0
    skipped_nonpositive_t = 0

    for idx in frame_idx:
        t = float(times[idx])
        x = positions[idx]
        if use_rescaled:
            y = x
        else:
            if t <= 0.0:
                skipped_nonpositive_t += 1
                continue
            y = x / (t ** gamma)
        f_vec = compute_velocity_overdamped(y, k_value, v0, l, r_floor, charges)
        r = np.linalg.norm(y, axis=1)
        unit = np.zeros_like(y)
        nz = r > 0.0
        unit[nz] = y[nz] / r[nz, None]
        f_rad = np.sum(unit * f_vec, axis=1)
        f_rad[~nz] = np.nan
        good = np.isfinite(f_rad) & np.isfinite(r)
        r1_all.append(r[mask1 & good])
        f1_all.append(f_rad[mask1 & good])
        r2_all.append(r[mask2 & good])
        f2_all.append(f_rad[mask2 & good])
        used_frames += 1

    if used_frames == 0:
        raise RuntimeError("No valid frames available for analysis (all skipped).")
    r1 = np.concatenate(r1_all) if r1_all else np.empty(0, dtype=np.float64)
    r2 = np.concatenate(r2_all) if r2_all else np.empty(0, dtype=np.float64)
    f1 = np.concatenate(f1_all) if f1_all else np.empty(0, dtype=np.float64)
    f2 = np.concatenate(f2_all) if f2_all else np.empty(0, dtype=np.float64)
    if r1.size == 0 or r2.size == 0:
        raise RuntimeError("Insufficient species-resolved data after filtering.")

    edges = np.linspace(0.0, float(max(np.max(r1), np.max(r2))), max(4, int(bins)) + 1, dtype=np.float64)
    centers, f1_mean, f1_sem, f1_count = _radial_bin_stats(r1, f1, edges)
    _, f2_mean, f2_sem, f2_count = _radial_bin_stats(r2, f2, edges)
    beta_r = 0.5 * centers
    f1_cross = _crossings_with_reference(centers, f1_mean, beta_r)
    f2_cross = _crossings_with_reference(centers, f2_mean, beta_r)

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(8.5, 5.4))
        created_fig = True
    else:
        fig = ax.figure
    ax.errorbar(centers, f1_mean, yerr=f1_sem, fmt="o-", ms=4, lw=1.5, capsize=2, label=f"Species 1 (q={q1:g})")
    ax.errorbar(centers, f2_mean, yerr=f2_sem, fmt="s-", ms=4, lw=1.5, capsize=2, label=f"Species 2 (q={q2:g})")
    ax.plot(centers, beta_r, "k--", lw=1.8, label=r"$\beta r$ (beta=1/2)")
    ax.set_xlabel(r"Radius $r = |y|$")
    ax.set_ylabel(r"Mean radial interaction force $\langle \hat{y}\cdot F \rangle$")
    ax.set_title("Radial Force-Balance Test in Self-Similar Coordinates (k=0)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    if show:
        plt.show()
    elif created_fig:
        plt.close(fig)

    data = {
        "centers": centers,
        "f1_mean": f1_mean,
        "f1_sem": f1_sem,
        "f1_count": f1_count,
        "f2_mean": f2_mean,
        "f2_sem": f2_sem,
        "f2_count": f2_count,
        "beta_r": beta_r,
        "r1": r1,
        "r2": r2,
        "f1": f1,
        "f2": f2,
    }
    summary = {
        "frames_used": used_frames,
        "n_requested_frames": len(frame_idx),
        "skipped_nonpositive_t": skipped_nonpositive_t,
        "use_rescaled": use_rescaled,
        "mean_r1": float(np.mean(r1)),
        "mean_r2": float(np.mean(r2)),
        "species_2_outer": float(np.mean(r2)) > float(np.mean(r1)),
        "f1_crossings": f1_cross,
        "f2_crossings": f2_cross,
        "q1": q1,
        "q2": q2,
    }
    return fig, ax, data, summary


__all__ = [
    "_crossings_with_reference",
    "_infer_rescaled_from_meta",
    "_radial_bin_stats",
    "_resolve_frame_indices",
    "_resolve_use_rescaled",
    "plot_inner_radius_ratio_power_vs_charge_value_ratio",
    "plot_radial_force_balance",
]
