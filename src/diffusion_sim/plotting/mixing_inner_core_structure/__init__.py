import matplotlib.pyplot as plt
import numpy as np


def compute_mixing_metric(sim, quantile=0.95, center="origin"):
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must satisfy 0 < quantile <= 1")
    if center not in {"origin", "com"}:
        raise ValueError("center must be 'origin' or 'com'")

    positions = np.asarray(sim["positions"])
    times = np.asarray(sim["times"])
    charges = np.asarray(sim.get("charges"))

    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(f"Expected positions shape (steps, n_particles, 2), got {positions.shape}.")
    if times.ndim != 1 or times.shape[0] != positions.shape[0]:
        raise ValueError("sim['times'] must be 1D with length equal to positions steps.")
    if charges.ndim != 1 or charges.shape[0] != positions.shape[1]:
        raise ValueError("sim['charges'] must have length equal to n_particles.")

    unique = np.unique(charges)
    if unique.size != 2:
        raise ValueError(
            f"Expected exactly two charge populations, found {unique.size}: {unique.tolist()}"
        )
    q_low, q_high = float(unique[0]), float(unique[1])
    low_mask = np.isclose(charges, q_low)
    high_mask = np.isclose(charges, q_high)

    if center == "com":
        com = np.mean(positions, axis=1, keepdims=True)
        radii = np.linalg.norm(positions - com, axis=2)
    else:
        radii = np.linalg.norm(positions, axis=2)

    low_r = radii[:, low_mask]
    high_r = radii[:, high_mask]
    r_low = np.quantile(low_r, quantile, axis=1)
    n_mix = np.sum(high_r <= r_low[:, None], axis=1).astype(np.int64)
    p_mix = n_mix / float(high_r.shape[1])

    return times, n_mix, p_mix, r_low


def plot_mixing_metric(
    sim,
    quantile=0.95,
    as_percentage=True,
    center="origin",
    t_start=None,
    t_end=None,
):
    times, n_mix, p_mix, _ = compute_mixing_metric(sim, quantile=quantile, center=center)
    y = 100.0 * p_mix if as_percentage else n_mix
    ylabel = "High-Charge Inside Low-Disk (%)" if as_percentage else "High-Charge Inside Low-Disk (count)"

    if t_start is not None or t_end is not None:
        t_min = float(np.min(times))
        t_max = float(np.max(times))
        lo = t_min if t_start is None else float(np.clip(float(t_start), t_min, t_max))
        hi = t_max if t_end is None else float(np.clip(float(t_end), t_min, t_max))
        if lo > hi:
            lo, hi = hi, lo
        mask = (times >= lo) & (times <= hi)
        times = times[mask]
        y = y[mask]

    plt.figure(figsize=(8, 4))
    plt.plot(times, y, linewidth=2)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_inner_particles_selfsimilar_radius(sim, t_ref, quantile=0.95, k=None, center="origin"):
    if not (0.0 < quantile <= 1.0):
        raise ValueError("quantile must satisfy 0 < quantile <= 1")
    if center not in {"origin", "com"}:
        raise ValueError("center must be 'origin' or 'com'")

    positions = np.asarray(sim["positions"])
    times = np.asarray(sim["times"])
    charges = np.asarray(sim.get("charges"))
    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(f"Expected positions shape (steps, n_particles, 2), got {positions.shape}.")
    if times.ndim != 1 or times.shape[0] != positions.shape[0]:
        raise ValueError("sim['times'] must be 1D with length equal to positions steps.")
    if charges.ndim != 1 or charges.shape[0] != positions.shape[1]:
        raise ValueError("sim['charges'] must have length equal to n_particles.")

    meta = sim.get("meta", {}) or {}
    k_val = k if k is not None else meta.get("k", None)
    if k_val is None:
        raise ValueError("k is required (pass k=... or include sim['meta']['k']).")
    if float(k_val) == -2.0:
        raise ValueError("k=-2 is invalid for self-similar scaling.")
    gamma = 1.0 / (float(k_val) + 2.0)

    unique = np.unique(charges)
    if unique.size != 2:
        raise ValueError(
            f"Expected exactly two charge populations, found {unique.size}: {unique.tolist()}"
        )
    low_mask = np.isclose(charges, float(unique[0]))
    high_mask = np.isclose(charges, float(unique[1]))

    ref_idx = int(np.abs(times - float(t_ref)).argmin())
    t_ref_actual = float(times[ref_idx])
    if t_ref_actual <= 0.0:
        raise ValueError("Reference time must be positive for self-similar scaling.")

    if center == "com":
        com = np.mean(positions, axis=1, keepdims=True)
        radii = np.linalg.norm(positions - com, axis=2)
    else:
        radii = np.linalg.norm(positions, axis=2)

    r_ref = radii[ref_idx]
    r_cut = float(np.quantile(r_ref[low_mask], quantile))
    inner_idx = np.flatnonzero((r_ref <= r_cut) & high_mask)
    if inner_idx.size == 0:
        raise ValueError("No inner high-charge particles found at the selected reference time.")

    valid_t = (times >= t_ref_actual) & (times > 0.0)
    t_plot = times[valid_t]
    if t_plot.size == 0:
        raise ValueError("No valid times to plot after t_ref.")
    r_ss = radii[valid_t][:, inner_idx] / (t_plot ** gamma)[:, None]
    r_ref_ss = r_cut / (t_ref_actual ** gamma)

    plt.figure(figsize=(9, 5))
    for j in range(r_ss.shape[1]):
        plt.plot(t_plot, r_ss[:, j], linewidth=1.0, alpha=0.7, color="#1f1fb4")
    plt.axhline(
        r_ref_ss,
        color="#d62728",
        linestyle="--",
        linewidth=2,
        label=f"R_ss at t_ref={t_ref_actual:.5g}",
    )
    plt.xlabel("Time")
    plt.ylabel("Self-Similar Radius  r / t^(1/(k+2))")
    plt.title(
        f"Inner Particles in Self-Similar Coordinates (q={quantile:.2f}, n_inner={inner_idx.size})"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


__all__ = [
    "compute_mixing_metric",
    "plot_inner_particles_selfsimilar_radius",
    "plot_mixing_metric",
]
