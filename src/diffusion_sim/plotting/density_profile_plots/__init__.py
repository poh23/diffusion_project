import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


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
    radii = np.asarray(sim["radii"]) if "radii" in sim else np.linalg.norm(np.asarray(sim["positions"]), axis=2)
    if density.shape != radii.shape:
        raise ValueError(f"density shape {density.shape} and radii shape {radii.shape} differ.")
    n = density.shape[1]

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))
        created_fig = True
    else:
        fig = ax.figure

    stride = sim.get("meta", {}).get("density_stride") if snap_to_stride else None
    if not stride or stride <= 1:
        stride = None

    charges = sim.get("charges")
    charge_groups = [("all", np.ones(n, dtype=bool))]
    if charges is not None:
        charges = np.asarray(charges)
        if charges.shape == (n,):
            unique_charges = np.unique(charges)
            if charge_value is not None:
                mask = np.isclose(charges, charge_value)
                if not np.any(mask):
                    raise ValueError(
                        f"Requested charge_value={charge_value} not found. Available charges: {unique_charges.tolist()}"
                    )
                charge_groups = [(f"q={float(charge_value):g}", mask)]
            elif split_by_charge and unique_charges.size > 0:
                charge_groups = [(f"q={float(q):g}", np.isclose(charges, q)) for q in unique_charges]
    elif charge_value is not None:
        raise ValueError("charge_value was set but sim['charges'] is missing.")

    for t in np.atleast_1d(times):
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
                print(f"Warning: insufficient finite density points at t~{sim_times[idx]:.3g}, {charge_label}; skipping.")
                continue

            r_sorted = np.sort(r[finite])
            d_sorted = d[finite][np.argsort(r[finite])]
            if window_frac and window_frac > 0.0:
                window = max(min_window, int(len(d_sorted) * window_frac))
                window = min(window, len(d_sorted))
                if window < 1:
                    window = 1
                if window % 2 == 0 and window > 1:
                    window -= 1
                d_plot = gaussian_filter1d(d_sorted, sigma=window / 6.0, mode="nearest")
            else:
                d_plot = d_sorted

            if scaled:
                scale = t_val ** gamma
                r_plot = r_sorted / scale
                d_plot = d_plot * (t_val ** (2.0 * gamma))
            else:
                r_plot = r_sorted

            label = f"t~{t_val:.3g}" if charge_label == "all" else f"{charge_label}, t~{t_val:.3g}"
            ax.plot(r_plot, d_plot, label=label)

    ax.set_xlabel(r"$r / t^{\frac{1}{k+2}}$" if scaled else "Radius")
    ax.set_ylabel(r"$\rho t^{\frac{2}{k+2}}$" if scaled else "Density (1 / Voronoi area)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()

    if show:
        plt.show()
    elif created_fig:
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
    if "density" not in sim:
        raise KeyError("Simulation dict must contain 'density' array.")

    meta = sim.get("meta", {}) or {}
    if k is None:
        k = meta.get("k", None)
    if k is None:
        raise ValueError("k is required (pass k=... or include it in sim['meta']).")

    density = np.asarray(sim["density"])
    sim_times = np.asarray(sim["times"])
    radii = np.asarray(sim["radii"]) if "radii" in sim else np.linalg.norm(np.asarray(sim["positions"]), axis=2)
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
    stride = meta.get("density_stride") if snap_to_stride else None
    if not stride or stride <= 1:
        stride = None

    charge_groups = [("all", np.ones(n, dtype=bool))]
    if split_by_charge:
        charges = sim.get("charges")
        if charges is not None:
            charges = np.asarray(charges)
            if charges.shape == (n,):
                unique_charges = np.unique(charges)
                if unique_charges.size > 0:
                    charge_groups = [(f"q={float(q):g}", np.isclose(charges, q)) for q in unique_charges]

    for t in np.atleast_1d(times):
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
                print(f"Warning: insufficient finite density points at t~{t_val:.3g}, {charge_label}; skipping.")
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
                d_plot = gaussian_filter1d(d_sorted, sigma=window / 6.0, mode="nearest")
            else:
                d_plot = d_sorted

            label = f"t~{t_val:.3g}" if charge_label == "all" else f"{charge_label}, t~{t_val:.3g}"
            ax1.plot(r_sorted, d_plot, label=label)

            x2 = ((r_group * r_group) / (t_val ** (2.0 * gamma)))[np.argsort((r_group * r_group) / (t_val ** (2.0 * gamma)))]
            y2 = (d_scaled ** 1.5)[np.argsort((r_group * r_group) / (t_val ** (2.0 * gamma)))]
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


__all__ = ["plot_density_vs_radius", "plot_scaled_density_vs_radius"]
