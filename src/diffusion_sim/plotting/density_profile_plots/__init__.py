import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _smooth_for_fit(y, *, window_frac=0.05, min_window=5):
    y = np.asarray(y, dtype=np.float64)
    if y.size < 3 or not window_frac or window_frac <= 0.0:
        return y
    window = max(min_window, int(y.size * window_frac))
    window = min(window, y.size)
    if window < 3:
        return y
    if window % 2 == 0:
        window -= 1
    return gaussian_filter1d(y, sigma=window / 6.0, mode="nearest")


def _fit_prefix_before_skyrocket(x, y, *, min_points=10, smooth_window_frac=0.05, min_window=5):
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, dtype=np.float64)[finite]
    y = np.asarray(y, dtype=np.float64)[finite]
    if x.size < min_points:
        return None

    y_smooth = _smooth_for_fit(y, window_frac=smooth_window_frac, min_window=min_window)
    cutoff = int(np.argmin(y_smooth)) + 1
    if cutoff < min_points:
        cutoff = x.size

    x_fit = x[:cutoff]
    y_fit = y[:cutoff]
    if x_fit.size < min_points:
        return None
    slope, intercept = np.polyfit(x_fit, y_fit, 1)
    return slope, intercept, x_fit, y_fit


def _density_inverse_charge_time_scale(charges, t_val, k):
    charges = np.asarray(charges, dtype=np.float64)
    unique = np.unique(charges)
    if unique.size not in {1, 2}:
        raise ValueError(f"Expected one or two charge populations, found {unique.size}: {unique.tolist()}")

    q1 = float(unique[0])
    n_total = int(charges.size)
    n1 = int(np.count_nonzero(np.isclose(charges, q1)))
    if unique.size == 1:
        q2 = 0.0
        n2 = 0
    else:
        q2 = float(unique[1])
        n2 = int(np.count_nonzero(np.isclose(charges, q2)))
    if n_total <= 0 or n1 <= 0 or n2 < 0:
        raise ValueError("Charge populations must be non-empty.")

    q_mean = (n1 / n_total) * q1 + (n2 / n_total) * q2
    scale_base = n_total * t_val * (q_mean ** 2)
    if scale_base <= 0.0:
        raise ValueError("Density scaling factor must be positive.")
    return (1.0 / n1) * (scale_base ** (2.0 / (k + 2.0)))


def _radius_charge_time_scale(charges, t_val, k):
    charges = np.asarray(charges, dtype=np.float64)
    unique = np.unique(charges)
    if unique.size not in {1, 2}:
        raise ValueError(f"Expected one or two charge populations, found {unique.size}: {unique.tolist()}")

    q1 = float(unique[0])
    n_total = int(charges.size)
    n1 = int(np.count_nonzero(np.isclose(charges, q1)))
    if unique.size == 1:
        q2 = 0.0
        n2 = 0
    else:
        q2 = float(unique[1])
        n2 = int(np.count_nonzero(np.isclose(charges, q2)))
    if n_total <= 0 or n1 <= 0 or n2 < 0:
        raise ValueError("Charge populations must be non-empty.")

    q_mean = (n1 / n_total) * q1 + (n2 / n_total) * q2
    scale_base = n_total * (q_mean ** 2) * t_val
    if scale_base <= 0.0:
        raise ValueError("Radius scaling factor must be positive.")
    return scale_base ** (1.0 / (k + 2.0))


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
    if "density" not in sim or sim.get("density") is None:
        raise KeyError("Simulation dict must contain 'density' array.")

    meta = sim.get("meta", {}) or {}
    if scaled:
        if k is None:
            k = meta.get("k", None)
        if k is None:
            raise ValueError("k is required when scaled=True (pass k=... or include it in sim['meta']).")
        if np.isclose(float(k), -2.0):
            raise ValueError("k=-2 is invalid (division by zero in exponent).")
        gamma = 1.0 / (k + 2.0)

    density = np.asarray(sim["density"])
    sim_times = np.asarray(sim["times"])
    radii_value = sim.get("radii")
    radii = np.asarray(radii_value) if radii_value is not None else np.linalg.norm(np.asarray(sim["positions"]), axis=2)
    if density.ndim != 2:
        raise ValueError(f"density must be a 2D array with shape (frames, particles), got shape {density.shape}.")
    if radii.ndim != 2:
        raise ValueError(f"radii must be a 2D array with shape (frames, particles), got shape {radii.shape}.")
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


def plot_inverse_density_squared_vs_radius_squared(
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
    fit_linear=False,
    fit_time=None,
    fit_charge_value=None,
    fit_min_points=10,
    fit_smooth_window_frac=0.05,
):
    if "density" not in sim or sim.get("density") is None:
        raise KeyError("Simulation dict must contain 'density' array.")

    meta = sim.get("meta", {}) or {}
    if scaled:
        if k is None:
            k = meta.get("k", None)
        if k is None:
            raise ValueError("k is required when scaled=True (pass k=... or include it in sim['meta']).")
        if np.isclose(float(k), -2.0):
            raise ValueError("k=-2 is invalid (division by zero in exponent).")
        gamma = 1.0 / (k + 2.0)

    density = np.asarray(sim["density"])
    sim_times = np.asarray(sim["times"])
    radii_value = sim.get("radii")
    radii = np.asarray(radii_value) if radii_value is not None else np.linalg.norm(np.asarray(sim["positions"]), axis=2)
    if density.ndim != 2:
        raise ValueError(f"density must be a 2D array with shape (frames, particles), got shape {density.shape}.")
    if radii.ndim != 2:
        raise ValueError(f"radii must be a 2D array with shape (frames, particles), got shape {radii.shape}.")
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

    fit_idx = None
    if fit_time is not None:
        fit_idx = int(np.abs(sim_times - fit_time).argmin())
        if stride is not None:
            fit_idx = int(round(fit_idx / stride) * stride)
            fit_idx = max(0, min(fit_idx, len(sim_times) - 1))

    charges = sim.get("charges")
    if scaled and charges is None:
        raise ValueError("sim['charges'] is required when scaled=True.")
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

    fit_done = False
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
            else:
                finite &= d != 0.0
            if finite.sum() < min_points:
                print(f"Warning: insufficient finite density points at t~{sim_times[idx]:.3g}, {charge_label}; skipping.")
                continue

            order = np.argsort(r[finite])
            r_sorted = r[finite][order]
            d_sorted = d[finite][order]
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
                radius_scale = _radius_charge_time_scale(charges, t_val, k)
                x_plot = (r_sorted / radius_scale) ** 2
                d_scaled = d_plot * _density_inverse_charge_time_scale(charges, t_val, k)
                y_plot = d_scaled ** -2
            else:
                x_plot = r_sorted ** 2
                y_plot = d_plot ** -2

            finite_plot = np.isfinite(x_plot) & np.isfinite(y_plot)
            if np.count_nonzero(finite_plot) < min_points:
                print(f"Warning: insufficient finite transformed points at t~{sim_times[idx]:.3g}, {charge_label}; skipping.")
                continue

            label = f"t~{t_val:.3g}" if charge_label == "all" else f"{charge_label}, t~{t_val:.3g}"
            ax.plot(x_plot[finite_plot], y_plot[finite_plot], label=label)

            do_fit = fit_linear and not fit_done
            if do_fit and fit_idx is not None:
                do_fit = idx == fit_idx
            if do_fit and fit_charge_value is not None:
                do_fit = charge_label == f"q={float(fit_charge_value):g}"
            if do_fit:
                fit = _fit_prefix_before_skyrocket(
                    x_plot[finite_plot],
                    y_plot[finite_plot],
                    min_points=fit_min_points,
                    smooth_window_frac=fit_smooth_window_frac,
                    min_window=min_window,
                )
                if fit is None:
                    print(f"Warning: insufficient points for linear fit at t~{t_val:.3g}, {charge_label}; skipping fit.")
                else:
                    slope, intercept, x_fit, _ = fit
                    x_line = np.array([float(np.min(x_fit)), float(np.max(x_fit))], dtype=np.float64)
                    y_line = slope * x_line + intercept
                    fit_label = (
                        f"fit {label}: y={slope:.3g}x+{intercept:.3g}"
                        if charge_label != "all"
                        else f"fit t~{t_val:.3g}: y={slope:.3g}x+{intercept:.3g}"
                    )
                    ax.plot(x_line, y_line, "--", color="black", linewidth=2.0, label=fit_label)
                fit_done = True

    ax.set_xlabel(
        r"$\left(r / \left[Nt\left(\frac{N_1}{N}q_1+\frac{N_2}{N}q_2\right)^2\right]^{\frac{1}{k+2}}\right)^2$"
        if scaled
        else r"$r^2$"
    )
    ax.set_ylabel(
        r"$\left(\rho\frac{1}{N_1}\left[Nt\left(\frac{N_1}{N}q_1+\frac{N_2}{N}q_2\right)^2\right]^{\frac{2}{k+2}}\right)^{-2}$"
        if scaled
        else r"$\rho^{-2}$"
    )
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


__all__ = [
    "plot_density_vs_radius",
    "plot_inverse_density_squared_vs_radius_squared",
    "plot_scaled_density_vs_radius",
]
