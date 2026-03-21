from pathlib import Path

import numpy as np

from ...io.h5 import load_h5
from ...io.npz import load_npz
from ...postprocess.energy_components import compute_energy_component_series


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


def _compute_low_and_high_inner_radii(sim, time, low_percentile=95.0, high_inner_percentile=5.0):
    sim_times = np.asarray(sim["times"], dtype=np.float64)
    positions = np.asarray(sim["positions"], dtype=np.float64)
    charges = np.asarray(sim.get("charges"))
    if charges is None:
        raise ValueError("Simulation dict must contain 'charges'.")

    if not (0.0 <= low_percentile <= 100.0):
        raise ValueError("low_percentile must satisfy 0 <= low_percentile <= 100.")
    if not (0.0 <= high_inner_percentile <= 100.0):
        raise ValueError("high_inner_percentile must satisfy 0 <= high_inner_percentile <= 100.")

    idx = _density_frame_index(sim_times, time, stride=None)
    t_used = float(sim_times[idx])
    r = np.linalg.norm(positions[idx], axis=1)

    unique = np.unique(charges)
    if unique.size != 2:
        raise ValueError(
            f"Expected exactly two charge populations, found {unique.size}: "
            f"{unique.tolist()}"
        )
    q_low = float(unique[0])
    q_high = float(unique[1])
    low_mask = np.isclose(charges, q_low)
    high_mask = np.isclose(charges, q_high)
    r_low = r[low_mask]
    r_high = r[high_mask]
    if r_low.size == 0 or r_high.size == 0:
        raise ValueError("One of the charge populations is empty.")

    r1 = float(np.percentile(r_low, low_percentile))
    r2 = float(np.percentile(r_high, high_inner_percentile))
    return t_used, r1, r2


__all__ = [
    "_compute_low_and_high_inner_radii",
    "_density_frame_index",
    "_energy_components_from_sim",
    "_get_charge_value_ratio",
    "_get_high_charge_fraction",
    "_load_sim_auto",
    "_parse_k_directory_name",
    "_population_charge_pair_sums",
    "_radial_profile_from_density",
]
