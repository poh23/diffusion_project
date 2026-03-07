import numpy as np
from numba import njit, prange

__all__ = ["compute_energy_numba", "compute_energy_components", "compute_std_numba"]


@njit(fastmath=True)
def _compute_energy_components_numba_impl(r, k, v0, l, r_floor, charges, charge_a, charge_b, split_populations):
    n = len(r)
    e_aa = 0.0
    e_ab = 0.0
    e_bb = 0.0
    coupling = v0 * (l ** (k + 1))

    for i in range(n):
        qi = charges[i]
        qi_is_a = abs(qi - charge_a) <= 1.0e-12
        qi_is_b = abs(qi - charge_b) <= 1.0e-12
        for j in range(i + 1, n):
            qj = charges[j]
            dx = r[i, 0] - r[j, 0]
            dy = r[i, 1] - r[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq)
            dist_eff = dist if dist > r_floor else r_floor
            charge_factor = qi * qj
            if k == 0.0:
                pair_energy = -coupling * charge_factor * np.log(dist_eff)
            else:
                pair_energy = coupling * charge_factor / (k * (dist_eff ** k))

            if not split_populations:
                e_aa += pair_energy
                continue

            qj_is_a = abs(qj - charge_a) <= 1.0e-12
            qj_is_b = abs(qj - charge_b) <= 1.0e-12
            if qi_is_a and qj_is_a:
                e_aa += pair_energy
            elif qi_is_b and qj_is_b:
                e_bb += pair_energy
            else:
                e_ab += pair_energy

    return e_aa + e_ab + e_bb, e_aa, e_ab, e_bb


def compute_energy_components(r, k, v0, l, r_floor, charges, population_values=None):
    charges = np.asarray(charges, dtype=np.float64)
    if charges.ndim != 1:
        raise ValueError("charges must be a 1D array")

    split_populations = False
    if population_values is not None:
        values = np.asarray(population_values, dtype=np.float64)
        if values.size >= 2:
            charge_a = float(values[0])
            charge_b = float(values[1])
            split_populations = True
        elif values.size == 1:
            charge_a = float(values[0])
            charge_b = float(values[0])
        else:
            charge_a = 1.0
            charge_b = 1.0
    else:
        unique = np.unique(charges)
        if unique.size >= 2:
            charge_a = float(unique[0])
            charge_b = float(unique[1])
            split_populations = True
        elif unique.size == 1:
            charge_a = float(unique[0])
            charge_b = float(unique[0])
        else:
            charge_a = 1.0
            charge_b = 1.0

    return _compute_energy_components_numba_impl(
        r,
        k,
        v0,
        l,
        r_floor,
        charges,
        charge_a,
        charge_b,
        split_populations,
    )


@njit(fastmath=True, parallel=True)
def compute_energy_numba(r, k, v0, l, r_floor, charges):
    n = len(r)
    pe = 0.0
    coupling = v0 * (l ** (k + 1))

    for i in prange(n):
        local_pe = 0.0
        qi = charges[i]
        for j in range(i + 1, n):
            qj = charges[j]
            dx = r[i, 0] - r[j, 0]
            dy = r[i, 1] - r[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq)
            dist_eff = dist if dist > r_floor else r_floor
            charge_factor = qi * qj
            if k == 0.0:
                # limit k->0 of coupling/(k * r^k) is -coupling * ln(r) up to a constant
                local_pe += -coupling * charge_factor * np.log(dist_eff)
            else:
                local_pe += coupling * charge_factor / (k * (dist_eff ** k))
        pe += local_pe
    return pe


@njit(fastmath=True)
def compute_std_numba(r):
    n = len(r)
    sum_r = 0.0
    sum_sq = 0.0

    for i in range(n):
        radius = np.sqrt(r[i, 0] * r[i, 0] + r[i, 1] * r[i, 1])
        sum_r += radius
        sum_sq += radius * radius

    mean = sum_r / n
    var = (sum_sq / n) - mean * mean
    if var < 0.0:
        var = 0.0
    return np.sqrt(var)
