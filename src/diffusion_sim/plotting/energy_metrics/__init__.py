import matplotlib.pyplot as plt
import numpy as np

from ...postprocess.energy_components import compute_population_energy_series
from ..shared_helpers import _energy_components_from_sim, _population_charge_pair_sums


def plot_energy(sim, scale_time=False, show=True):
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
                e = e * t_raw ** (k * gamma)
        else:
            print("Warning: k not found in sim['meta'], cannot scale time.")
            t = t_raw
    else:
        t = t_raw
        k = None

    plt.figure(figsize=(8, 4))
    plt.plot(t, e, linewidth=2)
    print(f"Final energy: {e[-1]:.4g} at final time {t_raw[-1]:.4g}")
    print(f"Initial energy: {e[1]:.4g} at initial time {t_raw[1]:.4g}")
    print(f"Middle energy : {e[len(e)//2]:.4g} at middle time {t_raw[len(t_raw)//2]:.4g}")
    print(f"Energy change: {e[-1] - e[1]:.4g} over time {t_raw[-1] - t_raw[1]:.4g}")
    if scale_time and k is not None:
        plt.xlabel(f"Time (scaled by t^(1/(k+2)) with k={k})")
        plt.ylabel(f"Potential Energy (scaled with k={k})")
    else:
        plt.xlabel("Time")
        plt.ylabel("Potential Energy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if show:
        plt.show()


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
    axes[1].set_xlabel(
        f"Time (scaled by t^(1/(k+2)) with k={meta['k']})"
        if scale_time and meta.get("k", None) is not None else "Time"
    )
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


__all__ = ["plot_energy", "plot_energy_by_charge"]
