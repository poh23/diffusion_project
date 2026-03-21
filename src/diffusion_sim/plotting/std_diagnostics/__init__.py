import matplotlib.pyplot as plt
import numpy as np


def plot_std(sim, show_theory=True, k=None, v0=None, l=None):
    meta = sim.get("meta", {}) or {}
    k = k if k is not None else meta.get("k", None)
    v0 = v0 if v0 is not None else meta.get("v0", None)
    l = l if l is not None else meta.get("l", None)
    t = np.asarray(sim["times"])
    s = np.asarray(sim["std"])
    m = t > 0
    t = t[m]
    s = s[m]

    plt.figure(figsize=(8, 4))
    plt.plot(t, s, linewidth=2, label="Simulation")
    can_theory = (k is not None) and (v0 is not None) and (l is not None)
    if show_theory and can_theory:
        gamma = 1.0 / (k + 2.0)
        plt.plot(t, (50 * t) ** gamma, "--", linewidth=1, label="Theory (no diffusion)")
        if meta.get("diffusion", True):
            D = meta.get("diffusion_coeff", 0.0)
            plt.plot(t, (2.0 * D * t) ** 0.5, ":", linewidth=1, label="Diffusion theory")
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


def plot_msd_by_charge(sim, particle_indices_by_charge=None, origin_window=None, same_plot=False, ax_list=None, show=True):
    positions = np.asarray(sim["positions"])
    times = np.asarray(sim["times"])
    charges = sim.get("charges")
    if charges is None:
        raise KeyError("Simulation dict must contain 'charges' for charge-wise MSD.")
    charges = np.asarray(charges)

    if positions.ndim != 3 or positions.shape[2] != 2:
        raise ValueError(f"Expected positions shape (steps, n_particles, 2), got {positions.shape}.")
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
                raise ValueError(f"Index {idx} has charge {charges[idx]}, expected charge {q}.")
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
        n_origins = int(np.count_nonzero(times <= float(times[0]) + float(origin_window)))
    else:
        raise TypeError("origin_window must be None, int, or float.")

    if n_origins < 1:
        raise ValueError("origin_window selected zero origins; increase the window.")
    if n_origins > times.shape[0]:
        n_origins = times.shape[0]

    for i, q in enumerate(unique_charges):
        ax = axes[0] if same_plot else axes[i]
        idx = selected_indices[q]
        traj = positions[:, idx, :]
        if n_origins == 1:
            disp = traj - traj[0]
            msd = np.sum(disp * disp, axis=1)
            t_plot = times - times[0]
        else:
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


__all__ = ["plot_msd_by_charge", "plot_std"]
