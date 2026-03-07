from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from ..io.h5 import load_h5, save_h5
from ..io.npz import load_npz, save_npz
from ..metrics import compute_energy_components


def compute_energy_component_series(
    r_hist: np.ndarray,
    *,
    k,
    v0,
    l,
    r_floor,
    charges,
    population_values=None,
    print_every: int | None = None,
    show_progress: bool = False,
):
    """
    Compute per-frame pair-energy components for one- or two-population runs.

    Returns:
      energy:    total pair energy
      energy_aa: within-population A pair energy
      energy_ab: cross-population pair energy
      energy_bb: within-population B pair energy
    """
    steps = r_hist.shape[0]
    energy = np.empty(steps, dtype=np.float64)
    energy_aa = np.empty(steps, dtype=np.float64)
    energy_ab = np.empty(steps, dtype=np.float64)
    energy_bb = np.empty(steps, dtype=np.float64)

    step_iter = range(steps)
    if show_progress:
        step_iter = tqdm(step_iter, total=steps, desc="energy", unit="step")

    for i in step_iter:
        energy[i], energy_aa[i], energy_ab[i], energy_bb[i] = compute_energy_components(
            r_hist[i],
            k,
            v0,
            l,
            r_floor,
            charges,
            population_values=population_values,
        )
        if print_every and (i + 1) % print_every == 0:
            print(f"[energy] processed step {i + 1}/{steps}")

    return energy, energy_aa, energy_ab, energy_bb


def compute_population_energy_series(energy_aa, energy_ab, energy_bb):
    energy_a = np.asarray(energy_aa, dtype=np.float64) + 0.5 * np.asarray(energy_ab, dtype=np.float64)
    energy_b = np.asarray(energy_bb, dtype=np.float64) + 0.5 * np.asarray(energy_ab, dtype=np.float64)
    return energy_a, energy_b


def augment_sim_with_energy_components(
    sim: dict,
    print_every: int | None = None,
    show_progress: bool = False,
) -> dict:
    meta = sim.get("meta", {}) or {}
    missing = [name for name in ("k", "v0", "l", "r_floor") if meta.get(name) is None]
    if missing:
        raise KeyError(f"Simulation metadata is missing required fields: {', '.join(missing)}")

    charges = sim.get("charges")
    if charges is None:
        raise KeyError("Simulation dict must contain 'charges' to compute energy components.")

    energy, energy_aa, energy_ab, energy_bb = compute_energy_component_series(
        np.asarray(sim["positions"]),
        k=meta["k"],
        v0=meta["v0"],
        l=meta["l"],
        r_floor=meta["r_floor"],
        charges=np.asarray(charges),
        population_values=meta.get("charge_values"),
        print_every=print_every,
        show_progress=show_progress,
    )

    sim_out = dict(sim)
    sim_out["energy"] = energy
    sim_out["energy_aa"] = energy_aa
    sim_out["energy_ab"] = energy_ab
    sim_out["energy_bb"] = energy_bb
    sim_out["meta"] = dict(meta)
    sim_out["meta"]["energy_components"] = "AA_AB_BB"
    return sim_out


def _load_sim(path: Path):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_h5(path)
    return load_npz(path)


def _save_sim(path: Path, sim: dict):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        save_h5(path, sim)
    else:
        save_npz(path, sim)


def process_energy_file(
    path: Path,
    overwrite: bool = False,
    suffix: str = "_with_energy_components",
    print_every: int | None = None,
    show_progress: bool = False,
):
    sim = _load_sim(path)
    sim_out = augment_sim_with_energy_components(
        sim,
        print_every=print_every,
        show_progress=show_progress,
    )
    out_path = path if overwrite else path.with_name(f"{path.stem}{suffix}{path.suffix}")
    _save_sim(out_path, sim_out)


def process_energy_files(
    files,
    overwrite: bool = False,
    suffix: str = "_with_energy_components",
    print_every: int | None = None,
    show_progress: bool = False,
):
    for name in files:
        process_energy_file(
            Path(name),
            overwrite=overwrite,
            suffix=suffix,
            print_every=print_every,
            show_progress=show_progress,
        )


def load_sim_for_energy(path: Path):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_h5(path)
    return load_npz(path)


def save_sim_with_energy(path: Path, sim: dict):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        save_h5(path, sim)
    else:
        save_npz(path, sim)
