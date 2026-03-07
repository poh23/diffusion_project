import json
from pathlib import Path

import numpy as np

__all__ = ["load_npz", "save_npz"]


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _pick(data, *names):
    for name in names:
        if name in data.files:
            return data[name]
    return None


def load_npz(path):
    """
    Load simulation data from .npz (expects new-format files with meta_json).
    Returns a dict with positions, times, energy, std, final_positions,
    optional energy components, density/radii if present, meta, source_path.
    """
    data = np.load(path, allow_pickle=True)

    positions = _pick(data, "positions", "r_history", "r_hist")
    times = _pick(data, "times", "t", "t_hist")
    energy = _pick(data, "energy", "pe", "pe_history", "pe_hist")
    std = _pick(data, "std", "std_history", "std_hist")
    final_positions = _pick(data, "final_positions", "r_final")
    energy_aa = _pick(data, "energy_aa")
    energy_ab = _pick(data, "energy_ab")
    energy_bb = _pick(data, "energy_bb")
    density = _pick(data, "density")
    radii = _pick(data, "radii")
    charges = _pick(data, "charges")

    if positions is None:
        raise KeyError(f"{path}: couldn't find positions array. Keys = {data.files}")

    if times is None:
        steps = positions.shape[0]
        times = np.arange(steps, dtype=float)

    if energy is None:
        energy = np.full(positions.shape[0], np.nan)

    if std is None:
        std = np.full(positions.shape[0], np.nan)

    if final_positions is None:
        final_positions = positions[-1]

    if "meta_json" not in data.files:
        raise KeyError(f"{path}: missing meta_json; legacy inference removed.")

    try:
        meta = json.loads(str(data["meta_json"].item()))
    except Exception as exc:
        raise ValueError(f"{path}: failed to parse meta_json") from exc

    return dict(
        positions=positions,
        times=times,
        energy=energy,
        energy_aa=energy_aa,
        energy_ab=energy_ab,
        energy_bb=energy_bb,
        std=std,
        final_positions=final_positions,
        charges=charges,
        density=density,
        radii=radii,
        meta=meta,
        source_path=path,
    )


def save_npz(out_path, sim_dict):
    meta_json = json.dumps(sim_dict["meta"], default=_json_default)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        "positions": sim_dict["positions"],
        "times": sim_dict["times"],
        "energy": sim_dict["energy"],
        "energy_aa": sim_dict.get("energy_aa"),
        "energy_ab": sim_dict.get("energy_ab"),
        "energy_bb": sim_dict.get("energy_bb"),
        "std": sim_dict["std"],
        "final_positions": sim_dict["final_positions"],
        "meta_json": np.array(meta_json, dtype=object),
    }

    arrays = {key: value for key, value in arrays.items() if value is not None}

    # optional
    if sim_dict.get("density") is not None:
        arrays["density"] = sim_dict["density"]
    if sim_dict.get("radii") is not None:
        arrays["radii"] = sim_dict["radii"]
    if sim_dict.get("charges") is not None:
        arrays["charges"] = sim_dict["charges"]

    np.savez_compressed(
        out_path,
        **arrays,
    )
    print(f"Saved: {out_path}")
    print(f"Meta: {sim_dict['meta']}")
