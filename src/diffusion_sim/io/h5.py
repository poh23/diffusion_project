import json
from pathlib import Path

import h5py
import numpy as np

__all__ = ["save_h5", "load_h5"]


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_h5(out_path, sim_dict):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("positions", data=sim_dict["positions"], compression=None)
        h5.create_dataset("times", data=sim_dict["times"], compression=None)
        h5.create_dataset("energy", data=sim_dict["energy"], compression=None)
        if sim_dict.get("energy_aa") is not None:
            h5.create_dataset("energy_aa", data=sim_dict["energy_aa"], compression=None)
        if sim_dict.get("energy_ab") is not None:
            h5.create_dataset("energy_ab", data=sim_dict["energy_ab"], compression=None)
        if sim_dict.get("energy_bb") is not None:
            h5.create_dataset("energy_bb", data=sim_dict["energy_bb"], compression=None)
        h5.create_dataset("std", data=sim_dict["std"], compression=None)
        h5.create_dataset("final_positions", data=sim_dict["final_positions"], compression=None)
        if sim_dict.get("charges") is not None:
            h5.create_dataset("charges", data=sim_dict["charges"], compression=None)

        if sim_dict.get("density") is not None:
            h5.create_dataset("density", data=sim_dict["density"], compression=None)
        if sim_dict.get("radii") is not None:
            h5.create_dataset("radii", data=sim_dict["radii"], compression=None)

        meta_json = json.dumps(sim_dict["meta"], default=_json_default)
        h5.attrs["meta_json"] = meta_json

    print(f"Saved: {out_path}")
    print(f"Meta: {sim_dict['meta']}")


def load_h5(path):
    path = Path(path)
    with h5py.File(path, "r") as h5:
        positions = h5["positions"][()]
        times = h5["times"][()]
        energy = h5["energy"][()]
        energy_aa = h5["energy_aa"][()] if "energy_aa" in h5 else None
        energy_ab = h5["energy_ab"][()] if "energy_ab" in h5 else None
        energy_bb = h5["energy_bb"][()] if "energy_bb" in h5 else None
        std = h5["std"][()]
        final_positions = h5["final_positions"][()]
        charges = h5["charges"][()] if "charges" in h5 else None
        density = h5["density"][()] if "density" in h5 else None
        radii = h5["radii"][()] if "radii" in h5 else None

        meta_json = h5.attrs.get("meta_json")
        if meta_json is None:
            raise KeyError(f"{path}: missing meta_json attribute.")
        meta = json.loads(meta_json)

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
        source_path=str(path),
    )
