import json
from pathlib import Path

import h5py
import numpy as np

__all__ = ["save_h5", "load_h5"]


def save_h5(out_path, sim_dict):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(out_path, "w") as h5:
        h5.create_dataset("positions", data=sim_dict["positions"], compression=None)
        h5.create_dataset("times", data=sim_dict["times"], compression=None)
        h5.create_dataset("energy", data=sim_dict["energy"], compression=None)
        h5.create_dataset("std", data=sim_dict["std"], compression=None)
        h5.create_dataset("final_positions", data=sim_dict["final_positions"], compression=None)

        if sim_dict.get("density") is not None:
            h5.create_dataset("density", data=sim_dict["density"], compression=None)
        if sim_dict.get("radii") is not None:
            h5.create_dataset("radii", data=sim_dict["radii"], compression=None)

        meta_json = json.dumps(sim_dict["meta"])
        h5.attrs["meta_json"] = meta_json

    print(f"Saved: {out_path}")
    print(f"Meta: {sim_dict['meta']}")


def load_h5(path):
    path = Path(path)
    with h5py.File(path, "r") as h5:
        positions = h5["positions"][()]
        times = h5["times"][()]
        energy = h5["energy"][()]
        std = h5["std"][()]
        final_positions = h5["final_positions"][()]
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
        std=std,
        final_positions=final_positions,
        density=density,
        radii=radii,
        meta=meta,
        source_path=str(path),
    )
