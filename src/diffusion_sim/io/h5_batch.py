import json
import pickle
from pathlib import Path

import h5py
import numpy as np

__all__ = [
    "open_h5_batch",
    "append_h5_batch",
    "write_h5_meta",
    "write_h5_resume_state",
    "load_h5_resume_state",
]


def _pack_state(state):
    if state is None:
        return None
    return np.void(pickle.dumps(state))


def _unpack_state(blob):
    if blob is None:
        return None
    return pickle.loads(bytes(blob))


def open_h5_batch(path, n_particles, *, resume=False, chunk_len=1024):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    h5 = h5py.File(path, mode)

    if "positions" not in h5:
        chunk_len = max(1, int(chunk_len))
        h5.create_dataset(
            "positions",
            shape=(0, n_particles, 2),
            maxshape=(None, n_particles, 2),
            chunks=(chunk_len, n_particles, 2),
            dtype=np.float64,
        )
        h5.create_dataset(
            "times",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_len,),
            dtype=np.float64,
        )
        h5.create_dataset(
            "energy",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_len,),
            dtype=np.float64,
        )
        h5.create_dataset(
            "std",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_len,),
            dtype=np.float64,
        )
        h5.create_dataset(
            "final_positions",
            shape=(n_particles, 2),
            maxshape=(n_particles, 2),
            dtype=np.float64,
        )

    return h5


def append_h5_batch(h5, positions, times, energy, std):
    n_new = len(times)
    if n_new == 0:
        return

    start = h5["positions"].shape[0]
    end = start + n_new

    h5["positions"].resize((end, h5["positions"].shape[1], 2))
    h5["times"].resize((end,))
    h5["energy"].resize((end,))
    h5["std"].resize((end,))

    h5["positions"][start:end] = positions
    h5["times"][start:end] = times
    h5["energy"][start:end] = energy
    h5["std"][start:end] = std


def write_h5_meta(h5, meta, *, config_hash=None):
    meta_json = json.dumps(meta)
    h5.attrs["meta_json"] = meta_json
    if config_hash is not None:
        h5.attrs["config_hash"] = config_hash


def write_h5_resume_state(h5, *, t_current, y_current, rng_state=None, stats=None, completed=False):
    grp = h5.require_group("resume")
    grp.attrs["t_current"] = float(t_current)
    grp.attrs["completed"] = bool(completed)

    if stats:
        if "n_steps" in stats:
            grp.attrs["n_steps"] = int(stats["n_steps"])
        if "nfev" in stats:
            grp.attrs["nfev"] = int(stats["nfev"])

    if "y_current" not in grp:
        grp.create_dataset("y_current", data=y_current, dtype=np.float64)
    else:
        grp["y_current"][...] = y_current

    packed = _pack_state(rng_state)
    if packed is not None:
        grp.attrs["rng_state"] = packed


def load_h5_resume_state(path):
    path = Path(path)
    with h5py.File(path, "r") as h5:
        resume = h5.get("resume")
        if resume is not None and "y_current" in resume and "t_current" in resume.attrs:
            y_current = resume["y_current"][()]
            t_current = float(resume.attrs["t_current"])
            rng_state = _unpack_state(resume.attrs.get("rng_state"))
            completed = bool(resume.attrs.get("completed", False))
            n_steps = resume.attrs.get("n_steps")
            nfev = resume.attrs.get("nfev")
            return dict(
                t_current=t_current,
                y_current=y_current,
                rng_state=rng_state,
                completed=completed,
                n_steps=n_steps,
                nfev=nfev,
            )

        if "positions" not in h5 or "times" not in h5:
            raise KeyError(f"{path}: missing positions/times for resume.")
        if h5["times"].shape[0] == 0:
            raise ValueError(f"{path}: empty times dataset; cannot resume.")
        t_current = float(h5["times"][-1])
        y_current = h5["positions"][-1]
        return dict(
            t_current=t_current,
            y_current=y_current,
            rng_state=None,
            completed=False,
            n_steps=None,
            nfev=None,
        )
