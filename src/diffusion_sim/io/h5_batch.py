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


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _pack_state(state):
    if state is None:
        return None
    return np.void(pickle.dumps(state))


def _unpack_state(blob):
    if blob is None:
        return None
    return pickle.loads(bytes(blob))


def _dataset_is_resizable(dataset) -> bool:
    return dataset.chunks is not None and dataset.maxshape is not None and dataset.maxshape[0] is None


def _make_dataset_resizable(h5, name, *, maxshape, chunks, dtype, copy_len=None):
    old = h5[name]
    if _dataset_is_resizable(old):
        return

    data_shape = old.shape
    attrs = dict(old.attrs.items())
    tmp_name = f"__tmp_resizable_{name}"
    if tmp_name in h5:
        del h5[tmp_name]

    new = h5.create_dataset(
        tmp_name,
        shape=data_shape,
        maxshape=maxshape,
        chunks=chunks,
        dtype=dtype,
    )
    for key, value in attrs.items():
        new.attrs[key] = value

    if copy_len is None:
        copy_len = max(1, int(chunks[0]))
    for start in range(0, data_shape[0], copy_len):
        end = min(start + copy_len, data_shape[0])
        new[start:end] = old[start:end]

    del h5[name]
    h5.move(tmp_name, name)


def _ensure_resumable_frame_datasets(h5, n_particles, *, chunk_len):
    specs = {
        "positions": ((None, n_particles, 2), (chunk_len, n_particles, 2), np.float64),
        "times": ((None,), (chunk_len,), np.float64),
        "energy": ((None,), (chunk_len,), np.float64),
        "energy_aa": ((None,), (chunk_len,), np.float64),
        "energy_ab": ((None,), (chunk_len,), np.float64),
        "energy_bb": ((None,), (chunk_len,), np.float64),
        "std": ((None,), (chunk_len,), np.float64),
    }
    existing_len = h5["times"].shape[0] if "times" in h5 else 0
    for name, (maxshape, chunks, dtype) in specs.items():
        if name not in h5:
            h5.create_dataset(
                name,
                shape=(existing_len, *maxshape[1:]),
                maxshape=maxshape,
                chunks=chunks,
                dtype=dtype,
            )
            if existing_len > 0:
                h5[name][...] = np.nan
            continue
        _make_dataset_resizable(h5, name, maxshape=maxshape, chunks=chunks, dtype=h5[name].dtype)


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
            "energy_aa",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_len,),
            dtype=np.float64,
        )
        h5.create_dataset(
            "energy_ab",
            shape=(0,),
            maxshape=(None,),
            chunks=(chunk_len,),
            dtype=np.float64,
        )
        h5.create_dataset(
            "energy_bb",
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
    else:
        chunk_len = max(1, int(chunk_len))
        if resume:
            _ensure_resumable_frame_datasets(h5, n_particles, chunk_len=chunk_len)
        else:
            existing_len = h5["times"].shape[0] if "times" in h5 else 0
            for name in ("energy_aa", "energy_ab", "energy_bb"):
                if name not in h5:
                    h5.create_dataset(
                        name,
                        shape=(existing_len,),
                        maxshape=(None,),
                        chunks=(chunk_len,),
                        dtype=np.float64,
                    )
                    if existing_len > 0:
                        h5[name][...] = np.nan

    return h5


def append_h5_batch(h5, positions, times, energy, std, energy_aa=None, energy_ab=None, energy_bb=None):
    n_new = len(times)
    if n_new == 0:
        return

    start = h5["positions"].shape[0]
    end = start + n_new

    h5["positions"].resize((end, h5["positions"].shape[1], 2))
    h5["times"].resize((end,))
    h5["energy"].resize((end,))
    h5["energy_aa"].resize((end,))
    h5["energy_ab"].resize((end,))
    h5["energy_bb"].resize((end,))
    h5["std"].resize((end,))

    h5["positions"][start:end] = positions
    h5["times"][start:end] = times
    h5["energy"][start:end] = energy
    if energy_aa is None:
        h5["energy_aa"][start:end] = np.nan
    else:
        h5["energy_aa"][start:end] = energy_aa
    if energy_ab is None:
        h5["energy_ab"][start:end] = np.nan
    else:
        h5["energy_ab"][start:end] = energy_ab
    if energy_bb is None:
        h5["energy_bb"][start:end] = np.nan
    else:
        h5["energy_bb"][start:end] = energy_bb
    h5["std"][start:end] = std


def write_h5_meta(h5, meta, *, config_hash=None):
    meta_json = json.dumps(meta, default=_json_default)
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
