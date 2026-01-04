import json
import numpy as np

__all__ = ["load_npz"]


def _pick(data, *names):
    for name in names:
        if name in data.files:
            return data[name]
    return None


def load_npz(path):
    """
    Load simulation data from .npz (expects new-format files with meta_json).
    Returns a dict with positions, times, energy, std, final_positions, meta, source_path.
    """
    data = np.load(path, allow_pickle=True)

    positions = _pick(data, "positions", "r_history", "r_hist")
    times = _pick(data, "times", "t", "t_hist")
    energy = _pick(data, "energy", "pe", "pe_history", "pe_hist")
    std = _pick(data, "std", "std_history", "std_hist")
    final_positions = _pick(data, "final_positions", "r_final")

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
        std=std,
        final_positions=final_positions,
        meta=meta,
        source_path=path,
    )
