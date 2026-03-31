from __future__ import annotations

import numpy as np

__all__ = ["compute_external_energy", "compute_external_velocity"]


def _normalized_center(params: dict | None) -> np.ndarray:
    if params is None:
        params = {}
    center = params.get("center", (0.0, 0.0))
    center_arr = np.asarray(center, dtype=np.float64)
    if center_arr.shape != (2,):
        raise ValueError("external potential center must be length 2")
    return center_arr


def _harmonic_velocity(r: np.ndarray, params: dict | None) -> np.ndarray:
    params = {} if params is None else params
    omega = float(params["omega"])
    center = _normalized_center(params)
    return -omega * (r - center)


def _harmonic_energy(r: np.ndarray, params: dict | None) -> float:
    params = {} if params is None else params
    omega = float(params["omega"])
    center = _normalized_center(params)
    disp = r - center
    return float(0.5 * omega * np.sum(disp * disp))


def compute_external_velocity(
    r: np.ndarray,
    potential_name: str | None,
    params: dict | None,
) -> np.ndarray:
    if potential_name is None:
        return np.zeros_like(r)
    if potential_name == "harmonic":
        return _harmonic_velocity(np.asarray(r, dtype=np.float64), params)
    raise ValueError(f"Unsupported external potential: {potential_name}")


def compute_external_energy(
    r: np.ndarray,
    potential_name: str | None,
    params: dict | None,
) -> float:
    if potential_name is None:
        return 0.0
    if potential_name == "harmonic":
        return _harmonic_energy(np.asarray(r, dtype=np.float64), params)
    raise ValueError(f"Unsupported external potential: {potential_name}")
