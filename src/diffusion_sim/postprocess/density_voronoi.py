import json
from pathlib import Path

import numpy as np
from scipy.spatial import Voronoi

from ..io.npz import load_npz


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def voronoi_density(points: np.ndarray) -> np.ndarray:
    """
    Compute density per point as 1 / area of its Voronoi cell.
    Points with unbounded cells get density=0.
    """
    vor = Voronoi(points)
    densities = np.zeros(len(points), dtype=np.float64)

    for idx, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or (-1 in region):
            densities[idx] = 0.0
            continue

        poly = vor.vertices[region]
        area = _polygon_area(poly)
        densities[idx] = 1.0 / area if area > 0 else 0.0

    return densities


def voronoi_density_by_charge(points: np.ndarray, charges: np.ndarray) -> np.ndarray:
    """
    Compute density per point with independent Voronoi tessellation per charge group.
    Points in groups that cannot form a valid Voronoi diagram get density=0.
    """
    n = points.shape[0]
    out = np.zeros(n, dtype=np.float64)
    unique_charges = np.unique(charges)

    for q in unique_charges:
        mask = np.isclose(charges, q)
        idx = np.flatnonzero(mask)
        if idx.size < 4:
            continue
        try:
            out[idx] = voronoi_density(points[idx])
        except Exception:
            # Degenerate groups (e.g. collinear points) are treated as zero density.
            out[idx] = 0.0
    return out


def compute_density_series(
    r_hist: np.ndarray,
    charges: np.ndarray | None = None,
    print_every: int | None = None,
    split_by_charge: bool = True,
) -> np.ndarray:
    """
    r_hist: (steps, n, 2)
    Returns density per particle per step: (steps, n), density = 1 / Voronoi cell area.
    """
    steps, n, _ = r_hist.shape
    out = np.zeros((steps, n), dtype=np.float64)
    valid_charges = (
        split_by_charge
        and charges is not None
        and np.asarray(charges).shape == (n,)
    )
    charge_arr = np.asarray(charges) if valid_charges else None

    for i in range(steps):
        if charge_arr is None:
            out[i] = voronoi_density(r_hist[i])
        else:
            out[i] = voronoi_density_by_charge(r_hist[i], charge_arr)
        if print_every and (i + 1) % print_every == 0:
            print(f"[density] processed step {i + 1}/{steps}")
    return out


def compute_density_and_radius_series(
    r_hist: np.ndarray,
    charges: np.ndarray | None = None,
    print_every: int | None = None,
    split_by_charge: bool = True,
):
    """
    r_hist: (steps, n, 2)
    Returns:
      densities: (steps, n)  # 1 / Voronoi cell area (0 for unbounded cells)
      radii:     (steps, n)  # sqrt(x^2 + y^2) for each particle
    """
    densities = compute_density_series(
        r_hist,
        charges=charges,
        print_every=print_every,
        split_by_charge=split_by_charge,
    )
    radii = np.linalg.norm(r_hist, axis=2)
    return densities, radii


def save_with_density(
    sim: dict,
    density: np.ndarray,
    radii: np.ndarray | None,
    out_path: Path,
    density_stride: int | None = None,
):
    meta = dict(sim["meta"])
    meta["density_method"] = "voronoi_2d"
    if sim.get("charges") is not None:
        meta["density_split_by_charge"] = True
    if density_stride is not None:
        meta["density_stride"] = density_stride
    meta_json = json.dumps(meta)

    arrays = {
        "positions": sim["positions"],
        "times": sim["times"],
        "energy": sim["energy"],
        "energy_aa": sim.get("energy_aa"),
        "energy_ab": sim.get("energy_ab"),
        "energy_bb": sim.get("energy_bb"),
        "std": sim["std"],
        "final_positions": sim["final_positions"],
        "density": density,
        "meta_json": np.array(meta_json, dtype=object),
    }
    arrays = {key: value for key, value in arrays.items() if value is not None}
    if sim.get("charges") is not None:
        arrays["charges"] = sim["charges"]
    if radii is not None:
        arrays["radii"] = radii

    np.savez_compressed(out_path, **arrays)
    print(f"Saved with density: {out_path}")


def process_file(path: Path, overwrite: bool = False, suffix: str = "_with_density", include_radii: bool = True,
                 print_every: int | None = None):
    sim = load_npz(path)
    densities, radii = compute_density_and_radius_series(
        sim["positions"],
        charges=sim.get("charges"),
        print_every=print_every,
        split_by_charge=True,
    )
    if not include_radii:
        radii = None

    out_path = path if overwrite else path.with_name(f"{path.stem}{suffix}.npz")
    save_with_density(sim, densities, radii, out_path)


def process_files(files, overwrite: bool = False, suffix: str = "_with_density", include_radii: bool = True,
                  print_every: int | None = None):
    """
    Programmatic API usable from VS Code/REPL:
        process_files(["data/file1.npz", "data/file2.npz"], overwrite=False)
    """
    for name in files:
        process_file(
            Path(name),
            overwrite=overwrite,
            suffix=suffix,
            include_radii=include_radii,
            print_every=print_every,
        )
