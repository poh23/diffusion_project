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


def compute_density_series(r_hist: np.ndarray, print_every: int | None = None) -> np.ndarray:
    """
    r_hist: (steps, n, 2)
    Returns density per particle per step: (steps, n), density = 1 / Voronoi cell area.
    """
    steps, n, _ = r_hist.shape
    out = np.zeros((steps, n), dtype=np.float64)
    for i in range(steps):
        out[i] = voronoi_density(r_hist[i])
        if print_every and (i + 1) % print_every == 0:
            print(f"[density] processed step {i + 1}/{steps}")
    return out


def compute_density_and_radius_series(r_hist: np.ndarray, print_every: int | None = None):
    """
    r_hist: (steps, n, 2)
    Returns:
      densities: (steps, n)  # 1 / Voronoi cell area (0 for unbounded cells)
      radii:     (steps, n)  # sqrt(x^2 + y^2) for each particle
    """
    densities = compute_density_series(r_hist, print_every=print_every)
    radii = np.linalg.norm(r_hist, axis=2)
    return densities, radii


def save_with_density(sim: dict, density: np.ndarray, radii: np.ndarray | None, out_path: Path):
    meta = dict(sim["meta"])
    meta["density_method"] = "voronoi_2d"
    meta_json = json.dumps(meta)

    arrays = {
        "positions": sim["positions"],
        "times": sim["times"],
        "energy": sim["energy"],
        "std": sim["std"],
        "final_positions": sim["final_positions"],
        "density": density,
        "meta_json": np.array(meta_json, dtype=object),
    }
    if radii is not None:
        arrays["radii"] = radii

    np.savez_compressed(out_path, **arrays)
    print(f"Saved with density: {out_path}")


def process_file(path: Path, overwrite: bool = False, suffix: str = "_with_density", include_radii: bool = True,
                 print_every: int | None = None):
    sim = load_npz(path)
    densities, radii = compute_density_and_radius_series(sim["positions"], print_every=print_every)
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
