import argparse
from pathlib import Path

import numpy as np

from .io.h5 import load_h5, save_h5
from .io.npz import load_npz, save_npz
from .postprocess.density_voronoi import (
    compute_density_and_radius_series,
    process_file,
    process_files,
)

SUPPORTED_INPUT_SUFFIXES = {".npz", ".h5", ".hdf5"}


def _parse_args():
    parser = argparse.ArgumentParser(description="Compute Voronoi density/radii for simulation outputs.")
    parser.add_argument(
        "files",
        nargs="+",
        help="One or more simulation output files, or directories when used with --directory.",
    )
    parser.add_argument(
        "--directory",
        action="store_true",
        help="Treat each input as a directory and process all supported files inside it.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input files in-place.")
    parser.add_argument("--suffix", default="_with_density", help="Suffix for output files when not overwriting.")
    parser.add_argument("--no-radii", action="store_true", help="Skip saving radii arrays.")
    parser.add_argument("--print-every", type=int, help="Print progress every N steps during density computation.")
    parser.add_argument("--stride", type=int, default=1, help="Compute density every N steps (default 1).")
    return parser.parse_args()


def _load_sim(path: Path):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_h5(path)
    return load_npz(path)


def _save_sim(path: Path, sim: dict, density: np.ndarray, radii: np.ndarray | None, stride: int | None):
    sim_out = dict(sim)
    sim_out["density"] = density
    sim_out["radii"] = radii
    sim_out["meta"] = dict(sim.get("meta", {}) or {})
    if sim.get("charges") is not None:
        sim_out["meta"]["density_split_by_charge"] = True
    if stride is not None:
        sim_out["meta"]["density_stride"] = stride

    if path.suffix.lower() in {".h5", ".hdf5"}:
        save_h5(path, sim_out)
    else:
        save_npz(path, sim_out)


def _collect_input_paths(inputs: list[str], directory: bool = False) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)

        if path.is_dir():
            matches = sorted(
                child for child in path.iterdir()
                if child.is_file() and child.suffix.lower() in SUPPORTED_INPUT_SUFFIXES
            )
            if not matches:
                raise ValueError(f"No supported simulation files found in directory: {path}")
            resolved.extend(matches)
            continue

        if directory:
            raise ValueError(f"Expected a directory for --directory: {path}")

        resolved.append(path)

    return resolved


def process_path(
    path: Path,
    overwrite: bool = False,
    suffix: str = "_with_density",
    include_radii: bool = True,
    print_every: int | None = None,
    stride: int = 1,
):
    if stride <= 1 and path.suffix.lower() not in {".h5", ".hdf5"}:
        process_file(
            path,
            overwrite=overwrite,
            suffix=suffix,
            include_radii=include_radii,
            print_every=print_every,
        )
        return

    sim = _load_sim(path)
    r_hist = sim["positions"]
    charges = sim.get("charges")
    r_hist_sub = r_hist[::stride]
    density_raw, radii_raw = compute_density_and_radius_series(
        r_hist_sub,
        charges=charges,
        print_every=print_every,
        split_by_charge=True,
    )

    total_steps = r_hist.shape[0]
    n = r_hist.shape[1]
    density = np.full((total_steps, n), np.nan, dtype=np.float64)
    density[::stride] = density_raw

    radii = None
    if include_radii:
        radii = np.full((total_steps, n), np.nan, dtype=np.float64)
        radii[::stride] = radii_raw

    if overwrite:
        out_path = path
    else:
        out_path = path.with_name(f"{path.stem}{suffix}{path.suffix}")

    _save_sim(out_path, sim, density, radii, stride)


def process_paths(
    paths: list[Path],
    overwrite: bool = False,
    suffix: str = "_with_density",
    include_radii: bool = True,
    print_every: int | None = None,
    stride: int = 1,
):
    total_files = len(paths)
    print(f"[density] computing density for {total_files} file(s)")

    if stride <= 1 and all(path.suffix.lower() not in {".h5", ".hdf5"} for path in paths):
        if len(paths) == 1:
            print(f"[density] processing file 1/1: {paths[0]}")
            process_file(
                paths[0],
                overwrite=overwrite,
                suffix=suffix,
                include_radii=include_radii,
                print_every=print_every,
            )
            return

        for index, path in enumerate(paths, start=1):
            print(f"[density] processing file {index}/{total_files}: {path}")
        process_files(
            [str(path) for path in paths],
            overwrite=overwrite,
            suffix=suffix,
            include_radii=include_radii,
            print_every=print_every,
        )
        return

    for index, path in enumerate(paths, start=1):
        print(f"[density] processing file {index}/{total_files}: {path}")
        process_path(
            path,
            overwrite=overwrite,
            suffix=suffix,
            include_radii=include_radii,
            print_every=print_every,
            stride=stride,
        )


def process_directory(
    directory: Path | str,
    overwrite: bool = False,
    suffix: str = "_with_density",
    include_radii: bool = True,
    print_every: int | None = None,
    stride: int = 1,
):
    paths = _collect_input_paths([str(directory)], directory=True)
    process_paths(
        paths,
        overwrite=overwrite,
        suffix=suffix,
        include_radii=include_radii,
        print_every=print_every,
        stride=stride,
    )


def main():
    args = _parse_args()
    include_radii = not args.no_radii
    paths = _collect_input_paths(args.files, directory=args.directory)
    process_paths(
        paths,
        overwrite=args.overwrite,
        suffix=args.suffix,
        include_radii=include_radii,
        print_every=args.print_every,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
