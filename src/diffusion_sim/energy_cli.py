import argparse
from pathlib import Path

import numpy as np

from .postprocess.energy_components import (
    compute_energy_component_series,
    load_sim_for_energy,
    process_energy_file,
    process_energy_files,
    save_sim_with_energy,
)

SUPPORTED_INPUT_SUFFIXES = {".npz", ".h5", ".hdf5"}


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-frame energy components (AA, AB, BB) for simulation outputs."
    )
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
    parser.add_argument(
        "--suffix",
        default="_with_energy_components",
        help="Suffix for output files when not overwriting.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        help="Print progress every N steps during energy-component computation.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Compute energy components every N steps and fill skipped frames with NaN (default 1).",
    )
    return parser.parse_args()


def _collect_input_paths(inputs: list[str], directory: bool = False) -> list[Path]:
    resolved: list[Path] = []
    for raw_path in inputs:
        path = Path(raw_path)

        if path.is_dir():
            matches = sorted(
                child
                for child in path.iterdir()
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
    suffix: str = "_with_energy_components",
    print_every: int | None = None,
    stride: int = 1,
):
    if stride <= 1:
        process_energy_file(
            path,
            overwrite=overwrite,
            suffix=suffix,
            print_every=print_every,
            show_progress=True,
        )
        return

    sim = load_sim_for_energy(path)
    meta = sim.get("meta", {}) or {}
    missing = [name for name in ("k", "v0", "l", "r_floor") if meta.get(name) is None]
    if missing:
        raise KeyError(f"Simulation metadata is missing required fields: {', '.join(missing)}")
    charges = sim.get("charges")
    if charges is None:
        raise KeyError("Simulation dict must contain 'charges' to compute energy components.")

    r_hist = np.asarray(sim["positions"])
    r_hist_sub = r_hist[::stride]
    energy_raw, energy_aa_raw, energy_ab_raw, energy_bb_raw = compute_energy_component_series(
        r_hist_sub,
        k=meta["k"],
        v0=meta["v0"],
        l=meta["l"],
        r_floor=meta["r_floor"],
        charges=np.asarray(charges),
        population_values=meta.get("charge_values"),
        print_every=print_every,
        show_progress=True,
    )

    total_steps = r_hist.shape[0]
    energy = np.full(total_steps, np.nan, dtype=np.float64)
    energy_aa = np.full(total_steps, np.nan, dtype=np.float64)
    energy_ab = np.full(total_steps, np.nan, dtype=np.float64)
    energy_bb = np.full(total_steps, np.nan, dtype=np.float64)
    energy[::stride] = energy_raw
    energy_aa[::stride] = energy_aa_raw
    energy_ab[::stride] = energy_ab_raw
    energy_bb[::stride] = energy_bb_raw

    sim_out = dict(sim)
    sim_out["energy"] = energy
    sim_out["energy_aa"] = energy_aa
    sim_out["energy_ab"] = energy_ab
    sim_out["energy_bb"] = energy_bb
    sim_out["meta"] = dict(meta)
    sim_out["meta"]["energy_components"] = "AA_AB_BB"
    sim_out["meta"]["energy_stride"] = int(stride)

    out_path = path if overwrite else path.with_name(f"{path.stem}{suffix}{path.suffix}")
    save_sim_with_energy(out_path, sim_out)


def process_paths(
    paths: list[Path],
    overwrite: bool = False,
    suffix: str = "_with_energy_components",
    print_every: int | None = None,
    stride: int = 1,
):
    total_files = len(paths)
    print(f"[energy] computing energy components for {total_files} file(s)")

    if len(paths) == 1:
        print(f"[energy] processing file 1/1: {paths[0]}")
        process_path(
            paths[0],
            overwrite=overwrite,
            suffix=suffix,
            print_every=print_every,
            stride=stride,
        )
        return

    for index, path in enumerate(paths, start=1):
        print(f"[energy] processing file {index}/{total_files}: {path}")

    if stride > 1:
        for path in paths:
            process_path(
                path,
                overwrite=overwrite,
                suffix=suffix,
                print_every=print_every,
                stride=stride,
            )
        return

    process_energy_files(
        [str(path) for path in paths],
        overwrite=overwrite,
        suffix=suffix,
        print_every=print_every,
        show_progress=True,
    )


def process_directory(
    directory: Path | str,
    overwrite: bool = False,
    suffix: str = "_with_energy_components",
    print_every: int | None = None,
    stride: int = 1,
):
    paths = _collect_input_paths([str(directory)], directory=True)
    process_paths(
        paths,
        overwrite=overwrite,
        suffix=suffix,
        print_every=print_every,
        stride=stride,
    )


def main():
    args = _parse_args()
    paths = _collect_input_paths(args.files, directory=args.directory)
    process_paths(
        paths,
        overwrite=args.overwrite,
        suffix=args.suffix,
        print_every=args.print_every,
        stride=args.stride,
    )


if __name__ == "__main__":
    main()
