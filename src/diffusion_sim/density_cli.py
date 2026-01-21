import argparse
from pathlib import Path

import numpy as np

from .io.npz import load_npz
from .postprocess.density_voronoi import (
    compute_density_and_radius_series,
    process_file,
    process_files,
    save_with_density,
)


def _parse_args():
    parser = argparse.ArgumentParser(description="Compute Voronoi density/radii for simulation outputs.")
    parser.add_argument("files", nargs="+", help="One or more .npz files to process.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input files in-place.")
    parser.add_argument("--suffix", default="_with_density", help="Suffix for output files when not overwriting.")
    parser.add_argument("--no-radii", action="store_true", help="Skip saving radii arrays.")
    parser.add_argument("--print-every", type=int, help="Print progress every N steps during density computation.")
    parser.add_argument("--stride", type=int, default=1, help="Compute density every N steps (default 1).")
    return parser.parse_args()


def main():
    args = _parse_args()
    include_radii = not args.no_radii
    files = [str(Path(path)) for path in args.files]

    if args.stride <= 1:
        if len(files) == 1:
            process_file(
                Path(files[0]),
                overwrite=args.overwrite,
                suffix=args.suffix,
                include_radii=include_radii,
                print_every=args.print_every,
            )
            return

        process_files(
            files,
            overwrite=args.overwrite,
            suffix=args.suffix,
            include_radii=include_radii,
            print_every=args.print_every,
        )
        return

    for path_str in files:
        path = Path(path_str)
        sim = load_npz(path)
        r_hist = sim["positions"]
        r_hist_sub = r_hist[:: args.stride]
        density_raw, radii_raw = compute_density_and_radius_series(
            r_hist_sub,
            print_every=args.print_every,
        )

        total_steps = r_hist.shape[0]
        n = r_hist.shape[1]
        density = np.full((total_steps, n), np.nan, dtype=np.float64)
        density[:: args.stride] = density_raw

        radii = None
        if include_radii:
            radii = np.full((total_steps, n), np.nan, dtype=np.float64)
            radii[:: args.stride] = radii_raw

        out_path = path if args.overwrite else path.with_name(f"{path.stem}{args.suffix}.npz")
        save_with_density(
            sim,
            density,
            radii,
            out_path,
            density_stride=args.stride,
        )


if __name__ == "__main__":
    main()
