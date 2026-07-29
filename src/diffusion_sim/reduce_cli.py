import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from .io.h5 import load_h5, save_h5
from .io.npz import load_npz, save_npz

SUPPORTED_INPUT_SUFFIXES = {".npz", ".h5", ".hdf5"}
FRAME_DATASETS = (
    "positions",
    "times",
    "energy",
    "std",
    "energy_aa",
    "energy_ab",
    "energy_bb",
    "density",
    "radii",
)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Reduce simulation output size by keeping selected saved frames."
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
    parser.add_argument("--suffix", default="_reduced", help="Suffix for output files when not overwriting.")
    parser.add_argument("--stride", type=int, default=1, help="Keep every Nth selected frame (default 1).")
    parser.add_argument(
        "--time-range",
        type=float,
        nargs=2,
        metavar=("START", "END"),
        help="Keep saved frames with START <= time <= END before applying stride.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    return parser.parse_args()


def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _load_sim(path: Path):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_h5(path)
    return load_npz(path)


def _save_sim(path: Path, sim: dict):
    if path.suffix.lower() in {".h5", ".hdf5"}:
        save_h5(path, sim)
    else:
        save_npz(path, sim)


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

        if path.suffix.lower() not in SUPPORTED_INPUT_SUFFIXES:
            raise ValueError(f"Unsupported simulation file type: {path}")
        resolved.append(path)

    return resolved


def _output_path(path: Path, overwrite: bool, suffix: str) -> Path:
    if overwrite:
        return path
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _selected_indices(times: np.ndarray, stride: int, time_range: tuple[float, float] | None) -> np.ndarray:
    if stride <= 0:
        raise ValueError("stride must be a positive integer")

    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")

    indices = np.arange(times.shape[0], dtype=np.int64)
    if time_range is not None:
        start, end = time_range
        if start > end:
            raise ValueError("time range START must be <= END")
        indices = indices[(times >= start) & (times <= end)]

    indices = indices[::stride]
    if indices.size == 0:
        raise ValueError("reduction selected no frames")
    return indices


def _reduced_meta(
    meta: dict,
    *,
    source_path: Path,
    stride: int,
    time_range: tuple[float, float] | None,
    original_frame_count: int,
    frame_count: int,
) -> dict:
    reduced = dict(meta or {})
    reduced["reduced_from"] = str(source_path)
    reduced["reduction_stride"] = int(stride)
    reduced["reduction_time_range"] = None if time_range is None else [float(time_range[0]), float(time_range[1])]
    reduced["reduction_original_frame_count"] = int(original_frame_count)
    reduced["reduction_frame_count"] = int(frame_count)
    return reduced


def reduce_sim_dict(
    sim: dict,
    *,
    source_path: Path,
    stride: int = 1,
    time_range: tuple[float, float] | None = None,
) -> dict:
    positions = np.asarray(sim["positions"])
    times = np.asarray(sim["times"], dtype=np.float64)
    if positions.shape[0] != times.shape[0]:
        raise ValueError("positions and times must have the same frame count")

    indices = _selected_indices(times, stride, time_range)
    reduced = dict(sim)
    for name in FRAME_DATASETS:
        value = sim.get(name)
        if value is not None:
            arr = np.asarray(value)
            if arr.shape[0] == times.shape[0]:
                reduced[name] = arr[indices]

    reduced["final_positions"] = reduced["positions"][-1]
    reduced["meta"] = _reduced_meta(
        sim.get("meta", {}) or {},
        source_path=source_path,
        stride=stride,
        time_range=time_range,
        original_frame_count=times.shape[0],
        frame_count=indices.size,
    )
    return reduced


def _copy_selected_dataset(src, dst, name: str, indices: np.ndarray, *, progress: bool):
    shape = (indices.size, *src[name].shape[1:])
    out = dst.create_dataset(name, shape=shape, dtype=src[name].dtype)
    iterator = range(0, indices.size, 256)
    if progress:
        iterator = tqdm(iterator, total=int(np.ceil(indices.size / 256)), desc=f"copy {name}", leave=False)

    for start in iterator:
        end = min(start + 256, indices.size)
        out[start:end] = src[name][indices[start:end]]


def reduce_h5_file(
    path: Path,
    out_path: Path,
    *,
    stride: int = 1,
    time_range: tuple[float, float] | None = None,
    progress: bool = True,
):
    path = Path(path)
    out_path = Path(out_path)
    tmp_path = out_path.with_name(f"{out_path.name}.tmp")

    with h5py.File(path, "r") as src:
        times = src["times"][()]
        indices = _selected_indices(times, stride, time_range)
        meta_json = src.attrs.get("meta_json")
        if meta_json is None:
            raise KeyError(f"{path}: missing meta_json attribute.")
        meta = json.loads(meta_json)
        reduced_meta = _reduced_meta(
            meta,
            source_path=path,
            stride=stride,
            time_range=time_range,
            original_frame_count=times.shape[0],
            frame_count=indices.size,
        )

        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(tmp_path, "w") as dst:
            for name in FRAME_DATASETS:
                if name in src and src[name].shape[0] == times.shape[0]:
                    _copy_selected_dataset(src, dst, name, indices, progress=progress)

            dst.create_dataset("final_positions", data=dst["positions"][-1], dtype=src["positions"].dtype)
            if "charges" in src:
                dst.create_dataset("charges", data=src["charges"][()], dtype=src["charges"].dtype)
            dst.attrs["meta_json"] = json.dumps(reduced_meta, default=_json_default)

    tmp_path.replace(out_path)
    print(f"Saved: {out_path}")
    print(f"Meta: {reduced_meta}")


def process_path(
    path: Path,
    *,
    overwrite: bool = False,
    suffix: str = "_reduced",
    stride: int = 1,
    time_range: tuple[float, float] | None = None,
    progress: bool = True,
):
    out_path = _output_path(path, overwrite, suffix)
    if path.suffix.lower() in {".h5", ".hdf5"}:
        reduce_h5_file(path, out_path, stride=stride, time_range=time_range, progress=progress)
        return

    sim = _load_sim(path)
    sim_out = reduce_sim_dict(sim, source_path=path, stride=stride, time_range=time_range)
    _save_sim(out_path, sim_out)


def process_paths(
    paths: list[Path],
    *,
    overwrite: bool = False,
    suffix: str = "_reduced",
    stride: int = 1,
    time_range: tuple[float, float] | None = None,
    progress: bool = True,
):
    print(f"[reduce] reducing {len(paths)} file(s)")
    iterator = paths
    if progress and len(paths) > 1:
        iterator = tqdm(paths, desc="files")

    for index, path in enumerate(iterator, start=1):
        if not progress or len(paths) == 1:
            print(f"[reduce] processing file {index}/{len(paths)}: {path}")
        process_path(
            path,
            overwrite=overwrite,
            suffix=suffix,
            stride=stride,
            time_range=time_range,
            progress=progress,
        )


def process_directory(
    directory: Path | str,
    *,
    overwrite: bool = False,
    suffix: str = "_reduced",
    stride: int = 1,
    time_range: tuple[float, float] | None = None,
    progress: bool = True,
):
    paths = _collect_input_paths([str(directory)], directory=True)
    process_paths(
        paths,
        overwrite=overwrite,
        suffix=suffix,
        stride=stride,
        time_range=time_range,
        progress=progress,
    )


def main():
    args = _parse_args()
    paths = _collect_input_paths(args.files, directory=args.directory)
    process_paths(
        paths,
        overwrite=args.overwrite,
        suffix=args.suffix,
        stride=args.stride,
        time_range=None if args.time_range is None else tuple(args.time_range),
        progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
