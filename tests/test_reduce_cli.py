import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.io.h5 import load_h5, save_h5
from diffusion_sim.io.npz import load_npz, save_npz
from diffusion_sim.reduce_cli import (
    _collect_input_paths,
    process_path,
    process_paths,
    reduce_sim_dict,
)


def _sample_sim(n_frames=6):
    positions = np.arange(n_frames * 4 * 2, dtype=np.float64).reshape(n_frames, 4, 2)
    return {
        "positions": positions,
        "times": np.arange(n_frames, dtype=np.float64) * 0.5,
        "energy": np.arange(n_frames, dtype=np.float64),
        "energy_aa": np.arange(n_frames, dtype=np.float64) + 10.0,
        "energy_ab": np.arange(n_frames, dtype=np.float64) + 20.0,
        "energy_bb": np.arange(n_frames, dtype=np.float64) + 30.0,
        "std": np.arange(n_frames, dtype=np.float64) + 40.0,
        "density": np.arange(n_frames * 4, dtype=np.float64).reshape(n_frames, 4),
        "radii": np.arange(n_frames * 4, dtype=np.float64).reshape(n_frames, 4) + 100.0,
        "final_positions": positions[-1],
        "charges": np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64),
        "meta": {"seed": 0},
    }


class TestReduceCli(unittest.TestCase):
    def test_reduce_sim_dict_stride_updates_frame_arrays_and_final_positions(self):
        sim = _sample_sim()

        reduced = reduce_sim_dict(sim, source_path=Path("run.npz"), stride=2)

        self.assertTrue(np.array_equal(reduced["times"], np.array([0.0, 1.0, 2.0])))
        self.assertTrue(np.array_equal(reduced["positions"], sim["positions"][[0, 2, 4]]))
        self.assertTrue(np.array_equal(reduced["energy_aa"], sim["energy_aa"][[0, 2, 4]]))
        self.assertTrue(np.array_equal(reduced["density"], sim["density"][[0, 2, 4]]))
        self.assertTrue(np.array_equal(reduced["charges"], sim["charges"]))
        self.assertTrue(np.array_equal(reduced["final_positions"], reduced["positions"][-1]))
        self.assertEqual(reduced["meta"]["reduction_original_frame_count"], 6)
        self.assertEqual(reduced["meta"]["reduction_frame_count"], 3)
        self.assertEqual(reduced["meta"]["reduction_stride"], 2)

    def test_reduce_sim_dict_time_range_applies_stride_after_filtering(self):
        sim = _sample_sim()

        reduced = reduce_sim_dict(
            sim,
            source_path=Path("run.npz"),
            stride=2,
            time_range=(0.5, 2.0),
        )

        self.assertTrue(np.array_equal(reduced["times"], np.array([0.5, 1.5])))
        self.assertTrue(np.array_equal(reduced["positions"], sim["positions"][[1, 3]]))
        self.assertEqual(reduced["meta"]["reduction_time_range"], [0.5, 2.0])

    def test_reduce_sim_dict_errors_when_selection_is_empty(self):
        sim = _sample_sim()

        with self.assertRaisesRegex(ValueError, "selected no frames"):
            reduce_sim_dict(sim, source_path=Path("run.npz"), time_range=(10.0, 11.0))

    def test_process_path_npz_writes_default_reduced_copy(self):
        sim = _sample_sim()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run.npz"
            save_npz(path, sim)

            process_path(path, stride=3, progress=False)
            reduced = load_npz(Path(tmp_dir) / "run_reduced.npz")

        self.assertTrue(np.array_equal(reduced["times"], np.array([0.0, 1.5])))
        self.assertTrue(np.array_equal(reduced["final_positions"], reduced["positions"][-1]))

    def test_process_path_h5_writes_reduced_copy(self):
        sim = _sample_sim()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "run.h5"
            save_h5(path, sim)

            process_path(path, stride=2, progress=False)
            reduced = load_h5(Path(tmp_dir) / "run_reduced.h5")

        self.assertTrue(np.array_equal(reduced["times"], np.array([0.0, 1.0, 2.0])))
        self.assertTrue(np.array_equal(reduced["positions"], sim["positions"][[0, 2, 4]]))
        self.assertTrue(np.array_equal(reduced["charges"], sim["charges"]))
        self.assertTrue(np.array_equal(reduced["final_positions"], reduced["positions"][-1]))

    def test_collect_input_paths_expands_supported_files_in_directory(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "run_b.npz", _sample_sim())
            save_h5(tmp_path / "run_a.h5", _sample_sim())
            (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

            paths = _collect_input_paths([str(tmp_path)])

        self.assertEqual([path.name for path in paths], ["run_a.h5", "run_b.npz"])

    def test_no_progress_suppresses_tqdm(self):
        sim = _sample_sim()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            first = tmp_path / "run_a.npz"
            second = tmp_path / "run_b.npz"
            save_npz(first, sim)
            save_npz(second, sim)

            with patch("diffusion_sim.reduce_cli.tqdm") as tqdm_mock:
                process_paths([first, second], stride=2, progress=False)

        tqdm_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
