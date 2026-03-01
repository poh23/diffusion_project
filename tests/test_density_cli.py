import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.density_cli import _collect_input_paths, process_directory
from diffusion_sim.io.npz import load_npz, save_npz


def _sample_sim():
    positions = np.array(
        [[
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
            [1.0, 1.0],
        ]],
        dtype=np.float64,
    )
    return {
        "positions": positions,
        "times": np.array([0.0], dtype=np.float64),
        "energy": np.array([0.0], dtype=np.float64),
        "std": np.array([0.0], dtype=np.float64),
        "final_positions": positions[-1],
        "meta": {"seed": 0},
    }


class TestDensityCli(unittest.TestCase):
    def test_collect_input_paths_auto_expands_directories_without_flag(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "run_a.npz", _sample_sim())
            save_npz(tmp_path / "run_b.npz", _sample_sim())

            paths = _collect_input_paths([str(tmp_path)])

        self.assertEqual([path.name for path in paths], ["run_a.npz", "run_b.npz"])

    def test_process_directory_runs_on_all_npz_files(self):
        sim = _sample_sim()

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "run_a.npz", sim)
            save_npz(tmp_path / "run_b.npz", sim)
            (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")

            process_directory(tmp_path)

            out_a = load_npz(tmp_path / "run_a_with_density.npz")
            out_b = load_npz(tmp_path / "run_b_with_density.npz")

        self.assertEqual(out_a["density"].shape, (1, 4))
        self.assertEqual(out_b["density"].shape, (1, 4))
        self.assertEqual(out_a["radii"].shape, (1, 4))
        self.assertEqual(out_b["radii"].shape, (1, 4))


if __name__ == "__main__":
    unittest.main()
