import sys
import unittest
from pathlib import Path
import tempfile

import h5py
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.io.h5 import save_h5
from diffusion_sim.io.h5_batch import append_h5_batch, open_h5_batch
from diffusion_sim.simulation import run_simulation
from diffusion_sim.io.npz import load_npz, save_npz


class TestIO(unittest.TestCase):
    def test_save_load_roundtrip(self):
        sim = run_simulation(
            n_particles=5,
            k=1.0,
            v0=1.0,
            l=1.0,
            t_duration=0.02,
            save_every=0.01,
            method="rk23",
            seed=0,
            r_floor=1e-12,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "sim.npz"
            save_npz(out_path, sim)
            loaded = load_npz(out_path)

        self.assertEqual(loaded["positions"].shape, sim["positions"].shape)
        self.assertEqual(loaded["times"].shape, sim["times"].shape)
        self.assertEqual(loaded["energy"].shape, sim["energy"].shape)
        self.assertIn("meta", loaded)

    def test_open_h5_batch_resume_migrates_fixed_size_datasets(self):
        sim = {
            "positions": np.zeros((2, 3, 2), dtype=np.float64),
            "times": np.array([0.0, 1.0], dtype=np.float64),
            "energy": np.array([2.0, 1.0], dtype=np.float64),
            "std": np.array([0.1, 0.2], dtype=np.float64),
            "final_positions": np.zeros((3, 2), dtype=np.float64),
            "charges": np.ones(3, dtype=np.float64),
            "meta": {"seed": 0},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "fixed.h5"
            save_h5(out_path, sim)

            with open_h5_batch(out_path, 3, resume=True, chunk_len=2) as h5:
                append_h5_batch(
                    h5,
                    np.ones((1, 3, 2), dtype=np.float64),
                    np.array([2.0], dtype=np.float64),
                    np.array([0.5], dtype=np.float64),
                    np.array([0.3], dtype=np.float64),
                )

            with h5py.File(out_path, "r") as h5:
                self.assertEqual(h5["positions"].shape, (3, 3, 2))
                self.assertIsNotNone(h5["positions"].chunks)
                self.assertIsNone(h5["positions"].maxshape[0])
                self.assertTrue(np.array_equal(h5["times"][()], np.array([0.0, 1.0, 2.0])))


if __name__ == "__main__":
    unittest.main()
