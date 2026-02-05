import sys
import unittest
from pathlib import Path
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

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


if __name__ == "__main__":
    unittest.main()
