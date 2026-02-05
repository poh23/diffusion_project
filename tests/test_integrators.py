import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.integrators.dop853 import run_dop853_chunked


class TestIntegrators(unittest.TestCase):
    def test_dop853_advances_final_state(self):
        r0 = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        r_final, positions, _, _, times = run_dop853_chunked(
            r0,
            k=1.0,
            v0=1.0,
            l=1.0,
            r_floor=1e-12,
            dt=0.1,
            steps=1,
            t0=0.0,
            rtol=1e-9,
            atol=1e-12,
            chunk_steps=10,
        )

        self.assertEqual(times.shape[0], 1)
        self.assertFalse(np.allclose(r_final, positions[-1]))


if __name__ == "__main__":
    unittest.main()
