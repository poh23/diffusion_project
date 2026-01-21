import unittest
import numpy as np
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.integrators.rk23 import run_rk23_dynamic


def _min_pairwise_dist(r, r_floor):
    n = r.shape[0]
    if n < 2:
        return r_floor
    min_dist_sq = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = r[i, 0] - r[j, 0]
            dy = r[i, 1] - r[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
    dist = np.sqrt(min_dist_sq)
    return dist if dist > r_floor else r_floor


class TestRK23Adaptive(unittest.TestCase):
    def test_no_nans_k_negative(self):
        rng = np.random.default_rng(0)
        n = 50
        r0 = rng.uniform(0.0, 1.0, size=(n, 2))

        r_final, positions, _, _, times = run_rk23_dynamic(
            r0,
            k=-1.0,
            v0=1.0,
            l=1.0,
            r_floor=1e-4,
            t_span=(0.0, 0.1),
            rtol=1e-6,
            atol=1e-9,
            max_step_global=0.02,
            eta=0.05,
            recompute_every=5,
            record=True,
        )

        self.assertTrue(np.all(np.isfinite(positions)))
        self.assertTrue(np.all(np.isfinite(times)))
        self.assertTrue(np.all(np.isfinite(r_final)))

    def test_repulsive_increases_min_distance(self):
        rng = np.random.default_rng(1)
        n = 50
        r0 = rng.uniform(0.0, 1.0, size=(n, 2))
        r_floor = 1e-4

        _, positions, _, _, _ = run_rk23_dynamic(
            r0,
            k=3.0,
            v0=1.0,
            l=1.0,
            r_floor=r_floor,
            t_span=(0.0, 1e-4),
            rtol=1e-6,
            atol=1e-9,
            max_step_global=1e-4,
            eta=0.05,
            recompute_every=1,
            record=True,
        )

        if len(positions) < 2:
            self.skipTest("RK23 took no steps; cannot compare distances.")

        d0 = _min_pairwise_dist(positions[0], r_floor)
        d1 = _min_pairwise_dist(positions[1], r_floor)
        self.assertGreaterEqual(d1, d0)


if __name__ == "__main__":
    unittest.main()
