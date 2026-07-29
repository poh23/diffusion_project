import sys
import unittest
import warnings
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.plotting import compute_std_by_charge, plot_std_by_charge


class TestStdDiagnostics(unittest.TestCase):
    def test_compute_std_by_charge_uses_radial_std(self):
        sim = {
            "times": np.array([1.0, 2.0], dtype=np.float64),
            "radii": np.array(
                [
                    [1.0, 3.0, 10.0, 14.0],
                    [2.0, 6.0, 20.0, 28.0],
                ],
                dtype=np.float64,
            ),
            "charges": np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float64),
            "meta": {},
        }

        times, std_by_charge = compute_std_by_charge(sim)

        self.assertTrue(np.allclose(times, [1.0, 2.0]))
        self.assertTrue(np.allclose(std_by_charge[1.0], [1.0, 2.0]))
        self.assertTrue(np.allclose(std_by_charge[5.0], [2.0, 4.0]))

    def test_compute_std_by_charge_self_similar_scales_radii_by_time(self):
        sim = {
            "times": np.array([1.0, 4.0], dtype=np.float64),
            "radii": np.array(
                [
                    [1.0, 3.0, 10.0, 14.0],
                    [2.0, 6.0, 20.0, 28.0],
                ],
                dtype=np.float64,
            ),
            "charges": np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float64),
            "meta": {"k": 0.0},
        }

        _, std_by_charge = compute_std_by_charge(sim, self_similar=True)

        self.assertTrue(np.allclose(std_by_charge[1.0], [1.0, 1.0]))
        self.assertTrue(np.allclose(std_by_charge[5.0], [2.0, 2.0]))

    def test_compute_std_by_charge_self_similar_t0_returns_nan_without_warning(self):
        sim = {
            "times": np.array([0.0, 1.0], dtype=np.float64),
            "radii": np.array(
                [
                    [1.0, 3.0],
                    [2.0, 6.0],
                ],
                dtype=np.float64,
            ),
            "charges": np.array([1.0, 1.0], dtype=np.float64),
            "meta": {"k": 0.0},
        }

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            _, std_by_charge = compute_std_by_charge(sim, self_similar=True)

        self.assertTrue(np.isnan(std_by_charge[1.0][0]))
        self.assertTrue(np.allclose(std_by_charge[1.0][1], 2.0))

    def test_plot_std_by_charge_returns_series(self):
        sim = {
            "times": np.array([1.0], dtype=np.float64),
            "radii": np.array([[1.0, 3.0]], dtype=np.float64),
            "charges": np.array([1.0, 1.0], dtype=np.float64),
            "meta": {},
        }

        fig, ax, times, std_by_charge = plot_std_by_charge(sim, show=False)

        self.assertIs(fig, ax.figure)
        self.assertTrue(np.allclose(times, [1.0]))
        self.assertTrue(np.allclose(std_by_charge[1.0], [1.0]))


if __name__ == "__main__":
    unittest.main()
