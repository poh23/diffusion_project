import tempfile
import sys
import unittest
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.io.h5 import save_h5
from diffusion_sim.io.npz import save_npz
from diffusion_sim.plotting import (
    compute_signed_mean_radius_difference,
    compute_signed_radial_wasserstein,
    plot_density_vs_radius,
    plot_inverse_density_squared_vs_radius_squared,
    plot_scaled_signed_mean_radius_difference_vs_ratio_by_k,
    plot_scaled_signed_radial_wasserstein_vs_ratio_by_k,
    plot_low_radius_vs_high_population_by_low_population,
    plot_low_radius_vs_high_population_by_initial_radii,
    plot_signed_mean_radius_difference_vs_charge_value_ratio,
    plot_signed_mean_radius_difference_vs_ratio,
    plot_signed_radial_wasserstein_vs_ratio,
    plot_low_inner_radius_vs_low_population,
)


class TestPlottingRadialWasserstein(unittest.TestCase):
    def test_plot_low_radius_vs_high_population_groups_by_low_population(self):
        sim_a = {
            "times": np.array([4.0], dtype=np.float64),
            "positions": np.array([[[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]]], dtype=np.float64),
            "charges": np.array([1.0, 1.0, 5.0], dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((3, 2), dtype=np.float64),
            "meta": {"k": 3.0},
        }
        sim_b = {
            "times": np.array([4.0], dtype=np.float64),
            "positions": np.array([[[1.0, 0.0], [3.0, 0.0], [10.0, 0.0], [11.0, 0.0]]], dtype=np.float64),
            "charges": np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {"k": 3.0},
        }
        sim_c = {
            "times": np.array([4.0], dtype=np.float64),
            "positions": np.array([[[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [10.0, 0.0]]], dtype=np.float64),
            "charges": np.array([1.0, 1.0, 1.0, 5.0], dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {"k": 3.0},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_npz(tmp_path / "b.npz", sim_b)
            save_npz(tmp_path / "a.npz", sim_a)
            save_npz(tmp_path / "c.npz", sim_c)

            _, ax, series = plot_low_radius_vs_high_population_by_low_population(
                tmp_path,
                time=4.0,
                low_percentile=100.0,
                scaled=True,
                show=False,
            )

        self.assertEqual(sorted(series), [2, 3])
        self.assertTrue(np.allclose(series[2]["n2"], [1.0, 2.0]))
        expected_a = 2.0 / ((3.0 * ((2.0 / 3.0) * 1.0 + (1.0 / 3.0) * 5.0) ** 2 * 4.0) ** (1.0 / 5.0))
        expected_b = 3.0 / ((4.0 * ((2.0 / 4.0) * 1.0 + (2.0 / 4.0) * 5.0) ** 2 * 4.0) ** (1.0 / 5.0))
        expected_c = 6.0 / ((4.0 * ((3.0 / 4.0) * 1.0 + (1.0 / 4.0) * 5.0) ** 2 * 4.0) ** (1.0 / 5.0))
        self.assertTrue(np.allclose(series[2]["r1"], [expected_a, expected_b]))
        self.assertTrue(np.allclose(series[3]["n2"], [1.0]))
        self.assertTrue(np.allclose(series[3]["r1"], [expected_c]))
        self.assertEqual(
            ax.get_ylabel(),
            r"$R_1 / \left[N\left((N_1/N)q_1 + (N_2/N)q_2\right)^2 t\right]^{1/(k+2)}$",
        )
        self.assertEqual([line.get_label() for line in ax.lines], ["N1=2", "N1=3"])
        self.assertEqual(ax.get_xscale(), "linear")
        self.assertEqual(ax.get_yscale(), "linear")

    def test_plot_low_radius_vs_high_population_loglog_fit_adds_power_law_per_series(self):
        def make_sim(low_outer_radius, n_high):
            low_positions = [[[1.0, 0.0], [low_outer_radius, 0.0]]]
            high_positions = [[[20.0 + i, 0.0] for i in range(n_high)]]
            positions = np.array([low_positions[0] + high_positions[0]], dtype=np.float64)
            return {
                "times": np.array([1.0], dtype=np.float64),
                "positions": positions,
                "charges": np.array([1.0, 1.0] + [5.0] * n_high, dtype=np.float64),
                "energy": np.array([0.0], dtype=np.float64),
                "std": np.array([0.0], dtype=np.float64),
                "final_positions": np.zeros((2 + n_high, 2), dtype=np.float64),
                "meta": {"k": -1.0},
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_npz(tmp_path / "n2_1.npz", make_sim(2.0, 1))
            save_npz(tmp_path / "n2_2.npz", make_sim(4.0, 2))
            save_npz(tmp_path / "n2_4.npz", make_sim(8.0, 4))

            _, ax, series = plot_low_radius_vs_high_population_by_low_population(
                tmp_path,
                time=1.0,
                low_percentile=100.0,
                scaled=False,
                loglog_fit=True,
                show=False,
            )

        self.assertTrue(np.allclose(series[2]["n2"], [1.0, 2.0, 4.0]))
        self.assertTrue(np.allclose(series[2]["r1"], [2.0, 4.0, 8.0]))
        self.assertEqual(ax.get_xscale(), "log")
        self.assertEqual(ax.get_yscale(), "log")
        self.assertEqual(len(ax.lines), 2)
        fit_line = ax.lines[1]
        self.assertIn("Fit N1=2", fit_line.get_label())
        self.assertTrue(np.allclose(fit_line.get_ydata(), 2.0 * fit_line.get_xdata()))

    def test_plot_low_radius_vs_high_population_normalizes_each_series_by_first_value(self):
        def make_sim(low_outer_radius, n_high):
            positions = np.array(
                [[[1.0, 0.0], [low_outer_radius, 0.0]] + [[20.0 + i, 0.0] for i in range(n_high)]],
                dtype=np.float64,
            )
            return {
                "times": np.array([1.0], dtype=np.float64),
                "positions": positions,
                "charges": np.array([1.0, 1.0] + [5.0] * n_high, dtype=np.float64),
                "energy": np.array([0.0], dtype=np.float64),
                "std": np.array([0.0], dtype=np.float64),
                "final_positions": np.zeros((2 + n_high, 2), dtype=np.float64),
                "meta": {"k": -1.0},
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_npz(tmp_path / "n2_2.npz", make_sim(6.0, 2))
            save_npz(tmp_path / "n2_1.npz", make_sim(3.0, 1))

            _, ax, series = plot_low_radius_vs_high_population_by_low_population(
                tmp_path,
                time=1.0,
                low_percentile=100.0,
                scaled=False,
                normalize=True,
                show=False,
            )

        self.assertTrue(np.allclose(series[2]["n2"], [1.0, 2.0]))
        self.assertTrue(np.allclose(series[2]["r1"], [1.0, 2.0]))
        self.assertEqual(ax.get_ylabel(), r"Normalized $R_1 / t^{k+2}$")

    def test_plot_low_radius_vs_high_population_includes_one_population_as_zero_high(self):
        sim = {
            "times": np.array([4.0], dtype=np.float64),
            "positions": np.array([[[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]]], dtype=np.float64),
            "charges": np.ones(3, dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((3, 2), dtype=np.float64),
            "meta": {"k": 3.0},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_npz(tmp_path / "one_pop.npz", sim)

            _, _, series = plot_low_radius_vs_high_population_by_low_population(
                tmp_path,
                time=4.0,
                low_percentile=100.0,
                scaled=True,
                show=False,
            )

        self.assertEqual(list(series), [3])
        self.assertTrue(np.allclose(series[3]["n2"], [0.0]))
        expected = 5.0 / ((3.0 * 1.0 ** 2 * 4.0) ** (1.0 / 5.0))
        self.assertTrue(np.allclose(series[3]["r1"], [expected]))

    def test_plot_low_radius_vs_high_population_unscaled_normalizes_by_time_power(self):
        sim = {
            "times": np.array([4.0], dtype=np.float64),
            "positions": np.array([[[1.0, 0.0], [5.0, 0.0], [10.0, 0.0]]], dtype=np.float64),
            "charges": np.array([1.0, 1.0, 5.0], dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((3, 2), dtype=np.float64),
            "meta": {"k": -1.0},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_npz(tmp_path / "run.npz", sim)

            _, ax, series = plot_low_radius_vs_high_population_by_low_population(
                tmp_path,
                time=4.0,
                low_percentile=100.0,
                scaled=False,
                show=False,
            )

        self.assertTrue(np.allclose(series[2]["r1"], [5.0 / (4.0 ** 1.0)]))
        self.assertEqual(ax.get_ylabel(), r"$R_1 / t^{k+2}$")

    def test_plot_low_radius_vs_high_population_groups_by_initial_radii(self):
        base = {
            "times": np.array([4.0], dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
        }
        sim_a = {
            **base,
            "positions": np.array([[[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]]], dtype=np.float64),
            "charges": np.array([1.0, 1.0, 5.0], dtype=np.float64),
            "final_positions": np.zeros((3, 2), dtype=np.float64),
            "meta": {"k": 3.0, "init_radii": {"low": [0.0, 1.0], "high": [2.0, 3.0]}},
        }
        sim_b = {
            **base,
            "positions": np.array([[[1.0, 0.0], [3.0, 0.0], [10.0, 0.0], [11.0, 0.0]]], dtype=np.float64),
            "charges": np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {"k": 3.0, "init_radii": {"low": [0.0, 1.0], "high": [2.0, 3.0]}},
        }
        sim_c = {
            **base,
            "positions": np.array([[[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [10.0, 0.0]]], dtype=np.float64),
            "charges": np.array([1.0, 1.0, 1.0, 5.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {"k": 3.0, "init_radii": {"low": [0.0, 2.0], "high": [3.0, 4.0]}},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_npz(tmp_path / "b.npz", sim_b)
            save_npz(tmp_path / "a.npz", sim_a)
            save_npz(tmp_path / "c.npz", sim_c)

            _, ax, series = plot_low_radius_vs_high_population_by_initial_radii(
                tmp_path,
                time=4.0,
                low_percentile=100.0,
                scaled=True,
                show=False,
            )

        self.assertEqual(
            sorted(series),
            ["low=[0,1], high=[2,3]", "low=[0,2], high=[3,4]"],
        )
        self.assertTrue(np.allclose(series["low=[0,1], high=[2,3]"]["n2"], [1.0, 2.0]))
        self.assertTrue(np.allclose(series["low=[0,2], high=[3,4]"]["n2"], [1.0]))
        self.assertEqual(
            [line.get_label() for line in ax.lines],
            ["low=[0,1], high=[2,3]", "low=[0,2], high=[3,4]"],
        )
        self.assertNotEqual(ax.lines[0].get_color(), ax.lines[1].get_color())

    def test_plot_low_radius_vs_high_population_rejects_scaled_zero_time(self):
        sim = {
            "times": np.array([0.0], dtype=np.float64),
            "positions": np.array([[[1.0, 0.0], [2.0, 0.0]]], dtype=np.float64),
            "charges": np.ones(2, dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((2, 2), dtype=np.float64),
            "meta": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_npz(tmp_path / "zero.npz", sim)

            with self.assertRaises(ValueError):
                plot_low_radius_vs_high_population_by_low_population(
                    tmp_path,
                    time=0.0,
                    scaled=True,
                    show=False,
                )

    def test_plot_low_radius_vs_high_population_reads_h5_nearest_frame_scaled(self):
        sim = {
            "times": np.array([1.0, 4.0], dtype=np.float64),
            "positions": np.array(
                [
                    [[100.0, 0.0], [200.0, 0.0], [300.0, 0.0]],
                    [[1.0, 0.0], [2.0, 0.0], [10.0, 0.0]],
                ],
                dtype=np.float64,
            ),
            "charges": np.array([1.0, 1.0, 5.0], dtype=np.float64),
            "energy": np.array([0.0, 0.0], dtype=np.float64),
            "std": np.array([0.0, 0.0], dtype=np.float64),
            "final_positions": np.zeros((3, 2), dtype=np.float64),
            "meta": {"k": 3.0},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_h5(tmp_path / "run.h5", sim)

            _, _, series = plot_low_radius_vs_high_population_by_low_population(
                tmp_path,
                time=4.0,
                low_percentile=100.0,
                scaled=True,
                show=False,
            )

        expected = 2.0 / ((3.0 * ((2.0 / 3.0) * 1.0 + (1.0 / 3.0) * 5.0) ** 2 * 4.0) ** (1.0 / 5.0))
        self.assertTrue(np.allclose(series[2]["r1"], [expected]))

    def test_plot_low_inner_radius_vs_low_population_scales_r1(self):
        sim_a = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 2.0, 10.0, 11.0]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float64),
            "positions": np.array([[[1.0, 0.0], [2.0, 0.0], [10.0, 0.0], [11.0, 0.0]]], dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {"k": 3.0},
        }
        sim_b = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[3.0, 4.0, 5.0, 10.0]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([1.0, 1.0, 1.0, 5.0], dtype=np.float64),
            "positions": np.array([[[3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [10.0, 0.0]]], dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {"k": 3.0},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_npz(tmp_path / "a.npz", sim_a)
            save_npz(tmp_path / "b.npz", sim_b)

            _, ax, n_low, r1, _, _ = plot_low_inner_radius_vs_low_population(
                tmp_path,
                time=4.0,
                low_percentile=100.0,
                show=False,
            )

        scale = 4.0 ** (1.0 / 5.0)
        expected_a = 2.0 / scale
        expected_b = 5.0 / scale
        self.assertTrue(np.allclose(n_low, [2.0, 3.0]))
        self.assertTrue(np.allclose(r1, [expected_a, expected_b]))
        self.assertEqual(ax.get_xlabel(), r"Low-Charge Population Size $N_{\mathrm{low}}$")
        self.assertEqual(ax.get_ylabel(), r"$r_1 / t^{1/(k+2)}$")
        self.assertTrue(any("Fit:" in line.get_label() for line in ax.lines))

    def test_plot_low_inner_radius_vs_low_population_supports_one_population(self):
        sim_a = {
            "times": np.array([4.0], dtype=np.float64),
            "positions": np.array([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]], dtype=np.float64),
            "charges": np.ones(3, dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((3, 2), dtype=np.float64),
            "meta": {"k": 0.0},
        }
        sim_b = {
            "times": np.array([4.0], dtype=np.float64),
            "positions": np.array([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]], dtype=np.float64),
            "charges": np.ones(4, dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {"k": 0.0},
        }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            save_npz(tmp_path / "a.npz", sim_a)
            save_npz(tmp_path / "b.npz", sim_b)

            _, ax, n_low, r1, _, _ = plot_low_inner_radius_vs_low_population(
                tmp_path,
                time=4.0,
                low_percentile=100.0,
                show=False,
            )

        self.assertTrue(np.allclose(n_low, [3.0, 4.0]))
        self.assertTrue(np.allclose(r1, [1.5, 2.0]))
        self.assertTrue(any("Fit:" in line.get_label() for line in ax.lines))

    def test_signed_mean_radius_difference_is_positive_when_higher_charge_is_farther_out(self):
        sim = {
            "times": np.array([0.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "meta": {},
        }

        actual_times, scores = compute_signed_mean_radius_difference(sim, times=[0.0])

        self.assertTrue(np.allclose(actual_times, [0.0]))
        self.assertTrue(np.allclose(scores, [2.0]))

    def test_signed_radial_wasserstein_is_positive_when_higher_charge_is_farther_out(self):
        sim = {
            "times": np.array([0.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "meta": {},
        }

        actual_times, scores = compute_signed_radial_wasserstein(
            sim,
            times=[0.0],
            n_bins=8,
            drop_zeros=False,
            min_points=1,
        )

        self.assertTrue(np.allclose(actual_times, [0.0]))
        self.assertEqual(scores.shape, (1,))
        self.assertGreater(scores[0], 0.0)

    def test_signed_radial_wasserstein_scaled_uses_scaled_radius_units(self):
        sim = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "meta": {"k": 0.0},
        }

        _, raw_scores = compute_signed_radial_wasserstein(
            sim,
            times=[4.0],
            n_bins=8,
            drop_zeros=False,
            min_points=1,
        )
        _, scaled_scores = compute_signed_radial_wasserstein(
            sim,
            times=[4.0],
            n_bins=8,
            drop_zeros=False,
            min_points=1,
            scaled=True,
        )

        self.assertGreater(raw_scores[0], 0.0)
        self.assertTrue(np.allclose(scaled_scores[0], raw_scores[0] / 2.0))

    def test_signed_mean_radius_difference_scaled_uses_scaled_radius_units(self):
        sim = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "meta": {"k": 0.0},
        }

        _, raw_scores = compute_signed_mean_radius_difference(sim, times=[4.0])
        _, scaled_scores = compute_signed_mean_radius_difference(
            sim,
            times=[4.0],
            scaled=True,
        )

        self.assertTrue(np.allclose(raw_scores, [2.0]))
        self.assertTrue(np.allclose(scaled_scores, [1.0]))

    def test_plot_signed_radial_wasserstein_vs_ratio_uses_high_charge_fraction(self):
        sim_a = {
            "times": np.array([0.0], dtype=np.float64),
            "radii": np.array([[1.0, 3.0, 3.2, 3.4]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }
        sim_b = {
            "times": np.array([0.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "run_b.npz", sim_b)
            save_npz(tmp_path / "run_a.npz", sim_a)

            _, _, ratios, scores, actual_times, labels = plot_signed_radial_wasserstein_vs_ratio(
                tmp_path,
                times=0.0,
                n_bins=8,
                drop_zeros=False,
                min_points=1,
                show=False,
            )

        self.assertTrue(np.allclose(ratios, [0.5, 0.75]))
        self.assertEqual(scores.shape, (2,))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.allclose(actual_times, [0.0, 0.0]))
        self.assertEqual(list(labels), ["run_b.npz", "run_a.npz"])

    def test_plot_signed_mean_radius_difference_vs_ratio_uses_high_charge_fraction(self):
        sim_a = {
            "times": np.array([0.0], dtype=np.float64),
            "radii": np.array([[1.0, 3.0, 3.2, 3.4]], dtype=np.float64),
            "charges": np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }
        sim_b = {
            "times": np.array([0.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "run_b.npz", sim_b)
            save_npz(tmp_path / "run_a.npz", sim_a)

            _, _, ratios, scores, actual_times, labels = plot_signed_mean_radius_difference_vs_ratio(
                tmp_path,
                times=0.0,
                show=False,
            )

        self.assertTrue(np.allclose(ratios, [0.5, 0.75]))
        self.assertTrue(np.allclose(scores, [2.0, 2.2]))
        self.assertTrue(np.allclose(actual_times, [0.0, 0.0]))
        self.assertEqual(list(labels), ["run_b.npz", "run_a.npz"])

    def test_plot_signed_mean_radius_difference_vs_charge_value_ratio_uses_charge_magnitudes(self):
        sim_a = {
            "times": np.array([0.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 2.0, 2.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }
        sim_b = {
            "times": np.array([0.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 3.0, 3.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "run_b.npz", sim_b)
            save_npz(tmp_path / "run_a.npz", sim_a)

            _, ax, x_values, scores, actual_times, labels = plot_signed_mean_radius_difference_vs_charge_value_ratio(
                tmp_path,
                times=0.0,
                show=False,
            )

        self.assertTrue(np.allclose(x_values, [2.0, 3.0]))
        self.assertTrue(np.allclose(scores, [2.0, 2.0]))
        self.assertTrue(np.allclose(actual_times, [0.0, 0.0]))
        self.assertEqual(list(labels), ["run_a.npz", "run_b.npz"])
        self.assertEqual(ax.get_xlabel(), r"Charge-Magnitude Ratio $\max(|q|) / \min(|q|)$")

    def test_plot_signed_radial_wasserstein_vs_ratio_supports_multiple_times(self):
        sim_a = {
            "times": np.array([0.0, 1.0], dtype=np.float64),
            "radii": np.array([[1.0, 3.0, 3.2, 3.4], [1.1, 3.1, 3.3, 3.5]], dtype=np.float64),
            "density": np.ones((2, 4), dtype=np.float64),
            "charges": np.array([-1.0, 1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((2, 4, 2), dtype=np.float64),
            "energy": np.array([0.0, 0.0], dtype=np.float64),
            "std": np.array([0.0, 0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }
        sim_b = {
            "times": np.array([0.0, 1.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2], [1.1, 1.3, 3.1, 3.3]], dtype=np.float64),
            "density": np.ones((2, 4), dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((2, 4, 2), dtype=np.float64),
            "energy": np.array([0.0, 0.0], dtype=np.float64),
            "std": np.array([0.0, 0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "run_b.npz", sim_b)
            save_npz(tmp_path / "run_a.npz", sim_a)

            _, _, ratios, scores, actual_times, labels = plot_signed_radial_wasserstein_vs_ratio(
                tmp_path,
                times=[0.0, 1.0],
                n_bins=8,
                drop_zeros=False,
                min_points=1,
                show=False,
            )

        self.assertTrue(np.allclose(ratios, [0.5, 0.75]))
        self.assertEqual(scores.shape, (2, 2))
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertTrue(np.allclose(actual_times, [[0.0, 0.0], [1.0, 1.0]]))
        self.assertEqual(list(labels), ["run_b.npz", "run_a.npz"])

    def test_plot_signed_radial_wasserstein_vs_ratio_scaled_forwards_scaling(self):
        sim = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {"k": 0.0},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            save_npz(tmp_path / "run_a.npz", sim)

            _, _, ratios, raw_scores, _, _ = plot_signed_radial_wasserstein_vs_ratio(
                tmp_path,
                times=4.0,
                n_bins=8,
                drop_zeros=False,
                min_points=1,
                show=False,
            )
            _, ax, _, scaled_scores, _, _ = plot_signed_radial_wasserstein_vs_ratio(
                tmp_path,
                times=4.0,
                n_bins=8,
                drop_zeros=False,
                min_points=1,
                scaled=True,
                show=False,
            )

        self.assertTrue(np.allclose(ratios, [0.5]))
        self.assertTrue(np.allclose(scaled_scores, raw_scores / 2.0))
        self.assertEqual(ax.get_ylabel(), "Scaled Signed Radial Wasserstein")

    def test_plot_scaled_signed_radial_wasserstein_vs_ratio_by_k_groups_by_subdirectory(self):
        sim_k0 = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }
        sim_k2 = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.4, 3.0, 3.4]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            k0_dir = tmp_path / "k0"
            k2_dir = tmp_path / "k2"
            k0_dir.mkdir()
            k2_dir.mkdir()
            save_npz(k0_dir / "run.npz", sim_k0)
            save_npz(k2_dir / "run.npz", sim_k2)

            _, ax, series = plot_scaled_signed_radial_wasserstein_vs_ratio_by_k(
                tmp_path,
                time=4.0,
                n_bins=8,
                drop_zeros=False,
                min_points=1,
                show=False,
            )

        self.assertEqual(len(series), 2)
        self.assertEqual([entry[0] for entry in series], [0.0, 2.0])
        self.assertEqual(ax.get_ylabel(), "Scaled Signed Radial Wasserstein")

    def test_plot_scaled_signed_mean_radius_difference_vs_ratio_by_k_groups_by_subdirectory(self):
        sim_k0 = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }
        sim_k2 = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.4, 3.0, 3.4]], dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "energy": np.array([0.0], dtype=np.float64),
            "std": np.array([0.0], dtype=np.float64),
            "final_positions": np.zeros((4, 2), dtype=np.float64),
            "meta": {},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            k0_dir = tmp_path / "k0"
            k2_dir = tmp_path / "k2"
            k0_dir.mkdir()
            k2_dir.mkdir()
            save_npz(k0_dir / "run.npz", sim_k0)
            save_npz(k2_dir / "run.npz", sim_k2)

            _, ax, series = plot_scaled_signed_mean_radius_difference_vs_ratio_by_k(
                tmp_path,
                time=4.0,
                show=False,
            )

        self.assertEqual(len(series), 2)
        self.assertEqual([entry[0] for entry in series], [0.0, 2.0])
        self.assertEqual(ax.get_ylabel(), "Scaled Signed Mean Radius Difference")

    def test_plot_density_vs_radius_scaled_rescales_axes(self):
        sim = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([-1.0, -1.0, 1.0, 1.0], dtype=np.float64),
            "meta": {"k": 0.0},
        }

        fig, ax = plot_density_vs_radius(
            sim,
            times=[4.0],
            scaled=True,
            drop_zeros=False,
            min_points=1,
            show=False,
        )

        first_line = ax.lines[0]
        self.assertTrue(np.allclose(first_line.get_xdata(), [0.5, 0.6]))
        self.assertTrue(np.allclose(first_line.get_ydata(), [4.0, 4.0]))
        self.assertEqual(ax.get_xlabel(), r"$r / t^{\frac{1}{k+2}}$")
        self.assertEqual(ax.get_ylabel(), r"$\rho t^{\frac{2}{k+2}}$")

    def test_plot_inverse_density_squared_vs_radius_squared_transforms_axes(self):
        sim = {
            "times": np.array([1.0], dtype=np.float64),
            "radii": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64),
            "density": np.array([[1.0, 2.0, 4.0, 8.0]], dtype=np.float64),
            "charges": np.ones(4, dtype=np.float64),
            "meta": {},
        }

        _, ax = plot_inverse_density_squared_vs_radius_squared(
            sim,
            times=[1.0],
            window_frac=0.0,
            split_by_charge=False,
            min_points=1,
            show=False,
        )

        first_line = ax.lines[0]
        self.assertTrue(np.allclose(first_line.get_xdata(), [1.0, 4.0, 9.0, 16.0]))
        self.assertTrue(np.allclose(first_line.get_ydata(), [1.0, 0.25, 0.0625, 0.015625]))
        self.assertEqual(ax.get_xlabel(), r"$r^2$")
        self.assertEqual(ax.get_ylabel(), r"$\rho^{-2}$")

    def test_plot_inverse_density_squared_vs_radius_squared_scaled_rescales_before_transform(self):
        sim = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 1.2, 3.0, 3.2]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float64),
            "meta": {"k": 0.0},
        }

        _, ax = plot_inverse_density_squared_vs_radius_squared(
            sim,
            times=[4.0],
            scaled=True,
            window_frac=0.0,
            drop_zeros=False,
            min_points=1,
            show=False,
        )

        first_line = ax.lines[0]
        radius_scale = (4.0 * 4.0 * (0.5 * 1.0 + 0.5 * 5.0) ** 2) ** 0.5
        density_scale = (1.0 / 2.0) * (4.0 * 4.0 * (0.5 * 1.0 + 0.5 * 5.0) ** 2)
        self.assertTrue(np.allclose(first_line.get_xdata(), [(1.0 / radius_scale) ** 2, (1.2 / radius_scale) ** 2]))
        self.assertTrue(np.allclose(first_line.get_ydata(), [density_scale ** -2, density_scale ** -2]))
        self.assertEqual(
            ax.get_xlabel(),
            r"$\left(r / \left[Nt\left(\frac{N_1}{N}q_1+\frac{N_2}{N}q_2\right)^2\right]^{\frac{1}{k+2}}\right)^2$",
        )
        self.assertEqual(
            ax.get_ylabel(),
            r"$\left(\rho\frac{1}{N_1}\left[Nt\left(\frac{N_1}{N}q_1+\frac{N_2}{N}q_2\right)^2\right]^{\frac{2}{k+2}}\right)^{-2}$",
        )

    def test_plot_inverse_density_squared_vs_radius_squared_scaled_supports_one_population(self):
        sim = {
            "times": np.array([4.0], dtype=np.float64),
            "radii": np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float64),
            "density": np.ones((1, 4), dtype=np.float64),
            "charges": np.ones(4, dtype=np.float64),
            "meta": {"k": 0.0},
        }

        _, ax = plot_inverse_density_squared_vs_radius_squared(
            sim,
            times=[4.0],
            scaled=True,
            window_frac=0.0,
            drop_zeros=False,
            min_points=1,
            show=False,
        )

        first_line = ax.lines[0]
        radius_scale = (4.0 * 4.0 * 1.0 ** 2) ** 0.5
        density_scale = (1.0 / 4.0) * (4.0 * 4.0 * 1.0 ** 2)
        self.assertTrue(
            np.allclose(
                first_line.get_xdata(),
                [(1.0 / radius_scale) ** 2, (2.0 / radius_scale) ** 2, (3.0 / radius_scale) ** 2, (4.0 / radius_scale) ** 2],
            )
        )
        self.assertTrue(np.allclose(first_line.get_ydata(), np.full(4, density_scale ** -2)))

    def test_plot_inverse_density_squared_vs_radius_squared_requires_density_array(self):
        sim = {
            "times": np.array([4.0], dtype=np.float64),
            "positions": np.zeros((1, 4, 2), dtype=np.float64),
            "density": None,
            "charges": np.ones(4, dtype=np.float64),
            "meta": {"k": 0.0},
        }

        with self.assertRaisesRegex(KeyError, "density"):
            plot_inverse_density_squared_vs_radius_squared(
                sim,
                times=[4.0],
                scaled=True,
                min_points=1,
                show=False,
            )

    def test_plot_inverse_density_squared_linear_fit_uses_selected_prefix_before_tail(self):
        x = np.arange(7, dtype=np.float64)
        y_tail = np.array([10.0, 8.0, 6.0, 4.0, 2.0, 100.0, 200.0], dtype=np.float64)
        y_other = np.array([1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0], dtype=np.float64)
        sim = {
            "times": np.array([1.0, 2.0], dtype=np.float64),
            "radii": np.sqrt(np.vstack([x, x])),
            "density": np.vstack([y_other ** -0.5, y_tail ** -0.5]),
            "charges": np.ones(7, dtype=np.float64),
            "meta": {},
        }

        _, ax = plot_inverse_density_squared_vs_radius_squared(
            sim,
            times=[1.0, 2.0],
            window_frac=0.0,
            split_by_charge=False,
            min_points=3,
            show=False,
            fit_linear=True,
            fit_time=2.0,
            fit_min_points=3,
            fit_smooth_window_frac=0.0,
        )

        fit_line = ax.lines[-1]
        self.assertEqual(fit_line.get_linestyle(), "--")
        self.assertTrue(np.allclose(fit_line.get_xdata(), [0.0, 4.0]))
        self.assertTrue(np.allclose(fit_line.get_ydata(), [10.0, 2.0]))
        self.assertIn("t~2", fit_line.get_label())


if __name__ == "__main__":
    unittest.main()
