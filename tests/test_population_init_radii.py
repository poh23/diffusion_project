import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.config import SimulationConfig
from diffusion_sim.simulation.helpers import init_fresh_positions_and_charges


class TestPopulationInitRadii(unittest.TestCase):
    def test_population_specific_radii_bound_each_charge_population(self):
        config = SimulationConfig(
            n_particles=20,
            charge_values=(1.0, 5.0),
            charge_counts=(8, 12),
            init_radii={"low": 0.25, "high": 1.0},
            seed=123,
        )
        rng = np.random.default_rng(config.seed)

        positions, charges = init_fresh_positions_and_charges(config, rng)
        radii = np.linalg.norm(positions, axis=1)

        low_mask = np.isclose(charges, 1.0)
        high_mask = np.isclose(charges, 5.0)
        self.assertEqual(int(np.count_nonzero(low_mask)), 8)
        self.assertEqual(int(np.count_nonzero(high_mask)), 12)
        self.assertTrue(np.all(radii[low_mask] <= 0.25))
        self.assertTrue(np.all(radii[high_mask] <= 1.0))

    def test_scalar_init_radius_keeps_single_disk_behavior(self):
        config = SimulationConfig(n_particles=20, init_radius=0.75, seed=123)
        rng = np.random.default_rng(config.seed)

        positions, charges = init_fresh_positions_and_charges(config, rng)
        radii = np.linalg.norm(positions, axis=1)

        self.assertEqual(positions.shape, (20, 2))
        self.assertEqual(charges.shape, (20,))
        self.assertTrue(np.all(radii <= 0.75))
        self.assertTrue(np.allclose(charges, 1.0))

    def test_population_specific_radius_ranges_bound_each_charge_population(self):
        config = SimulationConfig(
            n_particles=40,
            charge_values=(1.0, 5.0),
            charge_counts=(15, 25),
            init_radii={"low": [0.0, 1.0], "high": [2.0, 3.0]},
            seed=123,
        )
        rng = np.random.default_rng(config.seed)

        positions, charges = init_fresh_positions_and_charges(config, rng)
        radii = np.linalg.norm(positions, axis=1)

        low_mask = np.isclose(charges, 1.0)
        high_mask = np.isclose(charges, 5.0)
        self.assertEqual(int(np.count_nonzero(low_mask)), 15)
        self.assertEqual(int(np.count_nonzero(high_mask)), 25)
        self.assertTrue(np.all(radii[low_mask] >= 0.0))
        self.assertTrue(np.all(radii[low_mask] <= 1.0))
        self.assertTrue(np.all(radii[high_mask] >= 2.0))
        self.assertTrue(np.all(radii[high_mask] <= 3.0))


if __name__ == "__main__":
    unittest.main()
