import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.config import SimulationConfig, validate_config
from diffusion_sim.external_potentials import compute_external_velocity
from diffusion_sim.forces import compute_total_velocity_overdamped, compute_velocity_overdamped
from diffusion_sim.integrators.dop853 import run_dop853_chunked
from diffusion_sim.integrators.rk23 import run_rk23_dynamic


class TestExternalPotentialConfig(unittest.TestCase):
    def test_harmonic_external_potential_is_valid(self):
        config = SimulationConfig(
            external_potential="harmonic",
            external_potential_params={"omega": 0.5, "center": [1.0, -1.0]},
        )
        validate_config(config)

    def test_unknown_external_potential_is_rejected(self):
        config = SimulationConfig(
            external_potential="unknown",
            external_potential_params={"omega": 1.0},
        )
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_missing_harmonic_omega_is_rejected(self):
        config = SimulationConfig(
            external_potential="harmonic",
            external_potential_params={},
        )
        with self.assertRaises(ValueError):
            validate_config(config)


class TestExternalPotentialDynamics(unittest.TestCase):
    def test_harmonic_velocity_points_to_center(self):
        r = np.array([[2.0, -1.0]], dtype=np.float64)
        vel = compute_external_velocity(
            r,
            "harmonic",
            {"omega": 0.5, "center": [0.0, 1.0]},
        )
        self.assertTrue(np.allclose(vel, [[-1.0, 1.0]]))

    def test_total_velocity_matches_pairwise_when_external_is_none(self):
        r = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        charges = np.ones(2, dtype=np.float64)
        pairwise = compute_velocity_overdamped(r, 1.0, 1.0, 1.0, 1e-12, charges)
        total = compute_total_velocity_overdamped(
            r,
            1.0,
            1.0,
            1.0,
            1e-12,
            charges,
            external_potential=None,
            external_potential_params=None,
        )
        self.assertTrue(np.allclose(pairwise, total))

    def test_rk23_single_particle_harmonic_trap_moves_toward_center(self):
        r0 = np.array([[1.0, 0.0]], dtype=np.float64)
        charges = np.ones(1, dtype=np.float64)
        r_final, *_ = run_rk23_dynamic(
            r0,
            k=1.0,
            v0=0.0,
            l=1.0,
            r_floor=1e-12,
            charges=charges,
            t_span=(0.0, 0.5),
            rtol=1e-8,
            atol=1e-10,
            max_step_global=0.1,
            record=False,
            external_potential="harmonic",
            external_potential_params={"omega": 1.0, "center": [0.0, 0.0]},
        )
        self.assertLess(abs(r_final[0, 0]), abs(r0[0, 0]))

    def test_dop853_single_particle_harmonic_trap_moves_toward_center(self):
        r0 = np.array([[1.0, 0.0]], dtype=np.float64)
        charges = np.ones(1, dtype=np.float64)
        r_final, *_ = run_dop853_chunked(
            r0,
            k=1.0,
            v0=0.0,
            l=1.0,
            r_floor=1e-12,
            dt=0.1,
            steps=3,
            t0=0.0,
            rtol=1e-9,
            atol=1e-12,
            chunk_steps=10,
            charges=charges,
            external_potential="harmonic",
            external_potential_params={"omega": 1.0, "center": [0.0, 0.0]},
        )
        self.assertLess(abs(r_final[0, 0]), abs(r0[0, 0]))


if __name__ == "__main__":
    unittest.main()
