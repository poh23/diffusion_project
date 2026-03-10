import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.config import SimulationConfig, validate_config


class TestConfigValidation(unittest.TestCase):
    def test_rejects_non_positive_t_duration(self):
        config = SimulationConfig(t_duration=0.0)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_non_positive_save_every(self):
        config = SimulationConfig(save_every=0.0)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_dop853_requires_save_every(self):
        config = SimulationConfig(method="dop853", save_every=None)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_save_every_and_save_every_steps_are_mutually_exclusive(self):
        config = SimulationConfig(method="rk23", save_every=0.1, save_every_steps=10)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_dop853_rejects_save_every_steps(self):
        config = SimulationConfig(method="dop853", save_every=0.1, save_every_steps=10)
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
