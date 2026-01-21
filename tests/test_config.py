import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.config import SimulationConfig, validate_config


class TestConfigValidation(unittest.TestCase):
    def test_rejects_k_zero(self):
        config = SimulationConfig(k=0.0)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_non_positive_dt(self):
        config = SimulationConfig(dt=0.0)
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_rejects_non_positive_steps(self):
        config = SimulationConfig(steps=0)
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
