import sys
import unittest
from pathlib import Path
import json
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.config import SimulationConfig, load_config_file, validate_config


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

    def test_init_radii_requires_low_and_high(self):
        config = SimulationConfig(
            n_particles=2,
            charge_values=(1.0, 2.0),
            charge_counts=(1, 1),
            init_radii={"low": 0.5},
        )
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_init_radii_requires_charge_populations(self):
        config = SimulationConfig(init_radii={"low": 0.5, "high": 1.0})
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_init_radii_loads_from_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "n_particles": 2,
                        "charge_values": [1.0, 2.0],
                        "charge_counts": [1, 1],
                        "init_radii": {"low": 0.5, "high": 1.0},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config_file(path)

        self.assertEqual(config.init_radii, {"low": 0.5, "high": 1.0})
        validate_config(config)

    def test_init_radii_ranges_load_from_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "config.json"
            path.write_text(
                json.dumps(
                    {
                        "n_particles": 2,
                        "charge_values": [1.0, 2.0],
                        "charge_counts": [1, 1],
                        "init_radii": {"low": [0.0, 1.0], "high": [2.0, 3.0]},
                    }
                ),
                encoding="utf-8",
            )
            config = load_config_file(path)

        self.assertEqual(config.init_radii, {"low": [0.0, 1.0], "high": [2.0, 3.0]})
        validate_config(config)

    def test_init_radii_range_rejects_negative_inner(self):
        config = SimulationConfig(
            n_particles=2,
            charge_values=(1.0, 2.0),
            charge_counts=(1, 1),
            init_radii={"low": [-0.1, 1.0], "high": [2.0, 3.0]},
        )
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_init_radii_range_rejects_reversed_bounds(self):
        config = SimulationConfig(
            n_particles=2,
            charge_values=(1.0, 2.0),
            charge_counts=(1, 1),
            init_radii={"low": [1.0, 1.0], "high": [2.0, 3.0]},
        )
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_init_radii_range_rejects_wrong_length(self):
        config = SimulationConfig(
            n_particles=2,
            charge_values=(1.0, 2.0),
            charge_counts=(1, 1),
            init_radii={"low": [0.0, 1.0, 2.0], "high": [2.0, 3.0]},
        )
        with self.assertRaises(ValueError):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
