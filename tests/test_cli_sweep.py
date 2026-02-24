import sys
import unittest
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.cli import _build_sweep_configs, _parse_sweep_specs
from diffusion_sim.config import SimulationConfig


def _args_with_sweep(sweep):
    return Namespace(
        sweep=sweep,
        out=None,
        resume_from=None,
    )


class TestCliSweep(unittest.TestCase):
    def test_parse_single_sweep_coerces_numeric_values(self):
        config = SimulationConfig(k=1.0)
        specs = _parse_sweep_specs(_args_with_sweep(["k=0,1,2"]), config)
        self.assertEqual(len(specs), 1)
        self.assertEqual(specs[0][0], "k")
        self.assertEqual(specs[0][1], [0.0, 1.0, 2.0])

    def test_parse_two_sweeps_requires_equal_lengths(self):
        config = SimulationConfig()
        with self.assertRaises(ValueError):
            _parse_sweep_specs(
                _args_with_sweep(["k=0,1", "diffusion_coeff=0.1,0.2,0.3"]),
                config,
            )

    def test_build_sweep_configs_uses_zipped_pairs(self):
        config = SimulationConfig(k=1.0, diffusion_coeff=0.0)
        specs = [
            ("k", [0.0, 2.0]),
            ("diffusion_coeff", [0.1, 0.5]),
        ]
        runs = _build_sweep_configs(config, specs)
        self.assertEqual(len(runs), 2)
        self.assertEqual([r.k for r in runs], [0.0, 2.0])
        self.assertEqual([r.diffusion_coeff for r in runs], [0.1, 0.5])

    def test_parse_tuple_sweep_for_charge_counts(self):
        config = SimulationConfig(n_particles=700, charge_counts=(50, 650))
        specs = _parse_sweep_specs(
            _args_with_sweep(["charge-counts=[50,650],[100,600]"]),
            config,
        )
        self.assertEqual(specs[0][0], "charge_counts")
        self.assertEqual(specs[0][1], [(50, 650), (100, 600)])


if __name__ == "__main__":
    unittest.main()
