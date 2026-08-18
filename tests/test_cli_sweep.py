import sys
import unittest
from argparse import Namespace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.cli import _build_sweep_configs, _config_from_args, _parse_sweep_specs
from diffusion_sim.config import SimulationConfig


def _args_with_sweep(sweep):
    return Namespace(
        sweep=sweep,
        out=None,
        resume_from=None,
    )


def _args_with_config_overrides(**overrides):
    values = dict(
        config=None,
        out=None,
        out_format=None,
        n_particles=None,
        k=None,
        v0=None,
        l=None,
        r_floor=None,
        init_radius=None,
        init_radius_low=None,
        init_radius_high=None,
        init_radius_low_min=None,
        init_radius_low_max=None,
        init_radius_high_min=None,
        init_radius_high_max=None,
        charge_values=None,
        charge_counts=None,
        t0=None,
        t_duration=None,
        save_every=None,
        save_every_steps=None,
        method=None,
        seed=None,
        rtol=None,
        atol=None,
        first_step=None,
        max_step_global=None,
        eta=None,
        recompute_every=None,
        interpolate_sampling=None,
        diffusion=None,
        diffusion_coeff=None,
        diffusion_seed=None,
        diffusion_noise_var=None,
        batch_every=None,
        target_batch_mb=None,
        max_wall_time=None,
        resume_from=None,
        resume_force=None,
    )
    values.update(overrides)
    return Namespace(**values)


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

    def test_config_from_args_sets_init_radii_from_cli(self):
        config = _config_from_args(
            _args_with_config_overrides(
                init_radius_low=0.25,
                init_radius_high=1.5,
            )
        )
        self.assertEqual(config.init_radii, {"low": 0.25, "high": 1.5})

    def test_config_from_args_sets_init_radii_ranges_from_cli(self):
        config = _config_from_args(
            _args_with_config_overrides(
                init_radius_low_min=0.0,
                init_radius_low_max=1.0,
                init_radius_high_min=2.0,
                init_radius_high_max=3.0,
            )
        )
        self.assertEqual(config.init_radii, {"low": [0.0, 1.0], "high": [2.0, 3.0]})

    def test_config_from_args_rejects_partial_init_radius_range(self):
        with self.assertRaises(ValueError):
            _config_from_args(_args_with_config_overrides(init_radius_low_min=0.0))

    def test_config_from_args_rejects_scalar_and_range_for_same_population(self):
        with self.assertRaises(ValueError):
            _config_from_args(
                _args_with_config_overrides(
                    init_radius_low=1.0,
                    init_radius_low_min=0.0,
                    init_radius_low_max=1.0,
                )
            )


if __name__ == "__main__":
    unittest.main()
