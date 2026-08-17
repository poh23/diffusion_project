import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from diffusion_sim.config import SimulationConfig
from diffusion_sim.simulation.streaming import _run_dop853_stream


class TestDop853Streaming(unittest.TestCase):
    def test_auto_chunk_steps_when_config_uses_default_zero(self):
        config = SimulationConfig(
            method="dop853",
            save_every=0.1,
            t_duration=1.0,
            batch_every=0.5,
            out_format="h5",
            chunk_steps=0,
        )
        state = {
            "r0": np.array([[0.0, 0.0]], dtype=np.float64),
            "charges": np.ones(1, dtype=np.float64),
            "t_start_sim": 0.0,
            "t_end": 1.0,
            "resumed": False,
        }

        with patch("diffusion_sim.simulation.streaming.run_dop853_chunked") as run_mock:
            run_mock.return_value = (state["r0"], None, None, None, None, None, None, None)

            _run_dop853_stream(
                config,
                state=state,
                pbar=Mock(),
                accumulator=Mock(),
                last_state={},
                stop_condition=None,
            )

        self.assertGreater(run_mock.call_args.kwargs["chunk_steps"], 0)


if __name__ == "__main__":
    unittest.main()
