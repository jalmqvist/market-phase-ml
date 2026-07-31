from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import (  # noqa: E402
    StrategyEvaluation,
    build_experiment_id,
    build_strategy_evaluation_id,
    strategy_evaluations_to_frame,
    write_strategy_evaluations_parquet,
)


class TestStrategyEvaluation(unittest.TestCase):
    def test_build_experiment_id_is_stable(self):
        first = build_experiment_id(
            experiment={"variant": "A", "factors": {"dl_enabled": False, "selector_enabled": True}},
            experiment_surface={"surface_id": "trend_vol", "state_id": "LVTF"},
            seed=42,
        )
        second = build_experiment_id(
            experiment={"factors": {"selector_enabled": True, "dl_enabled": False}, "variant": "A"},
            experiment_surface={"state_id": "LVTF", "surface_id": "trend_vol"},
            seed=42,
        )
        self.assertEqual(first, second)

    def test_strategy_evaluation_id_is_deterministic(self):
        first = build_strategy_evaluation_id(
            surface_id="trend_vol",
            surface_version="1.0.0",
            state_id="LVTF",
            strategy_id="PhaseAware_policy_v1",
            experiment_id="exp_abc",
        )
        second = build_strategy_evaluation_id(
            surface_id="trend_vol",
            surface_version="1.0.0",
            state_id="LVTF",
            strategy_id="PhaseAware_policy_v1",
            experiment_id="exp_abc",
        )
        self.assertEqual(first, second)
        self.assertTrue(first.startswith("eval_"))

    def test_strategy_evaluations_to_frame_serializes_metadata(self):
        evaluation = StrategyEvaluation(
            evaluation_id="eval_1",
            surface_id="trend_vol",
            surface_version="1.0.0",
            state_id="LVTF",
            strategy_id="PhaseAware",
            expected_return=1.2,
            expected_sharpe=0.5,
            expected_drawdown=-3.0,
            win_rate=None,
            confidence=0.4,
            stability=0.1,
            n_folds=12,
            n_trades=120,
            metadata={"experiment_id": "exp_1", "source": "walkforward"},
        )

        df = strategy_evaluations_to_frame([evaluation])
        self.assertEqual(len(df), 1)
        payload = json.loads(df.iloc[0]["metadata"])
        self.assertEqual(payload["experiment_id"], "exp_1")

    def test_write_strategy_evaluations_parquet(self):
        evaluation = StrategyEvaluation(
            evaluation_id="eval_1",
            surface_id="trend_vol",
            surface_version="1.0.0",
            state_id="LVTF",
            strategy_id="PhaseAware",
            expected_return=1.2,
            expected_sharpe=0.5,
            expected_drawdown=-3.0,
            win_rate=None,
            confidence=None,
            stability=None,
            n_folds=1,
            n_trades=10,
            metadata={"experiment_id": "exp_1"},
        )
        with self.subTest("writes readable parquet"):
            from tempfile import TemporaryDirectory

            with TemporaryDirectory() as tmp_dir:
                path = Path(tmp_dir) / "strategy_evaluations.parquet"
                write_strategy_evaluations_parquet(evaluations=[evaluation], output_path=path)
                self.assertTrue(path.exists())
                loaded = pd.read_parquet(path)
                self.assertEqual(int(len(loaded)), 1)


if __name__ == "__main__":
    unittest.main()
