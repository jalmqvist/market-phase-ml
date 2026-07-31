from __future__ import annotations

import hashlib
import json
import sys
import unittest
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation import (  # noqa: E402
    EVALUATION_SCHEMA_VERSION,
    StrategyEvaluation,
    build_strategy_evaluations,
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
        payload = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "surface_id": "trend_vol",
            "surface_version": "1.0.0",
            "state_id": "LVTF",
            "strategy_id": "PhaseAware_policy_v1",
            "experiment_id": "exp_abc",
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        expected = f"eval_{digest[:24]}"
        self.assertEqual(first, expected)

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

    def test_strategy_evaluation_from_record(self):
        record = {
            "evaluation_id": "eval_1",
            "surface_id": "trend_vol",
            "surface_version": "1.0.0",
            "state_id": "LVTF",
            "strategy_id": "PhaseAware",
            "expected_return": 1.2,
            "expected_sharpe": 0.5,
            "expected_drawdown": -3.0,
            "win_rate": None,
            "confidence": 0.4,
            "stability": None,
            "n_folds": 12,
            "n_trades": 120,
            "metadata": "{\"experiment_id\":\"exp_1\",\"source\":\"walkforward\"}",
        }
        evaluation = StrategyEvaluation.from_record(record)
        self.assertEqual(evaluation.evaluation_id, "eval_1")
        self.assertEqual(evaluation.metadata["experiment_id"], "exp_1")
        self.assertEqual(evaluation.confidence, 0.4)
        self.assertIsNone(evaluation.win_rate)

    def test_build_strategy_evaluations_from_specs(self):
        wf_df = pd.DataFrame(
            {
                "Pair": ["EURUSD", "GBPUSD"],
                "Baseline Return (%)": [1.0, 3.0],
                "Baseline Sharpe": [0.2, 0.4],
                "Baseline Max DD (%)": [-2.0, -4.0],
                "Baseline Trades": [10, 20],
                "Dynamic Return (%)": [2.0, 4.0],
                "Dynamic Sharpe": [0.3, 0.5],
                "Dynamic Max DD (%)": [-3.0, -5.0],
                "Dynamic Trades": [30, 40],
                "Confident Bars (%)": [50.0, 70.0],
            }
        )
        strategy_specs = [
            {
                "strategy_id": "PhaseAware",
                "expected_return_col": "Baseline Return (%)",
                "expected_sharpe_col": "Baseline Sharpe",
                "expected_drawdown_col": "Baseline Max DD (%)",
                "n_trades_col": "Baseline Trades",
                "confidence_col": None,
                "strategy_role": "baseline",
            },
            {
                "strategy_id": "StrategySelector_Dynamic_WF",
                "expected_return_col": "Dynamic Return (%)",
                "expected_sharpe_col": "Dynamic Sharpe",
                "expected_drawdown_col": "Dynamic Max DD (%)",
                "n_trades_col": "Dynamic Trades",
                "confidence_col": "Confident Bars (%)",
                "strategy_role": "dynamic_selector",
            },
        ]
        evaluations = build_strategy_evaluations(
            wf_df=wf_df,
            surface_id="trend_vol",
            surface_version="1.0.0",
            state_id="LVTF",
            experiment_id="exp_1",
            mode_tag="_dl",
            strategy_specs=strategy_specs,
        )
        self.assertEqual(len(evaluations), 2)
        self.assertEqual(evaluations[0].n_folds, 2)
        self.assertEqual(evaluations[0].metadata["pair_count"], 2)
        self.assertEqual(evaluations[1].confidence, 0.6)


if __name__ == "__main__":
    unittest.main()
