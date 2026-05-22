import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.models import StrategyPerformanceTracker
from src.repro import DEFAULT_EXPERIMENT_SEED, resolve_experiment_seed
from src.strategies import StrategySelector_Dynamic
from main import generate_walkforward_folds_by_pos


class _StaticSignalStrategy:
    def generate_signals(self, df: pd.DataFrame):
        z = pd.Series(np.zeros(len(df), dtype=float), index=df.index)
        return z, z.copy(), z.copy()


class _TieSelector:
    feature_cols = ["adx"]
    required_feature_cols = ["adx"]

    @staticmethod
    def predict_proba(_features_df: pd.DataFrame) -> dict[str, float]:
        return {
            "TrendFollowing": 0.5,
            "MeanReversion": 0.5,
            "PhaseAware": 0.0,
        }


class TestReproducibility(unittest.TestCase):
    def test_seed_resolution_precedence(self):
        prev = os.environ.get("EXPERIMENT_SEED")
        try:
            if "EXPERIMENT_SEED" in os.environ:
                del os.environ["EXPERIMENT_SEED"]
            self.assertEqual(
                resolve_experiment_seed(cli_seed=None),
                DEFAULT_EXPERIMENT_SEED,
            )

            os.environ["EXPERIMENT_SEED"] = "77"
            self.assertEqual(resolve_experiment_seed(cli_seed=None), 77)
            self.assertEqual(resolve_experiment_seed(cli_seed=13), 13)
        finally:
            if prev is None:
                os.environ.pop("EXPERIMENT_SEED", None)
            else:
                os.environ["EXPERIMENT_SEED"] = prev

    def test_fold_generation_is_deterministic(self):
        dates = pd.to_datetime(
            [
                "2021-01-01",
                "2020-01-01",
                "2020-07-01",
                "2021-07-01",
                "2022-01-01",
                "2022-07-01",
                "2023-01-01",
                "2023-07-01",
            ]
        )
        folds_a = generate_walkforward_folds_by_pos(
            pd.DatetimeIndex(dates),
            train_years=1,
            test_months=6,
            step_months=6,
        )
        folds_b = generate_walkforward_folds_by_pos(
            pd.DatetimeIndex(reversed(dates)),
            train_years=1,
            test_months=6,
            step_months=6,
        )
        self.assertEqual(folds_a, folds_b)

    def test_selector_tie_break_is_stable(self):
        idx = pd.date_range("2024-01-01", periods=12, freq="D")
        df = pd.DataFrame(
            {
                "adx": np.linspace(10.0, 20.0, len(idx)),
                "phase": ["LV_Trend"] * len(idx),
            },
            index=idx,
        )
        dynamic = StrategySelector_Dynamic(
            selector_trained={"EURUSD": _TieSelector()},
            tf_strategies={"TF4": _StaticSignalStrategy()},
            mr_strategies={"MR42": _StaticSignalStrategy()},
            default_tf="TF4",
            default_mr="MR42",
            tau_enter=0.4,
            tau_exit=0.3,
            use_prob_margin=False,
            use_hysteresis=False,
            use_min_hold=False,
            min_hold_bars=0,
            use_max_hold=False,
        )
        _, _, _, selected = dynamic.generate_signals(df, "EURUSD", return_selected=True)
        # Deterministic tie-break is alphabetical after equal probabilities.
        self.assertTrue((selected == "MeanReversion").all())

    def test_strategy_tracker_repeatability_and_aggregate_stability(self):
        idx = pd.date_range("2023-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {
                "phase": ["LV_Trend"] * len(idx),
                "adx": np.linspace(20.0, 25.0, len(idx)),
                "atr_pct": np.linspace(0.01, 0.02, len(idx)),
                "plus_di": np.linspace(15.0, 17.0, len(idx)),
                "minus_di": np.linspace(11.0, 13.0, len(idx)),
                "rsi": np.linspace(45.0, 55.0, len(idx)),
                "returns": np.linspace(-0.01, 0.01, len(idx)),
            },
            index=idx,
        )
        eq = pd.Series(np.linspace(1.0, 1.5, len(idx)), index=idx)
        tracker = StrategyPerformanceTracker(window_days=5)
        # Intentionally provide unstable dict order to ensure deterministic outcome.
        strategy_results = {
            "TF4": {"equity_curve": eq},
            "MR42": {"equity_curve": eq},
        }
        a = tracker.compute_strategy_returns(df, dict(reversed(strategy_results.items())))
        b = tracker.compute_strategy_returns(df, strategy_results)
        self.assertTrue(a.equals(b))
        self.assertAlmostEqual(float(a["best_return"].mean()), float(b["best_return"].mean()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
