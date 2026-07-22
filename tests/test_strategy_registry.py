import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.strategies import (
    _DEFAULT_EVALUATED_MR_STRATEGY_IDS,
    _DEFAULT_EVALUATED_TF_STRATEGY_IDS,
    PhaseAwareStrategy,
    StrategySelector_Dynamic,
    instantiate_evaluated_strategy_dicts,
    run_backtests,
)
from src.strategy_registry import (
    DEFAULT_PHASEAWARE_POLICY_ID,
    EvaluationPolicy,
    EvaluationPolicyRegistry,
    StrategyCapabilities,
    StrategyDefinition,
    StrategyRegistry,
    get_default_policy_registry,
    get_default_strategy_registry,
    phaseaware_strategy_name,
    resolve_phaseaware_strategy_pair,
)


class TestStrategyRegistry(unittest.TestCase):
    def test_default_registry_queries(self):
        registry = get_default_strategy_registry()

        self.assertIn("TF4", registry.available())
        self.assertIn("MR42", registry.available())
        self.assertEqual(
            [definition.strategy_id for definition in registry.by_family("TrendFollowing")],
            ["TF1", "TF2", "TF3", "TF4", "TF5"],
        )
        self.assertEqual(
            [definition.strategy_id for definition in registry.supporting_surface("trend_vol")],
            registry.available(),
        )
        self.assertEqual(
            [definition.strategy_id for definition in registry.supporting_state("LVTF")],
            ["TF1", "TF2", "TF3", "TF4", "TF5"],
        )
        self.assertEqual(
            [definition.strategy_id for definition in registry.supporting_asset("fx")],
            registry.available(),
        )

    def test_policy_registry_resolves_phaseaware_default(self):
        policy_registry = get_default_policy_registry()
        policy = policy_registry.get(DEFAULT_PHASEAWARE_POLICY_ID)

        self.assertEqual(policy.policy_id, DEFAULT_PHASEAWARE_POLICY_ID)
        self.assertEqual(policy.strategies, ("TF4", "MR42"))
        self.assertEqual(
            resolve_phaseaware_strategy_pair(DEFAULT_PHASEAWARE_POLICY_ID),
            ("TF4", "MR42"),
        )
        self.assertEqual(
            phaseaware_strategy_name(DEFAULT_PHASEAWARE_POLICY_ID),
            "PhaseAware_TF4_MR42",
        )

    def test_duplicate_strategy_ids_fail_validation(self):
        duplicate = StrategyDefinition(
            strategy_id="X",
            display_name="X",
            family="TrendFollowing",
            implementation=list,
            capabilities=StrategyCapabilities(),
        )
        with self.assertRaises(ValueError):
            StrategyRegistry([duplicate, duplicate])

    def test_invalid_surface_reference_fails_validation(self):
        invalid = StrategyDefinition(
            strategy_id="X",
            display_name="X",
            family="TrendFollowing",
            implementation=list,
            capabilities=StrategyCapabilities(
                supported_surfaces=("does_not_exist",),
            ),
        )
        with self.assertRaises(ValueError):
            StrategyRegistry([invalid])

    def test_invalid_policy_reference_fails_validation(self):
        strategy_registry = StrategyRegistry(
            [
                StrategyDefinition(
                    strategy_id="TFX",
                    display_name="TFX",
                    family="TrendFollowing",
                    implementation=list,
                    capabilities=StrategyCapabilities(),
                )
            ]
        )
        with self.assertRaises(ValueError):
            EvaluationPolicyRegistry(
                [
                    EvaluationPolicy(
                        policy_id="bad",
                        display_name="Bad",
                        strategies=("MISSING",),
                    )
                ],
                strategy_registry=strategy_registry,
            )


class TestStrategyRegistryIntegration(unittest.TestCase):
    @staticmethod
    def _make_df(rows: int = 120) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=rows, freq="D")
        base = np.linspace(1.10, 1.25, rows)
        return pd.DataFrame(
            {
                "Open": base,
                "High": base + 0.01,
                "Low": base - 0.01,
                "Close": base + np.sin(np.linspace(0, 8, rows)) * 0.005,
                "Volume": np.full(rows, 1000.0),
                "rsi": np.linspace(35.0, 65.0, rows),
                "adx": np.linspace(15.0, 35.0, rows),
                "plus_di": np.linspace(20.0, 30.0, rows),
                "minus_di": np.linspace(30.0, 20.0, rows),
                "phase": (
                    ["LV_Trend", "HV_Trend", "LV_Ranging", "HV_Ranging"]
                    * (rows // 4)
                    + ["LV_Trend"] * (rows % 4)
                )[:rows],
                "atr": np.full(rows, 0.01),
                "stop_atr_mult": np.full(rows, 2.0),
            },
            index=idx,
        )

    def test_phaseaware_strategy_name_matches_policy(self):
        tf_strategy, mr_strategy = resolve_phaseaware_strategy_pair()
        strategy = PhaseAwareStrategy(tf_strategy, mr_strategy)
        self.assertEqual(strategy.name, phaseaware_strategy_name())

    def test_dynamic_selector_can_resolve_default_policy(self):
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
            evaluation_policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
            tau_enter=0.4,
            tau_exit=0.3,
            use_prob_margin=False,
            use_hysteresis=False,
            use_min_hold=False,
            min_hold_bars=0,
            use_max_hold=False,
        )
        _, _, _, selected = dynamic.generate_signals(df, "EURUSD", return_selected=True)
        self.assertTrue((selected == "MeanReversion").all())

    def test_legacy_standalone_strategy_sets_exclude_mr3(self):
        tf_strategies, mr_strategies = instantiate_evaluated_strategy_dicts()

        self.assertEqual(list(tf_strategies), list(_DEFAULT_EVALUATED_TF_STRATEGY_IDS))
        self.assertEqual(list(mr_strategies), list(_DEFAULT_EVALUATED_MR_STRATEGY_IDS))
        self.assertNotIn("MR3", mr_strategies)

    def test_run_backtests_accepts_evaluation_policy_id(self):
        results = run_backtests(
            self._make_df(),
            initial_capital=1000.0,
            evaluation_policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
        )
        self.assertNotIn("MR3", results)
        self.assertIn("TF4", results)
        self.assertIn("MR42", results)
        self.assertIn(phaseaware_strategy_name(DEFAULT_PHASEAWARE_POLICY_ID), results)


if __name__ == "__main__":
    unittest.main(verbosity=2)
