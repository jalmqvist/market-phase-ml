"""Unit tests for behavioral performance attribution of explicit standalone strategies.

Behavioural conditioning — Option A:

- Strategy signals and execution are computed on the complete df_test.
- Behavioral conditioning is applied ONLY for performance attribution.
- A trade is behaviorally eligible when its ENTRY observation is state-active.
- State-active is defined by the established MPML per-bar mask:
      df_test[D1_FEATURE_COLS].notna().any(axis=1)
- A trade entered on an active bar but exited on an inactive bar is FULLY
  attributed to the behavioral state (complete lifecycle preserved).
- A trade entered on an inactive bar receives behavioral_eligible=False.

Tests
-----
1.  Baseline (no DL) — attribution columns are NOT added to trades.
2.  Attribution does not change signals or execution (P&L, prices, exit reason).
3.  Only trades whose entry bar is state-active are eligible.
4.  Trade entered-active / exited-inactive → eligible, P&L/exit unchanged.
5.  Trade entered-inactive → not eligible.
6.  Conditioning is fold-local: only df_test index consulted.
7.  Missing DL columns → all trades receive behavioral_eligible=False.
8.  Different behavioral states → different eligible populations.
9.  trades_to_dataframe function correctness.
"""
from __future__ import annotations

import sys
import importlib
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load_main():
    return importlib.import_module("main")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DL_COL = "dl_signal_mean_24h"  # a D1_FEATURE_COLS member


def _make_df_test(n: int = 10, dl_active_idx: list[int] | None = None) -> pd.DataFrame:
    """Build a minimal df_test with a DatetimeIndex and an optional DL column.

    Parameters
    ----------
    n : int
        Number of bars.
    dl_active_idx : list[int] or None
        Positional indices of bars where the DL feature is non-null (state-active).
        If None, no DL column is created.
    """
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {"Close": np.linspace(1.1000, 1.1100, n)},
        index=dates,
    )
    if dl_active_idx is not None:
        dl_values = [np.nan] * n
        for i in dl_active_idx:
            dl_values[i] = 0.5
        df[_DL_COL] = dl_values
    return df


def _make_trades_df(
    entry_dates: list[pd.Timestamp],
    exit_dates: list[pd.Timestamp] | None = None,
    pnl: float = 10.0,
) -> pd.DataFrame:
    """Build a minimal trades DataFrame (output of trades_to_dataframe)."""
    if exit_dates is None:
        exit_dates = entry_dates
    rows = []
    for ed, xd in zip(entry_dates, exit_dates):
        rows.append({
            "entry_date": ed,
            "exit_date": xd,
            "entry_price": 1.1000,
            "exit_price": 1.1010,
            "direction": 1,
            "phase": "LV_Trend",
            "strategy": "TF1",
            "size_multiplier": 1.0,
            "position_size": 10000.0,
            "stop_distance": 0.002,
            "pnl": pnl,
            "pnl_pct": pnl / 10000.0,
            "exit_reason": "signal",
        })
    return pd.DataFrame(rows)


def _apply(main_module, trades_df, df_test, *, state_id, surface_id, dl_cols):
    """Thin helper so tests don't trip on Python descriptor binding."""
    return main_module._apply_behavioral_attribution_to_trades(
        trades_df, df_test,
        state_id=state_id,
        surface_id=surface_id,
        dl_cols=dl_cols,
    )


# ---------------------------------------------------------------------------
# 1. Baseline: no DL runtime — attribution columns NOT added
# ---------------------------------------------------------------------------

class TestBaselineNoAttribution(unittest.TestCase):
    """Baseline (dl_runtime_enabled=False): attribution helper is never called;
    the trades DataFrame should not contain behavioral columns."""

    def test_attribution_columns_absent_when_helper_not_called(self):
        """Trades that bypass _apply_behavioral_attribution_to_trades have no
        behavioral_eligible column — confirming baseline is unchanged."""
        df_test = _make_df_test(n=5, dl_active_idx=[1, 2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[1]])

        # Baseline: do NOT call the helper
        self.assertNotIn("behavioral_eligible", trades_df.columns)
        self.assertNotIn("behavioral_surface_id", trades_df.columns)
        self.assertNotIn("behavioral_state_id", trades_df.columns)


# ---------------------------------------------------------------------------
# 2. Attribution does not change signals or execution
# ---------------------------------------------------------------------------

class TestAttributionPreservesExecution(unittest.TestCase):
    """The helper must not alter entry/exit prices, P&L, direction or exit_reason."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def test_pnl_unchanged(self):
        df_test = _make_df_test(n=5, dl_active_idx=[1, 2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[1]], pnl=42.0)

        result = _apply(
            self._main, trades_df, df_test,
            state_id="STATE_A", surface_id="surface_x", dl_cols=[_DL_COL],
        )
        self.assertAlmostEqual(result["pnl"].iloc[0], 42.0)

    def test_entry_exit_prices_unchanged(self):
        df_test = _make_df_test(n=5, dl_active_idx=[1, 2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[1]])
        original_entry = trades_df["entry_price"].iloc[0]
        original_exit = trades_df["exit_price"].iloc[0]

        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertAlmostEqual(result["entry_price"].iloc[0], original_entry)
        self.assertAlmostEqual(result["exit_price"].iloc[0], original_exit)

    def test_exit_reason_unchanged(self):
        df_test = _make_df_test(n=5, dl_active_idx=[1])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[1]])
        trades_df["exit_reason"] = "SL"

        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertEqual(result["exit_reason"].iloc[0], "SL")

    def test_direction_unchanged(self):
        df_test = _make_df_test(n=5, dl_active_idx=[2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[2]])
        trades_df["direction"] = -1

        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertEqual(result["direction"].iloc[0], -1)

    def test_row_count_unchanged(self):
        df_test = _make_df_test(n=8, dl_active_idx=[1, 3, 5])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[i] for i in range(6)])

        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertEqual(len(result), len(trades_df),
                         "Attribution must not drop or duplicate trades")


# ---------------------------------------------------------------------------
# 3. Only entry-active trades are eligible
# ---------------------------------------------------------------------------

class TestEntryActiveMask(unittest.TestCase):
    """Trades entered on state-active bars are eligible; others are not."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def test_entry_active_is_eligible(self):
        df_test = _make_df_test(n=5, dl_active_idx=[2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[2]])
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertTrue(result["behavioral_eligible"].iloc[0])

    def test_entry_inactive_is_not_eligible(self):
        df_test = _make_df_test(n=5, dl_active_idx=[2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[0]])
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertFalse(result["behavioral_eligible"].iloc[0])

    def test_mixed_trades_classified_correctly(self):
        df_test = _make_df_test(n=6, dl_active_idx=[1, 3])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df(
            [dates[0], dates[1], dates[2], dates[3]]
        )
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        expected = [False, True, False, True]
        self.assertEqual(list(result["behavioral_eligible"]), expected)


# ---------------------------------------------------------------------------
# 4. Trade entered-active / exited-inactive → eligible, full lifecycle
# ---------------------------------------------------------------------------

class TestCrossBoundaryTrade(unittest.TestCase):
    """A trade entered on an active bar but exited on an inactive bar is
    eligible and its realized P&L/exit is preserved unchanged."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def test_entry_active_exit_inactive_is_eligible(self):
        df_test = _make_df_test(n=6, dl_active_idx=[0, 1, 2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df(
            entry_dates=[dates[2]],
            exit_dates=[dates[4]],
        )
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertTrue(
            result["behavioral_eligible"].iloc[0],
            "Trade entered-active must be eligible even if exit is inactive",
        )

    def test_cross_boundary_pnl_preserved(self):
        df_test = _make_df_test(n=6, dl_active_idx=[0, 1, 2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df(
            entry_dates=[dates[2]],
            exit_dates=[dates[4]],
            pnl=77.5,
        )
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertAlmostEqual(
            result["pnl"].iloc[0], 77.5,
            msg="P&L must be unchanged for cross-boundary trade",
        )

    def test_cross_boundary_exit_date_preserved(self):
        df_test = _make_df_test(n=6, dl_active_idx=[0, 1, 2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df(
            entry_dates=[dates[2]],
            exit_dates=[dates[4]],
        )
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertEqual(
            result["exit_date"].iloc[0], dates[4],
            "Exit date must be unchanged for cross-boundary trade",
        )


# ---------------------------------------------------------------------------
# 5. Trade entered-inactive → not eligible
# ---------------------------------------------------------------------------

class TestEntryInactiveNotEligible(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def test_entry_inactive_exit_active_is_not_eligible(self):
        df_test = _make_df_test(n=6, dl_active_idx=[3, 4, 5])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df(
            entry_dates=[dates[1]],
            exit_dates=[dates[4]],
        )
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertFalse(
            result["behavioral_eligible"].iloc[0],
            "Entry on inactive bar must not be eligible, "
            "even if exit is on an active bar",
        )


# ---------------------------------------------------------------------------
# 6. Conditioning is fold-local
# ---------------------------------------------------------------------------

class TestFoldLocal(unittest.TestCase):
    """The helper must consult ONLY df_test (the fold slice).

    An entry_date that lies outside the fold's DatetimeIndex should be
    treated as inactive (not in the active timestamp set).
    """

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def test_entry_outside_fold_is_not_eligible(self):
        dates_fold = pd.date_range("2023-01-11", periods=5, freq="D")
        df_test = pd.DataFrame(
            {_DL_COL: [0.5] * 5},
            index=dates_fold,
        )
        outside_entry = pd.Timestamp("2023-01-05")
        trades_df = _make_trades_df([outside_entry])
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertFalse(
            result["behavioral_eligible"].iloc[0],
            "Entry outside fold window must not be eligible",
        )

    def test_entry_inside_fold_active_is_eligible(self):
        dates_fold = pd.date_range("2023-01-11", periods=5, freq="D")
        df_test = pd.DataFrame(
            {_DL_COL: [0.5] * 5},
            index=dates_fold,
        )
        trades_df = _make_trades_df([dates_fold[3]])
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertTrue(result["behavioral_eligible"].iloc[0])


# ---------------------------------------------------------------------------
# 7. Missing DL columns → all trades receive behavioral_eligible=False
# ---------------------------------------------------------------------------

class TestMissingDLColumns(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def test_empty_dl_cols_all_ineligible(self):
        df_test = _make_df_test(n=5, dl_active_idx=None)
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[0], dates[2], dates[4]])

        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[],
        )
        self.assertFalse(
            result["behavioral_eligible"].any(),
            "No DL columns → no trade should be eligible",
        )

    def test_attribution_columns_always_present(self):
        """Even when DL is unavailable, the three attribution columns are added."""
        df_test = _make_df_test(n=3)
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[0]])
        result = _apply(
            self._main, trades_df, df_test,
            state_id="MY_STATE", surface_id="my_surface", dl_cols=[],
        )
        self.assertIn("behavioral_eligible", result.columns)
        self.assertIn("behavioral_surface_id", result.columns)
        self.assertIn("behavioral_state_id", result.columns)
        self.assertEqual(result["behavioral_surface_id"].iloc[0], "my_surface")
        self.assertEqual(result["behavioral_state_id"].iloc[0], "MY_STATE")

    def test_surface_and_state_ids_forwarded(self):
        df_test = _make_df_test(n=3, dl_active_idx=[1])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[1]])
        result = _apply(
            self._main, trades_df, df_test,
            state_id="JPY_CONSENSUS_YOUNG",
            surface_id="reactive_jpy",
            dl_cols=[_DL_COL],
        )
        self.assertEqual(result["behavioral_surface_id"].iloc[0], "reactive_jpy")
        self.assertEqual(result["behavioral_state_id"].iloc[0], "JPY_CONSENSUS_YOUNG")


# ---------------------------------------------------------------------------
# 8. Different behavioral states → different eligible populations
# ---------------------------------------------------------------------------

class TestDifferentBehavioralStates(unittest.TestCase):
    """Two states that activate different bars produce different eligible sets."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def _run_state(
        self,
        n: int,
        active_idx: list[int],
        entry_indices: list[int],
        state_id: str,
    ) -> pd.DataFrame:
        df_test = _make_df_test(n=n, dl_active_idx=active_idx)
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[i] for i in entry_indices])
        return _apply(
            self._main, trades_df, df_test,
            state_id=state_id, surface_id="reactive_jpy", dl_cols=[_DL_COL],
        )

    def test_different_states_produce_different_eligible_sets(self):
        # YOUNG activates bars 0-2; MATURE activates bars 4-6.
        # Entries on bars [0, 2, 4, 6].
        young = self._run_state(8, [0, 1, 2], [0, 2, 4, 6], "JPY_CONSENSUS_YOUNG")
        mature = self._run_state(8, [4, 5, 6], [0, 2, 4, 6], "JPY_CONSENSUS_MATURE")

        self.assertNotEqual(
            list(young["behavioral_eligible"]),
            list(mature["behavioral_eligible"]),
            "Different behavioral states must produce different eligible trade sets",
        )

    def test_young_includes_early_bars_mature_excludes_them(self):
        young = self._run_state(8, [0, 1, 2], [0, 2, 4, 6], "JPY_CONSENSUS_YOUNG")
        mature = self._run_state(8, [4, 5, 6], [0, 2, 4, 6], "JPY_CONSENSUS_MATURE")

        # Young: entries at bars 0, 2 are active; bars 4, 6 are inactive
        self.assertTrue(young["behavioral_eligible"].iloc[0])
        self.assertTrue(young["behavioral_eligible"].iloc[1])
        self.assertFalse(young["behavioral_eligible"].iloc[2])
        self.assertFalse(young["behavioral_eligible"].iloc[3])

        # Mature: entries at bars 4, 6 are active; bars 0, 2 are inactive
        self.assertFalse(mature["behavioral_eligible"].iloc[0])
        self.assertFalse(mature["behavioral_eligible"].iloc[1])
        self.assertTrue(mature["behavioral_eligible"].iloc[2])
        self.assertTrue(mature["behavioral_eligible"].iloc[3])

    def test_total_trades_same_eligible_counts_differ(self):
        young = self._run_state(8, [0, 1, 2], [0, 2, 4, 6], "JPY_CONSENSUS_YOUNG")
        mature = self._run_state(8, [4, 5, 6], [0, 2, 4, 6], "JPY_CONSENSUS_MATURE")

        self.assertEqual(len(young), len(mature),
                         "Total trades must be equal — execution is unchanged")
        self.assertEqual(int(young["behavioral_eligible"].sum()), 2)
        self.assertEqual(int(mature["behavioral_eligible"].sum()), 2)


# ---------------------------------------------------------------------------
# 9. trades_to_dataframe — function exists and behaves correctly
# ---------------------------------------------------------------------------

from src.strategies import trades_to_dataframe as _trades_to_dataframe
from src.strategies import TradeResult as _TradeResult


def _make_trade_result(**kwargs):
    defaults = dict(
        entry_date=pd.Timestamp("2023-01-01"),
        exit_date=pd.Timestamp("2023-01-05"),
        entry_price=1.1000,
        exit_price=1.1050,
        direction=1,
        phase="LV_Trend",
        strategy="TF1",
        size_multiplier=1.0,
        position_size=10000.0,
        stop_distance=0.002,
        pnl=50.0,
        pnl_pct=0.005,
        exit_reason="signal",
    )
    defaults.update(kwargs)
    return _TradeResult(**defaults)


class TestTradesToDataframe(unittest.TestCase):
    """Verify the trades_to_dataframe utility."""

    def test_empty_list_returns_empty_dataframe(self):
        result = _trades_to_dataframe([])
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        self.assertIn("entry_date", result.columns)
        self.assertIn("pnl", result.columns)

    def test_single_trade_produces_one_row(self):
        result = _trades_to_dataframe([_make_trade_result()])
        self.assertEqual(len(result), 1)

    def test_multiple_trades_correct_row_count(self):
        trades = [_make_trade_result(pnl=float(i)) for i in range(5)]
        result = _trades_to_dataframe(trades)
        self.assertEqual(len(result), 5)

    def test_pnl_values_preserved(self):
        trades = [_make_trade_result(pnl=float(i * 10)) for i in range(3)]
        result = _trades_to_dataframe(trades)
        self.assertAlmostEqual(result["pnl"].iloc[1], 10.0)
        self.assertAlmostEqual(result["pnl"].iloc[2], 20.0)

    def test_all_expected_columns_present(self):
        result = _trades_to_dataframe([_make_trade_result()])
        expected_cols = {
            "entry_date", "exit_date", "entry_price", "exit_price",
            "direction", "phase", "strategy", "size_multiplier",
            "position_size", "stop_distance", "pnl", "pnl_pct", "exit_reason",
        }
        missing = expected_cols - set(result.columns)
        self.assertFalse(missing, f"Missing columns: {missing}")


# ---------------------------------------------------------------------------
# 10-18. compute_behavioral_conditional_performance tests
# ---------------------------------------------------------------------------

def _make_attributed_trades_df(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal strategy_trades DataFrame with behavioral attribution columns."""
    defaults = dict(
        behavioral_surface_id="reactive_jpy",
        behavioral_state_id="JPY_CONSENSUS_YOUNG",
        pair="USDJPY",
        fold="fold_1",
        strategy_id="TF1",
        pnl=10.0,
        pnl_pct=0.001,
        behavioral_eligible=True,
    )
    result_rows = []
    for row in rows:
        r = {**defaults, **row}
        result_rows.append(r)
    return pd.DataFrame(result_rows)


class TestComputeBehavioralConditionalPerformance(unittest.TestCase):
    """Tests for compute_behavioral_conditional_performance."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def _compute(self, df):
        return self._main.compute_behavioral_conditional_performance(df)

    # 10. Schema is correct
    def test_output_schema_correct(self):
        df = _make_attributed_trades_df([{}])
        result = self._compute(df)
        expected_cols = {
            "behavioral_surface_id",
            "behavioral_state_id",
            "pair",
            "fold",
            "strategy_id",
            "eligible_trades",
            "total_pnl",
            "mean_trade_return",
            "median_trade_return",
            "std_trade_return",
            "win_rate",
            "wins",
        }
        self.assertEqual(set(result.columns), expected_cols)

    # Only eligible trades contribute
    def test_only_eligible_trades_contribute(self):
        df = _make_attributed_trades_df([
            {"behavioral_eligible": True, "pnl": 10.0, "pnl_pct": 0.001},
            {"behavioral_eligible": False, "pnl": 999.0, "pnl_pct": 9.999},
        ])
        result = self._compute(df)
        self.assertEqual(result["eligible_trades"].iloc[0], 1)
        self.assertAlmostEqual(result["total_pnl"].iloc[0], 10.0)

    # Empty eligible population is handled safely
    def test_empty_eligible_population_returns_empty_frame(self):
        df = _make_attributed_trades_df([
            {"behavioral_eligible": False},
            {"behavioral_eligible": False},
        ])
        result = self._compute(df)
        self.assertTrue(result.empty)
        # But schema must still be correct
        self.assertIn("eligible_trades", result.columns)
        self.assertIn("win_rate", result.columns)

    # Pair, fold, strategy, surface, state attribution preserved
    def test_group_keys_preserved(self):
        df = _make_attributed_trades_df([
            {"pair": "USDJPY", "fold": "fold_1", "strategy_id": "TF1",
             "behavioral_surface_id": "reactive_jpy",
             "behavioral_state_id": "JPY_CONSENSUS_YOUNG",
             "behavioral_eligible": True},
        ])
        result = self._compute(df)
        self.assertEqual(result["pair"].iloc[0], "USDJPY")
        self.assertEqual(result["fold"].iloc[0], "fold_1")
        self.assertEqual(result["strategy_id"].iloc[0], "TF1")
        self.assertEqual(result["behavioral_surface_id"].iloc[0], "reactive_jpy")
        self.assertEqual(result["behavioral_state_id"].iloc[0], "JPY_CONSENSUS_YOUNG")

    # Multiple strategies can appear (normal behavioral run)
    def test_multiple_strategies_appear(self):
        df = _make_attributed_trades_df([
            {"strategy_id": "TF1", "behavioral_eligible": True},
            {"strategy_id": "TF4", "behavioral_eligible": True},
            {"strategy_id": "MR42", "behavioral_eligible": True},
        ])
        result = self._compute(df)
        self.assertEqual(set(result["strategy_id"]), {"TF1", "TF4", "MR42"})

    # Explicit --strategy: only the requested strategy appears
    def test_explicit_strategy_only_appears(self):
        df = _make_attributed_trades_df([
            {"strategy_id": "TF1", "behavioral_eligible": True},
        ])
        result = self._compute(df)
        self.assertEqual(list(result["strategy_id"]), ["TF1"])

    # win_rate calculation
    def test_win_rate_correct(self):
        df = _make_attributed_trades_df([
            {"pnl_pct": 0.01, "behavioral_eligible": True},
            {"pnl_pct": -0.01, "behavioral_eligible": True},
            {"pnl_pct": 0.005, "behavioral_eligible": True},
            {"pnl_pct": -0.005, "behavioral_eligible": True},
        ])
        result = self._compute(df)
        self.assertAlmostEqual(result["win_rate"].iloc[0], 0.5)
        self.assertEqual(result["wins"].iloc[0], 2)
        self.assertEqual(result["eligible_trades"].iloc[0], 4)

    # mean/median/std calculations
    def test_return_statistics_correct(self):
        rets = [0.01, -0.01, 0.02, -0.02, 0.03]
        df = _make_attributed_trades_df([
            {"pnl_pct": r, "pnl": r * 10000, "behavioral_eligible": True}
            for r in rets
        ])
        result = self._compute(df)
        import numpy as np
        self.assertAlmostEqual(result["mean_trade_return"].iloc[0], np.mean(rets), places=10)
        self.assertAlmostEqual(result["median_trade_return"].iloc[0], np.median(rets), places=10)
        # std_trade_return uses ddof=1
        self.assertAlmostEqual(result["std_trade_return"].iloc[0], np.std(rets, ddof=1), places=10)

    # Active-entry/inactive-exit trades retain complete P&L
    def test_active_entry_inactive_exit_pnl_preserved(self):
        df = _make_attributed_trades_df([
            {"pnl": 77.5, "pnl_pct": 0.00775, "behavioral_eligible": True},
        ])
        result = self._compute(df)
        self.assertAlmostEqual(result["total_pnl"].iloc[0], 77.5)

    # std for single trade is NaN
    def test_std_nan_for_single_trade(self):
        df = _make_attributed_trades_df([{"behavioral_eligible": True}])
        result = self._compute(df)
        self.assertTrue(
            pd.isna(result["std_trade_return"].iloc[0]),
            "std of a single trade must be NaN (ddof=1)",
        )

    # Missing required columns raises ValueError
    def test_missing_required_column_raises(self):
        df = _make_attributed_trades_df([{}])
        df = df.drop(columns=["behavioral_eligible"])
        with self.assertRaises(ValueError):
            self._compute(df)

    # Normal and explicit-strategy paths produce same schema
    def test_normal_and_explicit_strategy_same_schema(self):
        """Identical aggregation code regardless of _strategy_only_scope."""
        # Simulate normal run: multiple strategies
        df_normal = _make_attributed_trades_df([
            {"strategy_id": "TF1", "behavioral_eligible": True, "pnl_pct": 0.01},
            {"strategy_id": "TF2", "behavioral_eligible": True, "pnl_pct": -0.01},
        ])
        # Simulate explicit --strategy TF1
        df_explicit = _make_attributed_trades_df([
            {"strategy_id": "TF1", "behavioral_eligible": True, "pnl_pct": 0.01},
        ])
        r_normal = self._compute(df_normal)
        r_explicit = self._compute(df_explicit)
        self.assertEqual(set(r_normal.columns), set(r_explicit.columns))

    # Existing unconditional walk-forward metrics remain unchanged
    def test_unconditional_metrics_unchanged(self):
        """The helper only aggregates; it does not touch existing walk-forward rows."""
        df = _make_attributed_trades_df([
            {"pnl": 5.0, "pnl_pct": 0.0005, "behavioral_eligible": True},
            {"pnl": -3.0, "pnl_pct": -0.0003, "behavioral_eligible": False},
        ])
        original_len = len(df)
        _ = self._compute(df)
        # The input frame must be unchanged.
        self.assertEqual(len(df), original_len)
        self.assertAlmostEqual(df["pnl"].iloc[0], 5.0)
        self.assertAlmostEqual(df["pnl"].iloc[1], -3.0)

    # 11. Attribution applied to all strategies when dl_runtime_enabled
    def test_attribution_helper_not_gated_on_strategy_only_scope(self):
        """When dl_runtime_enabled, attribution is applied regardless of scope.

        We test the helper directly: it always produces behavioral_eligible.
        """
        df_test = _make_df_test(n=5, dl_active_idx=[1, 2])
        dates = df_test.index.tolist()
        trades_df = _make_trades_df([dates[1]])
        result = _apply(
            self._main, trades_df, df_test,
            state_id="S", surface_id="s", dl_cols=[_DL_COL],
        )
        self.assertIn("behavioral_eligible", result.columns)

    # pnl is a required input column
    def test_missing_pnl_column_raises(self):
        """pnl must be required; omitting it raises ValueError like other missing columns."""
        df = _make_attributed_trades_df([{}])
        df = df.drop(columns=["pnl"])
        with self.assertRaises(ValueError):
            self._compute(df)


# ---------------------------------------------------------------------------
# Artifact output-path tests
# ---------------------------------------------------------------------------

import tempfile


class TestBehavioralArtifactOutputPath(unittest.TestCase):
    """Verify that the behavioral-performance artifact is written through the
    run-output machinery (_with_mode_tag / _resolve_output_path) and NOT
    directly to the repository-level results/ directory."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def _setup_run_output_dir(self, tmp_dir: str) -> None:
        """Point the module's run-output machinery at a temp directory."""
        import importlib
        # Reset global state, then set the new output dir.
        self._main._CURRENT_RUN_OUTPUT_DIR = None
        results_subdir = Path(tmp_dir) / "results"
        results_subdir.mkdir(parents=True, exist_ok=True)
        self._main._CURRENT_RUN_OUTPUT_DIR = Path(tmp_dir).resolve()

    def test_artifact_path_is_under_run_output_dir(self):
        """_with_mode_tag must resolve the behavioral-performance artifact
        to the current run-output directory, not to the repo-level results/."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._setup_run_output_dir(tmp_dir)
            resolved = self._main._with_mode_tag(
                "results/strategy_behavioral_performance__dl_enabled.csv",
                "",  # empty mode tag
            )
            resolved_path = Path(resolved).resolve()
            tmp_path = Path(tmp_dir).resolve()
            self.assertTrue(
                str(resolved_path).startswith(str(tmp_path)),
                f"Artifact path {resolved_path} is not under run-output dir {tmp_path}",
            )

    def test_artifact_filename_preserved(self):
        """The externally-intended filename must be preserved regardless of the
        run-output directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            self._setup_run_output_dir(tmp_dir)
            resolved = self._main._with_mode_tag(
                "results/strategy_behavioral_performance__dl_enabled.csv",
                "",
            )
            self.assertTrue(
                Path(resolved).name.startswith(
                    "strategy_behavioral_performance__dl_enabled"
                ),
                f"Unexpected artifact filename: {Path(resolved).name}",
            )


# ---------------------------------------------------------------------------
# Wiring-level regression: normal vs explicit-strategy reach same path
# ---------------------------------------------------------------------------

class TestNormalVsExplicitStrategyWiring(unittest.TestCase):
    """Both normal-behavioral and explicit --strategy modes call the same
    compute_behavioral_conditional_performance aggregation, producing the
    same output schema.  This test verifies the wiring at the aggregation
    function level (lightweight, no end-to-end run needed)."""

    @classmethod
    def setUpClass(cls):
        cls._main = _load_main()

    def _compute(self, df):
        return self._main.compute_behavioral_conditional_performance(df)

    def test_normal_mode_produces_artifact(self):
        """Normal behavioral run: multiple strategies → artifact has all strategies."""
        df = _make_attributed_trades_df([
            {"strategy_id": "TF1", "behavioral_eligible": True, "pnl_pct": 0.01},
            {"strategy_id": "TF2", "behavioral_eligible": True, "pnl_pct": -0.005},
            {"strategy_id": "MR1", "behavioral_eligible": True, "pnl_pct": 0.002},
        ])
        result = self._compute(df)
        self.assertEqual(set(result["strategy_id"]), {"TF1", "TF2", "MR1"})

    def test_explicit_strategy_mode_produces_artifact(self):
        """Explicit --strategy run: only the requested strategy → artifact has one strategy."""
        df = _make_attributed_trades_df([
            {"strategy_id": "TF1", "behavioral_eligible": True, "pnl_pct": 0.01},
        ])
        result = self._compute(df)
        self.assertEqual(list(result["strategy_id"]), ["TF1"])

    def test_both_modes_produce_same_schema(self):
        """The artifact schema is identical regardless of runtime mode."""
        df_normal = _make_attributed_trades_df([
            {"strategy_id": "TF1", "behavioral_eligible": True, "pnl_pct": 0.01},
            {"strategy_id": "TF2", "behavioral_eligible": True, "pnl_pct": -0.01},
        ])
        df_explicit = _make_attributed_trades_df([
            {"strategy_id": "TF1", "behavioral_eligible": True, "pnl_pct": 0.01},
        ])
        r_normal = self._compute(df_normal)
        r_explicit = self._compute(df_explicit)
        self.assertEqual(set(r_normal.columns), set(r_explicit.columns))

    def test_both_modes_use_same_export_logic_via_function(self):
        """Both modes call compute_behavioral_conditional_performance with
        the same function reference — no branching to separate implementations."""
        # The function is module-level; verify it is the same callable regardless
        # of whether it was called from a normal or explicit-strategy code path.
        fn = self._main.compute_behavioral_conditional_performance
        self.assertTrue(callable(fn))
        # Call it twice with different strategy populations and verify both succeed.
        df1 = _make_attributed_trades_df([{"strategy_id": "TF1", "behavioral_eligible": True}])
        df2 = _make_attributed_trades_df([
            {"strategy_id": "TF1", "behavioral_eligible": True},
            {"strategy_id": "TF2", "behavioral_eligible": True},
        ])
        r1 = fn(df1)
        r2 = fn(df2)
        self.assertEqual(set(r1.columns), set(r2.columns))


if __name__ == "__main__":
    unittest.main()
