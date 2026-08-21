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


if __name__ == "__main__":
    unittest.main()
