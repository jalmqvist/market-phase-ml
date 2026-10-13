"""Tests for the opt-in dl_conditional_portfolio_timeline export."""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import main as mpml_main
from src.strategies import Backtester


def _fixture_df(n=12, start="2020-01-01"):
    idx = pd.date_range(start, periods=n, freq="D")
    close = pd.Series(1.10 + 0.002 * np.arange(n), index=idx)
    df = pd.DataFrame({
        "Open": close, "High": close + 0.001, "Low": close - 0.001, "Close": close,
        "phase": "Trend", "atr": 0.01, "stop_atr_mult": 2.0,
    }, index=idx)
    return df


def _run(df, signals):
    bt = Backtester(initial_capital=10_000.0, spread_pips=1.0, slippage_pips=0.5)
    return bt, bt.run(df, pd.Series(signals, index=df.index), "T")


SIGNALS = [0, 1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0]


def _rows(df, res, bt, fold=0, pair="EURUSD"):
    sel = pd.Series(["TF"] * len(df), index=df.index, dtype=object)
    return mpml_main._build_dl_conditional_portfolio_rows(
        pair_name=pair, fold_id=fold, df_test=df, selected_s=sel,
        dyn_res=res, initial_capital=bt.initial_capital,
    )


def test_flag_default_false_and_parser_accepts_flag():
    sig = inspect.signature(mpml_main.main)
    assert sig.parameters["export_dl_conditional_portfolio_timeline"].default is False
    src = Path(mpml_main.__file__).read_text()
    assert '"--export-dl-conditional-portfolio-timeline"' in src
    assert "default=False" in src.split('"--export-dl-conditional-portfolio-timeline"')[1][:200]


def test_values_match_backtester_accounting():
    df = _fixture_df()
    bt, res = _run(df, SIGNALS)
    out = pd.DataFrame(_rows(df, res, bt))
    eq = res["equity_curve"].values
    assert len(out) == len(df)
    np.testing.assert_allclose(out["portfolio_equity"], eq)
    assert out["portfolio_pnl"].iloc[0] == pytest.approx(eq[0] - 10_000.0)
    np.testing.assert_allclose(out["portfolio_pnl"].iloc[1:], np.diff(eq))
    assert out["portfolio_return"].iloc[1] == pytest.approx((eq[1] - eq[0]) / eq[0])
    # sum of per-bar PnL equals total realized PnL (incl. costs)
    assert out["portfolio_pnl"].sum() == pytest.approx(eq[-1] - 10_000.0)
    # executed position is end-of-bar position from backtester
    np.testing.assert_array_equal(out["executed_position"], res["execution_timeline"]["position"])


def test_export_does_not_mutate_backtest_result():
    df = _fixture_df()
    bt, res = _run(df, SIGNALS)
    before = res["equity_curve"].copy()
    _rows(df, res, bt)
    pd.testing.assert_series_equal(before, res["equity_curve"])
    _, res2 = _run(df, SIGNALS)
    pd.testing.assert_series_equal(res["equity_curve"], res2["equity_curve"])


def test_dl_available_matches_existing_definition_and_keeps_unavailable_bars():
    df = _fixture_df()
    bt, res = _run(df, SIGNALS)
    cols = [sorted(mpml_main.D1_FEATURE_COLS)[0]]
    df[cols[0]] = np.where(np.arange(len(df)) % 2 == 0, 0.5, np.nan)
    out = pd.DataFrame(_rows(df, res, bt))
    expected = df[mpml_main.get_dl_feature_columns(df)].notna().any(axis=1).values
    np.testing.assert_array_equal(out["dl_available"], expected)
    assert len(out) == len(df)
    assert (~out["dl_available"]).any()


def test_no_dl_columns_gives_all_false_but_rows_kept():
    df = _fixture_df()
    bt, res = _run(df, SIGNALS)
    out = pd.DataFrame(_rows(df, res, bt))
    assert len(out) == len(df) and not out["dl_available"].any()


def test_missing_equity_preserved_not_zero():
    df = _fixture_df()
    bt, res = _run(df, SIGNALS)
    res = dict(res)
    eq = res["equity_curve"].copy()
    eq.iloc[5] = np.nan
    res["equity_curve"] = eq
    out = pd.DataFrame(_rows(df, res, bt))
    assert np.isnan(out["portfolio_equity"].iloc[5])
    assert np.isnan(out["portfolio_pnl"].iloc[5]) and np.isnan(out["portfolio_pnl"].iloc[6])


def test_no_equity_returns_no_rows():
    df = _fixture_df()
    assert mpml_main._build_dl_conditional_portfolio_rows(
        pair_name="X", fold_id=0, df_test=df, selected_s=pd.Series(["a"] * len(df)),
        dyn_res={}, initial_capital=1.0) == []


def test_writer_schema_ordering_and_duplicate_timestamps_across_folds(tmp_path, monkeypatch):
    monkeypatch.setattr(mpml_main, "_CURRENT_RUN_OUTPUT_DIR", tmp_path)
    df = _fixture_df()
    bt, res = _run(df, SIGNALS)
    rows = (
        _rows(df, res, bt, fold=1, pair="USDJPY")
        + _rows(df, res, bt, fold=1, pair="EURUSD")
        + _rows(df, res, bt, fold=0, pair="EURUSD")  # same timestamps, different fold
    )
    path = mpml_main._write_dl_conditional_portfolio_timeline(rows)
    assert path == tmp_path / "dl_conditional_portfolio_timeline.csv"
    out = pd.read_csv(path)
    assert list(out.columns) == mpml_main.DL_CONDITIONAL_PORTFOLIO_TIMELINE_COLUMNS
    for c in ("timestamp", "pair", "fold", "selected_strategy", "dl_available", "portfolio_equity"):
        assert c in out.columns
    assert len(out) == 3 * len(df)
    assert not out.duplicated(["pair", "fold", "timestamp"]).any()
    keys = list(zip(out["pair"], out["fold"]))
    assert keys == sorted(keys)
    assert out["timestamp"].is_monotonic_increasing is False  # restarts per (pair, fold)
    first = out[(out.pair == "EURUSD") & (out.fold == 0)]["timestamp"].is_monotonic_increasing
    assert first


def test_writer_empty_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(mpml_main, "_CURRENT_RUN_OUTPUT_DIR", tmp_path)
    assert mpml_main._write_dl_conditional_portfolio_timeline([]) is None
    assert not (tmp_path / "dl_conditional_portfolio_timeline.csv").exists()


def test_default_path_has_no_export_hook():
    src = inspect.getsource(mpml_main.main)
    assert "if _wf_run_dynamic and export_dl_conditional_portfolio_timeline" in src
    assert "if export_dl_conditional_portfolio_timeline:" in src
