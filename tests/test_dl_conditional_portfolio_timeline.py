"""Tests for the opt-in dl_conditional_portfolio_timeline export."""
from __future__ import annotations

import json

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
    return bt, bt.run(
        df, pd.Series(signals, index=df.index), "T", include_mtm=True
    )


SIGNALS = [0, 1, 1, 1, 0, 0, -1, -1, 0, 0, 0, 0]
EXPECTED_COLUMNS = [
    "timestamp",
    "pair",
    "fold",
    "selected_strategy",
    "dl_available",
    "executed_position",
    "portfolio_equity",
    "portfolio_pnl",
    "portfolio_return",
    "portfolio_equity_mtm",
    "portfolio_pnl_mtm",
    "portfolio_return_mtm",
]


def _rows(df, res, bt, fold=0, pair="EURUSD"):
    sel = pd.Series(["TF"] * len(df), index=df.index, dtype=object)
    return mpml_main._build_dl_conditional_portfolio_rows(
        pair_name=pair, fold_id=fold, df_test=df, selected_s=sel,
        dyn_res=res, initial_capital=bt.initial_capital,
    )


def test_flag_default_false():
    assert mpml_main.main.__defaults__ is None
    assert mpml_main.main.__kwdefaults__[
        "export_dl_conditional_portfolio_timeline"
    ] is False


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


def test_forced_end_of_data_liquidation_flattens_exported_position():
    df = _fixture_df(n=4)
    signals = pd.Series([1, 1, 1, 1], index=df.index)
    bt = Backtester(initial_capital=10_000.0)
    result = bt.run(df, signals, "T", include_mtm=True)

    positions = result["execution_timeline"]["position"].to_numpy()
    assert np.any(positions[:-1] != 0)
    assert result["trades"][-1].exit_reason == "end_of_data"
    assert positions[-1] == 0
    assert result["equity_curve"].iloc[-1] == pytest.approx(
        result["equity_curve_mtm"].iloc[-1], rel=1e-12, abs=1e-9
    )
    assert result["execution_timeline"]["equity"].iloc[-1] == pytest.approx(
        result["equity_curve"].iloc[-1], rel=1e-12, abs=1e-9
    )
    exported = pd.DataFrame(_rows(df, result, bt))
    assert exported["executed_position"].iloc[-1] == 0
    assert exported["portfolio_equity"].iloc[-1] == pytest.approx(
        exported["portfolio_equity_mtm"].iloc[-1], rel=1e-12, abs=1e-9
    )


def test_flat_portfolio_mtm_and_timeline_returns_reconcile():
    df = _fixture_df(n=4)
    bt = Backtester(initial_capital=10_000.0)
    result = bt.run(
        df, pd.Series(0, index=df.index), "T", include_mtm=True
    )
    np.testing.assert_allclose(
        result["equity_curve_mtm"], result["equity_curve"], rtol=1e-12, atol=1e-9
    )
    out = pd.DataFrame(_rows(df, result, bt))
    for equity_col, pnl_col, return_col in (
        ("portfolio_equity", "portfolio_pnl", "portfolio_return"),
        ("portfolio_equity_mtm", "portfolio_pnl_mtm", "portfolio_return_mtm"),
    ):
        previous = out[equity_col].shift(1)
        previous.iloc[0] = bt.initial_capital
        expected_pnl = out[equity_col] - previous
        np.testing.assert_allclose(out[pnl_col], expected_pnl, rtol=1e-12, atol=1e-9)
        expected_return = expected_pnl / previous.where(previous != 0)
        np.testing.assert_allclose(
            out[return_col], expected_return, rtol=1e-12, atol=1e-12
        )


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
    df[sorted(mpml_main.D1_FEATURE_COLS)[0]] = np.nan
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
    with pytest.raises(ValueError, match="requires dynamic backtest results"):
        mpml_main._build_dl_conditional_portfolio_rows(
            pair_name="X", fold_id=0, df_test=df,
            selected_s=pd.Series(["a"] * len(df), index=df.index),
            dyn_res={}, initial_capital=1.0)


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


def test_writer_rejects_duplicate_pair_fold_timestamp_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(mpml_main, "_CURRENT_RUN_OUTPUT_DIR", tmp_path)
    df = _fixture_df()
    bt, res = _run(df, SIGNALS)
    rows = _rows(df, res, bt)
    with pytest.raises(ValueError, match="duplicate pair/fold/timestamp"):
        mpml_main._write_dl_conditional_portfolio_timeline(rows + rows)


def test_cli_flag_controls_export_and_manifest(tmp_path, monkeypatch):
    df = _fixture_df()
    signals = pd.Series(SIGNALS, index=df.index)
    selected = pd.Series(["TF"] * len(df), index=df.index)
    observed_flags = []

    def lightweight_main(*, output_dir, export_dl_conditional_portfolio_timeline, **kwargs):
        observed_flags.append(export_dl_conditional_portfolio_timeline)
        output_dir.mkdir(parents=True)
        monkeypatch.setattr(mpml_main, "_CURRENT_RUN_OUTPUT_DIR", output_dir)
        manifest = {}
        if export_dl_conditional_portfolio_timeline:
            result, rows = mpml_main._run_dynamic_selector_backtest(
                backtester=Backtester(),
                df_test=df,
                signals=signals,
                strategy_name="Dynamic",
                sl_pct_series=pd.Series(0.0, index=df.index),
                tp_pct_series=pd.Series(0.0, index=df.index),
                export_timeline=True,
                pair_name="EURUSD",
                fold_id=0,
                selected_s=selected,
            )
            assert "equity_curve_mtm" in result
            path = mpml_main._write_dl_conditional_portfolio_timeline(rows)
            assert path is not None
            manifest["dl_conditional_portfolio_timeline"] = (
                mpml_main._dl_conditional_portfolio_timeline_manifest(
                    rows, written=True
                )
            )
        (output_dir / "experiment_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    monkeypatch.setattr(mpml_main, "main", lightweight_main)
    disabled_dir = tmp_path / "default"
    mpml_main._run_cli(["--output-dir", str(disabled_dir)])
    assert observed_flags == [False]
    assert not (disabled_dir / "dl_conditional_portfolio_timeline.csv").exists()
    assert "dl_conditional_portfolio_timeline" not in json.loads(
        (disabled_dir / "experiment_manifest.json").read_text()
    )

    enabled_dir = tmp_path / "enabled"
    mpml_main._run_cli([
        "--output-dir", str(enabled_dir),
        "--export-dl-conditional-portfolio-timeline",
    ])
    assert observed_flags == [False, True]
    written = pd.read_csv(enabled_dir / "dl_conditional_portfolio_timeline.csv")
    assert len(written) == len(df)
    assert list(written.columns) == mpml_main.DL_CONDITIONAL_PORTFOLIO_TIMELINE_COLUMNS
    manifest_entry = json.loads(
        (enabled_dir / "experiment_manifest.json").read_text()
    )["dl_conditional_portfolio_timeline"]
    assert manifest_entry["filename"] == "dl_conditional_portfolio_timeline.csv"
    assert manifest_entry["schema_version"] == "1.1.0"
    assert "realized capital plus unrealized PnL" in manifest_entry["mtm_definition"]
    assert manifest_entry["rows"] == len(df) and manifest_entry["written"] is True


@pytest.mark.parametrize("use_atr_sizing", [False, True])
def test_mtm_is_opt_in_and_preserves_all_legacy_results(use_atr_sizing):
    df = _fixture_df()
    signals = pd.Series(SIGNALS, index=df.index)
    bt = Backtester(use_atr_sizing=use_atr_sizing)
    realized = bt.run(df, signals, "T")
    marked = bt.run(df, signals, "T", include_mtm=True)

    assert "equity_curve_mtm" not in realized
    assert set(marked) == set(realized) | {"equity_curve_mtm"}
    for key in realized:
        left, right = realized[key], marked[key]
        if isinstance(left, pd.Series):
            pd.testing.assert_series_equal(left, right)
        elif isinstance(left, pd.DataFrame):
            pd.testing.assert_frame_equal(left, right)
        elif key == "trades":
            assert left == right
        else:
            np.testing.assert_equal(left, right)


def test_reversal_marks_only_new_position_exit_cost():
    df = _fixture_df(n=4)
    df["Close"] = [1.10, 1.10, 1.12, 1.13]
    df["Open"] = df["Close"]
    df["High"] = df["Close"] + 0.001
    df["Low"] = df["Close"] - 0.001
    bt = Backtester(
        initial_capital=10_000.0, spread_pips=2.0, slippage_pips=1.0
    )
    result = bt.run(
        df, pd.Series([1, -1, -1, -1], index=df.index), "T", include_mtm=True
    )

    assert result["execution_timeline"]["position"].iloc[1] == 1
    assert result["execution_timeline"]["position"].iloc[2] == -1
    assert result["trades"][0].exit_reason == "signal"
    assert result["equity_curve_mtm"].iloc[2] == pytest.approx(
        result["equity_curve"].iloc[2]
        - result["equity_curve"].iloc[2] * bt._calculate_trade_cost(1.12),
        rel=1e-12,
        abs=1e-9,
    )


@pytest.mark.parametrize("direction,valuation", [
    (1, 1.12),
    (1, 1.08),
    (-1, 1.12),
    (-1, 1.08),
])
@pytest.mark.parametrize("use_atr_sizing", [False, True])
def test_open_position_mtm_matches_mode_costs_and_direction(
    direction, valuation, use_atr_sizing
):
    df = _fixture_df(n=4)
    df["Close"] = [1.10, 1.10, valuation, 1.11]
    df["Open"] = df["Close"]
    df["High"] = df["Close"] + 0.001
    df["Low"] = df["Close"] - 0.001
    bt = Backtester(
        initial_capital=10_000.0,
        spread_pips=2.0,
        slippage_pips=1.0,
        commission_per_trade=4.0,
        use_atr_sizing=use_atr_sizing,
        risk_pct=0.02,
    )
    result = bt.run(
        df, pd.Series([direction] * len(df), index=df.index), "T",
        include_mtm=True,
    )

    entry_cost = bt._calculate_trade_cost(1.10)
    exit_cost = bt._calculate_trade_cost(valuation)
    after_entry_cost = bt.initial_capital * (1 - entry_cost)
    if use_atr_sizing:
        size = after_entry_cost * bt.risk_pct / (2.0 * 0.01)
        open_pnl = (
            size * direction * (valuation - 1.10)
            - size * 1.10 * exit_cost
        )
    else:
        open_pnl = after_entry_cost * (
            direction * (valuation - 1.10) / 1.10 - exit_cost
        )

    assert result["equity_curve"].iloc[1] == pytest.approx(after_entry_cost)
    assert result["equity_curve_mtm"].iloc[1] < result["equity_curve"].iloc[1]
    assert result["equity_curve"].iloc[2] == result["equity_curve"].iloc[1]
    assert result["equity_curve_mtm"].iloc[2] == pytest.approx(
        after_entry_cost + open_pnl
    )
    assert result["equity_curve_mtm"].iloc[2] != pytest.approx(
        result["equity_curve"].iloc[2]
    )
    assert result["equity_curve_mtm"].iloc[-1] == result["equity_curve"].iloc[-1]


@pytest.mark.parametrize("exit_kind", ["signal", "SL", "TP"])
@pytest.mark.parametrize("use_atr_sizing", [False, True])
def test_flat_and_exit_bars_reconcile_realized_and_mtm(exit_kind, use_atr_sizing):
    df = _fixture_df(n=5)
    df["Close"] = 1.10
    df["Open"] = df["Close"]
    df["High"] = df["Close"] + 0.001
    df["Low"] = df["Close"] - 0.001
    signals = [1, 0, 0, 0, 0]
    sl = pd.Series(0.0, index=df.index)
    tp = pd.Series(0.0, index=df.index)
    if exit_kind == "signal":
        df.loc[df.index[2], "Close"] = 1.12
    elif exit_kind == "SL":
        sl.iloc[1] = 0.005
        df.loc[df.index[2], "Low"] = 1.08
    else:
        tp.iloc[1] = 0.005
        df.loc[df.index[2], "High"] = 1.12

    bt = Backtester(use_atr_sizing=use_atr_sizing)
    signals = pd.Series(signals, index=df.index)
    result = bt.run(
        df, pd.Series(signals, index=df.index), "T", sl, tp, include_mtm=True
    )
    realized = bt.run(df, signals, "T", sl, tp)
    flat = result["execution_timeline"]["position"].to_numpy() == 0
    np.testing.assert_allclose(
        result["equity_curve_mtm"].to_numpy()[flat],
        result["equity_curve"].to_numpy()[flat],
    )
    assert result["trades"][0].exit_reason == (
        "signal" if exit_kind == "signal" else exit_kind
    )
    assert result["equity_curve"].iloc[2] == pytest.approx(
        realized["equity_curve"].iloc[2]
    )
    np.testing.assert_allclose(
        result["equity_curve_mtm"].iloc[2:],
        result["equity_curve"].iloc[2:],
    )
    assert result["trades"] == realized["trades"]
    assert result["equity_curve_mtm"].iloc[-1] == result["equity_curve"].iloc[-1]


def test_invalid_open_position_valuation_is_nan_without_fabricating_price():
    df = _fixture_df(n=5)
    df.loc[df.index[2], "Close"] = np.nan
    signals = pd.Series([1, 1, 1, 1, 1], index=df.index)
    result = Backtester().run(df, signals, "T", include_mtm=True)

    assert np.isnan(result["equity_curve_mtm"].iloc[2])
    assert result["equity_curve_mtm"].iloc[1] < result["equity_curve"].iloc[1]
    assert result["equity_curve_mtm"].iloc[-1] == result["equity_curve"].iloc[-1]


def test_invalid_entry_price_does_not_produce_finite_mtm_value():
    df = _fixture_df(n=4)
    df.loc[df.index[1], "Close"] = np.nan
    result = Backtester().run(
        df, pd.Series([1, 1, 1, 1], index=df.index), "T", include_mtm=True
    )

    assert np.isnan(result["equity_curve_mtm"].iloc[1])
    assert np.isnan(result["equity_curve_mtm"].iloc[2])


def test_timeline_mtm_differences_ignore_dl_availability_transitions():
    df = _fixture_df()
    df[sorted(mpml_main.D1_FEATURE_COLS)[0]] = [
        np.nan if i % 2 else 1.0 for i in range(len(df))
    ]
    bt, result = _run(df, SIGNALS)
    out = pd.DataFrame(_rows(df, result, bt))

    expected = result["equity_curve_mtm"]
    np.testing.assert_allclose(out["portfolio_equity_mtm"], expected)
    expected_pnl = expected.diff()
    expected_pnl.iloc[0] = expected.iloc[0] - bt.initial_capital
    np.testing.assert_allclose(out["portfolio_pnl_mtm"], expected_pnl)
    expected_return = expected_pnl / expected.shift(1)
    expected_return.iloc[0] = expected_pnl.iloc[0] / bt.initial_capital
    np.testing.assert_allclose(out["portfolio_return_mtm"], expected_return)
    assert (~out["dl_available"]).any() and out["portfolio_pnl_mtm"].notna().any()


def test_timeline_rejects_missing_mtm_or_misaligned_required_series():
    df = _fixture_df()
    bt, result = _run(df, SIGNALS)
    missing = dict(result)
    missing.pop("equity_curve_mtm")
    with pytest.raises(ValueError, match="requires equity_curve_mtm"):
        _rows(df, missing, bt)

    misaligned = dict(result)
    misaligned["equity_curve_mtm"] = result["equity_curve_mtm"].iloc[:-1]
    with pytest.raises(ValueError, match="timestamps do not match test bars"):
        _rows(df, misaligned, bt)


def test_timeline_rejects_duplicate_or_misaligned_timestamps():
    df = _fixture_df()
    bt, result = _run(df, SIGNALS)
    duplicate_df = pd.concat([df, df.iloc[[-1]]])
    with pytest.raises(ValueError, match="timestamps must be unique"):
        mpml_main._build_dl_conditional_portfolio_rows(
            pair_name="EURUSD",
            fold_id=0,
            df_test=duplicate_df,
            selected_s=pd.Series(["TF"] * len(duplicate_df), index=duplicate_df.index),
            dyn_res=result,
            initial_capital=bt.initial_capital,
        )

    misaligned_selection = pd.Series(
        ["TF"] * len(df), index=df.index + pd.Timedelta(days=1)
    )
    with pytest.raises(ValueError, match="selected strategy state timestamps"):
        mpml_main._build_dl_conditional_portfolio_rows(
            pair_name="EURUSD",
            fold_id=0,
            df_test=df,
            selected_s=misaligned_selection,
            dyn_res=result,
            initial_capital=bt.initial_capital,
        )

    duplicated_execution = dict(result)
    timeline = result["execution_timeline"].copy()
    timeline.loc[1, "timestamp"] = timeline.loc[0, "timestamp"]
    duplicated_execution["execution_timeline"] = timeline
    with pytest.raises(ValueError, match="execution_timeline index contains duplicates"):
        _rows(df, duplicated_execution, bt)


def test_writer_checks_required_columns_on_every_row(tmp_path, monkeypatch):
    monkeypatch.setattr(mpml_main, "_CURRENT_RUN_OUTPUT_DIR", tmp_path)
    df = _fixture_df()
    bt, result = _run(df, SIGNALS)
    rows = _rows(df, result, bt)
    del rows[-1]["portfolio_return_mtm"]

    with pytest.raises(ValueError, match="missing required columns"):
        mpml_main._write_dl_conditional_portfolio_timeline(rows)


def test_writer_has_exact_schema_and_sorts_timestamps_within_each_key(tmp_path, monkeypatch):
    monkeypatch.setattr(mpml_main, "_CURRENT_RUN_OUTPUT_DIR", tmp_path)
    df = _fixture_df()
    bt, result = _run(df, SIGNALS)
    rows = _rows(df, result, bt, fold=0, pair="EURUSD")
    path = mpml_main._write_dl_conditional_portfolio_timeline(
        rows[2:] + rows[:2]
    )
    exported = pd.read_csv(path)

    assert list(exported.columns) == EXPECTED_COLUMNS
    assert list(exported["timestamp"]) == sorted(exported["timestamp"])
    assert not exported.duplicated(["pair", "fold", "timestamp"]).any()


def test_mtm_return_is_nan_when_previous_equity_is_zero_or_invalid():
    df = _fixture_df()
    bt, result = _run(df, SIGNALS)
    result = dict(result)
    mtm = result["equity_curve_mtm"].copy()
    mtm.iloc[0] = 0.0
    mtm.iloc[2] = np.nan
    result["equity_curve_mtm"] = mtm
    out = pd.DataFrame(_rows(df, result, bt))

    assert np.isnan(out["portfolio_return_mtm"].iloc[1])
    assert np.isnan(out["portfolio_return_mtm"].iloc[3])


def test_realized_and_mtm_returns_are_nan_for_nonfinite_previous_equity():
    df = _fixture_df()
    bt, result = _run(df, SIGNALS)
    result = dict(result)
    result["equity_curve"] = result["equity_curve"].copy()
    result["equity_curve_mtm"] = result["equity_curve_mtm"].copy()
    result["equity_curve"].iloc[0] = np.inf
    result["equity_curve_mtm"].iloc[0] = np.inf
    out = pd.DataFrame(_rows(df, result, bt))

    assert np.isnan(out["portfolio_return"].iloc[1])
    assert np.isnan(out["portfolio_return_mtm"].iloc[1])
