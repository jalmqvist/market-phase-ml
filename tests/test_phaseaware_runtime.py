"""Tests for the causal PhaseAware(TF4, MR42) runtime (mpml.runtime.phaseaware)."""
import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mpml.runtime import phaseaware as rt
from mpml.runtime.phaseaware import (
    ForwardLabelError,
    InsufficientHistory,
    InvalidHistory,
    NonFiniteOutput,
    UnmatchedPhase,
    latest_decision,
    required_history,
)
from src.phases import MarketPhaseDetector
from src.strategies import PhaseAwareStrategy

REPO_ROOT = Path(__file__).resolve().parents[1]


def make_history(n=1200, seed=20240101, tz=None):
    """Deterministic EURUSD-like D1 OHLCV with alternating volatility regimes."""
    rng = np.random.default_rng(seed)
    vol = np.where((np.arange(n) // 150) % 2 == 0, 0.004, 0.009)
    drift = np.where((np.arange(n) // 300) % 2 == 0, 0.0004, 0.0)
    close = 1.10 * np.exp(np.cumsum(drift + rng.normal(0, 1, n) * vol * 0.6))
    open_ = np.concatenate([[1.10], close[:-1]]) * (1 + rng.normal(0, 0.0005, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2018-01-01", periods=n, freq="B", tz=tz)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1000.0},
        index=idx,
    )


def canonical(df):
    """The canonical research-engine calls, evaluated directly."""
    phased = MarketPhaseDetector().detect_phases(df)
    sig, sl, tp = PhaseAwareStrategy("TF4", "MR42").generate_signals(phased)
    return phased, sig, sl, tp


def digest(values):
    return hashlib.sha256(",".join(f"{v:.6f}" for v in values).encode()).hexdigest()


@pytest.fixture(scope="module")
def hist():
    return make_history()


# 1. Import isolation ------------------------------------------------------

HEAVY = [
    "main", "src.models", "src.visualization", "src.data", "src.data_sources",
    "src.dl_daily_features", "src.dl_surface_loader", "src.evaluation_scope",
    "schemas", "sklearn", "matplotlib", "seaborn", "scipy", "xgboost",
    "yfinance", "torch",
]


def test_runtime_import_isolation():
    code = (
        "import sys\n"
        "import mpml.runtime.phaseaware as p\n"
        "p.required_history()\n"
        f"heavy = {HEAVY!r}\n"
        "loaded = sorted(m for m in sys.modules if any(m == h or m.startswith(h + '.') for h in heavy))\n"
        "assert not loaded, loaded\n"
        "assert not any('Backtester' in n for n in dir(p))\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


def test_runtime_decision_path_stays_isolated(hist):
    code = (
        "import sys\n"
        "sys.path.insert(0, 'tests')\n"
        "from test_phaseaware_runtime import make_history, HEAVY\n"
        "from mpml.runtime.phaseaware import latest_decision\n"
        "latest_decision(make_history(400))\n"
        "loaded = sorted(m for m in sys.modules if any(m == h or m.startswith(h + '.') for h in HEAVY))\n"
        "assert not loaded, loaded\n"
    )
    res = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr


# 2. Backward compatibility of public imports -------------------------------

def test_src_public_exports_still_importable():
    import src

    assert len(src.__all__) == 20
    for name in src.__all__:
        assert getattr(src, name) is not None, name
    namespace = {}
    exec("from src import *", namespace)
    for name in src.__all__:
        assert name in namespace


def test_src_exports_are_the_submodule_objects():
    import src
    import src.phases
    import src.strategies
    import src.strategy_registry as reg

    assert src.MarketPhaseDetector is src.phases.MarketPhaseDetector
    assert src.PhaseAwareStrategy is src.strategies.PhaseAwareStrategy
    assert src.Backtester is src.strategies.Backtester
    assert src.get_default_strategy_registry is reg.get_default_strategy_registry
    from src.data import MarketDataPipeline
    from src.models import PhaseMLExperiment
    assert src.MarketDataPipeline is MarketDataPipeline
    assert src.PhaseMLExperiment is PhaseMLExperiment
    with pytest.raises(AttributeError):
        src.does_not_exist


def test_strategies_dl_feature_cols_alias_preserved():
    import src.strategies as s
    from src.dl_daily_features import D1_FEATURE_COLS

    assert s._DL_D1_FEATURE_COLS == D1_FEATURE_COLS
    from src.strategies import _DL_D1_FEATURE_COLS
    assert _DL_D1_FEATURE_COLS == D1_FEATURE_COLS


def test_registry_behavior_unchanged():
    from src import get_default_strategy_registry

    registry = get_default_strategy_registry()
    ids = {d.strategy_id for d in registry.all()}
    assert ids == {"TF1", "TF2", "TF3", "TF4", "TF5", "MR1", "MR2", "MR3", "MR32", "MR42", "MR5"}
    s = PhaseAwareStrategy("TF4", "MR42")
    assert s.name == "PhaseAware_TF4_MR42"
    assert s.PHASE_SIZE_MULTIPLIER == {
        "HV_Trend": 0.5, "LV_Trend": 1.5, "HV_Ranging": 0.5, "LV_Ranging": 1.0,
    }
    assert MarketPhaseDetector.STOP_MULTIPLIERS == {
        "HV_Trend": 2.0, "LV_Trend": 1.0, "HV_Ranging": 2.0, "LV_Ranging": 0.5,
    }


# 3. Input validation -------------------------------------------------------

@pytest.mark.parametrize("col", ["next_return", "next_direction", "next_direction_binary", "future_ret", "label"])
def test_forward_label_columns_rejected(hist, col):
    df = hist.copy()
    df[col] = 0.0
    with pytest.raises(ForwardLabelError):
        latest_decision(df)


def test_research_prepare_output_is_rejected(hist):
    from src.data import MarketDataPipeline

    prepared = MarketDataPipeline().prepare(hist)
    with pytest.raises(ForwardLabelError):
        latest_decision(prepared)


def test_validation_errors(hist):
    with pytest.raises(InvalidHistory):
        latest_decision(hist.drop(columns=["High"]))
    with pytest.raises(InvalidHistory):
        latest_decision(hist.iloc[::-1])
    with pytest.raises(InvalidHistory):
        latest_decision(pd.concat([hist, hist.iloc[-1:]]))
    with pytest.raises(InvalidHistory):
        latest_decision(hist.reset_index(drop=True))
    bad = hist.copy()
    bad.iloc[-1, bad.columns.get_loc("Close")] = np.nan
    with pytest.raises(InvalidHistory):
        latest_decision(bad)
    with pytest.raises(InvalidHistory):
        latest_decision([1, 2, 3])


def test_input_not_mutated_and_extra_columns_ignored(hist):
    df = hist.copy()
    df["adx"] = 99.0  # must not leak into the canonical strategies
    before = df.copy()
    d = latest_decision(df)
    pd.testing.assert_frame_equal(df, before)
    assert d == latest_decision(hist)


def test_timezone_aware_input_preserves_timestamp(hist):
    aware = make_history(tz="UTC")
    d_aware = latest_decision(aware)
    d_naive = latest_decision(hist)
    assert d_aware.timestamp == aware.index[-1]
    assert d_aware.timestamp.tz is not None
    assert (d_aware.phase, d_aware.signal, d_aware.sl_pct, d_aware.tp_pct) == (
        d_naive.phase, d_naive.signal, d_naive.sl_pct, d_naive.tp_pct)


# 4/5. Warm-up --------------------------------------------------------------

def test_required_history_derivation():
    assert required_history() == 139  # (atr 14 - 1) + median min_periods 126
    assert required_history() < 252


def test_warmup_components_match_observed_indicator_behaviour(hist):
    """Pin each derivation term to what the canonical indicators actually do."""
    from src.strategies import _lwma
    from ta.momentum import StochasticOscillator
    from ta.volatility import BollingerBands

    phased = MarketPhaseDetector().detect_phases(hist)
    pos = lambda s: int(np.flatnonzero(s.to_numpy() > 0)[0])
    assert pos(phased["atr"]) + 1 == 14                      # atr_period
    assert pos(phased["adx"]) + 1 == 28                      # 2 * adx_period
    med = phased["atr_pct"].rolling(252, min_periods=126).median()
    assert int(np.flatnonzero(med.notna().to_numpy())[0]) + 1 == 126  # min_periods
    lw = _lwma(hist["Close"], 40)
    assert int(np.flatnonzero(lw.notna().to_numpy())[0]) + 1 == 40
    st = StochasticOscillator(hist["High"], hist["Low"], hist["Close"], window=5, smooth_window=3).stoch()
    assert int(np.flatnonzero(st.notna().to_numpy())[0]) + 1 == 5
    bb = BollingerBands(hist["Close"], 20, 2.0).bollinger_hband()
    assert int(np.flatnonzero(bb.notna().to_numpy())[0]) + 1 == 20


def test_rsi_not_consumed(hist):
    phased, *_ = canonical(hist)
    assert "rsi" not in phased.columns


def test_insufficient_history_just_below_threshold(hist):
    n = required_history()
    with pytest.raises(InsufficientHistory) as exc:
        latest_decision(hist.iloc[: n - 1])
    assert exc.value.required == n and exc.value.available == n - 1


def test_success_at_threshold(hist):
    n = required_history()
    d = latest_decision(hist.iloc[:n])
    assert d.input_bars == n and d.required_history == n and d.warmup_complete
    assert d.timestamp == hist.index[n - 1]


# 6/7/8. Causality ----------------------------------------------------------

CUTOFFS = [139, 140, 200, 253, 400, 777, 1199]  # inclusive row positions (1199 = full history)


@pytest.fixture(scope="module")
def full_canonical(hist):
    return canonical(hist)


@pytest.mark.parametrize("k", CUTOFFS)
def test_prefix_parity_includes_cutoff_bar(hist, full_canonical, k):
    phased, sig, sl, tp = full_canonical
    d = latest_decision(hist.iloc[: k + 1])  # .iloc[:k+1] includes bar k
    assert d.timestamp == hist.index[k]
    assert d.input_bars == k + 1
    assert d.phase == phased["phase"].iloc[k]
    assert d.signal == sig.iloc[k]
    assert d.sl_pct == sl.iloc[k]
    assert d.tp_pct == tp.iloc[k]


@pytest.mark.parametrize("k", [200, 500, 900])
def test_later_bars_do_not_change_earlier_decision(hist, k):
    base = latest_decision(hist.iloc[: k + 1])
    perturbed = hist.copy()
    rng = np.random.default_rng(7)
    tail = perturbed.iloc[k + 1:]
    perturbed.iloc[k + 1:, perturbed.columns.get_loc("Close")] = tail["Close"] * rng.uniform(0.8, 1.2, len(tail))
    perturbed["High"] = perturbed[["Open", "High", "Low", "Close"]].max(axis=1)
    perturbed["Low"] = perturbed[["Open", "High", "Low", "Close"]].min(axis=1)
    again = latest_decision(perturbed.iloc[: k + 1])
    assert dataclasses.replace(again) == base
    ph_a, sig_a, sl_a, tp_a = canonical(hist)
    ph_b, sig_b, sl_b, tp_b = canonical(perturbed)
    assert (ph_a["phase"].iloc[: k + 1] == ph_b["phase"].iloc[: k + 1]).all()
    assert (sig_a.iloc[: k + 1] == sig_b.iloc[: k + 1]).all()


def test_phase_and_volatility_classification_are_trailing_only(hist, full_canonical):
    phased = full_canonical[0]
    cols = ["atr", "atr_pct", "adx", "high_vol", "trending", "phase"]
    for k in (150, 300, 600, 1000):
        prefix = MarketPhaseDetector().detect_phases(hist.iloc[: k + 1])
        pd.testing.assert_frame_equal(prefix[cols], phased[cols].iloc[: k + 1])


def test_same_bar_consistency(hist):
    for k in (300, 700, 1100):
        sub = hist.iloc[: k + 1]
        d = latest_decision(sub)
        phased, sig, sl, tp = canonical(sub)
        assert phased.index[-1] == sig.index[-1] == sl.index[-1] == tp.index[-1] == d.timestamp
        mult = PhaseAwareStrategy.PHASE_SIZE_MULTIPLIER[d.phase]
        assert abs(d.signal) in (0.0, mult)
        if d.phase.endswith("Trend"):
            assert d.sl_pct == 0.0 and d.tp_pct == 0.0  # TF4: no fixed SL/TP
        else:
            assert (d.sl_pct, d.tp_pct) == (0.025, 0.0125)  # MR42


def test_signal_is_not_shifted(hist, full_canonical):
    _, sig, _, _ = full_canonical
    assert latest_decision(hist).signal == sig.iloc[-1]


def test_all_phases_and_signs_exercised(full_canonical):
    phased, sig, _, _ = full_canonical
    assert set(phased["phase"].iloc[200:]) == set(PhaseAwareStrategy.PHASE_SIZE_MULTIPLIER)
    assert {-1.5, 1.5, 0.5, -0.5, 1.0, -1.0}.issubset(set(sig.iloc[200:]))


def test_history_start_dependence_is_real(hist):
    """Documented behaviour: a different start can change the forward-filled state."""
    # This deterministic cutoff follows an established short state. Starting
    # later omits that entry, so the state remains flat instead.
    full = latest_decision(hist.iloc[:161]).signal
    trimmed = latest_decision(hist.iloc[20:161]).signal
    assert (full, trimmed) == (-0.5, 0.0)


# Output validation -----------------------------------------------------------

def test_unmatched_phase_raises(hist, monkeypatch):
    def bad(self, df):
        out = df.copy()
        out["phase"] = "Unknown"
        return out
    monkeypatch.setattr(MarketPhaseDetector, "detect_phases", bad)
    with pytest.raises(UnmatchedPhase):
        latest_decision(hist)


def test_non_finite_output_raises(hist, monkeypatch):
    def nan_signals(self, df):
        z = pd.Series(0.0, index=df.index)
        return pd.Series(np.nan, index=df.index), z, z
    monkeypatch.setattr(PhaseAwareStrategy, "generate_signals", nan_signals)
    with pytest.raises(NonFiniteOutput):
        latest_decision(hist)


def test_result_is_frozen_and_complete(hist):
    d = latest_decision(hist)
    assert d.composition == "PhaseAware_TF4_MR42"
    assert isinstance(d.mpml_revision, str) and d.mpml_revision
    with pytest.raises(dataclasses.FrozenInstanceError):
        d.signal = 1.0
    assert {f.name for f in dataclasses.fields(d)} >= {
        "timestamp", "phase", "signal", "sl_pct", "tp_pct",
        "required_history", "input_bars", "composition", "mpml_revision",
    }


# 9. Pinned canonical output (synthetic EURUSD-like fixture) ----------------

PINNED_SIGNAL_SHA256 = "f8ae0e6376e0cdc7531eadf89bc5d31699c084ed2ac4a00e92c15f3476d32aa2"
PINNED_PHASE_SHA256 = "256c689b0afac524b8336b5b2c24f511fd17fc636532a4c6ee3a0b0655809d4b"


def test_pinned_canonical_signals_unchanged(full_canonical):
    phased, sig, _, _ = full_canonical
    assert digest(sig.to_numpy()) == PINNED_SIGNAL_SHA256
    assert hashlib.sha256(",".join(phased["phase"]).encode()).hexdigest() == PINNED_PHASE_SHA256
