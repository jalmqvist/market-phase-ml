# Causal PhaseAware runtime

**Status: implemented and tested in MPML.** The current public entry point is
`mpml.runtime.phaseaware.latest_decision(history)`. It returns an immutable
`PhaseAwareDecision` for the one fixed composition `PhaseAware_TF4_MR42`.
There is no dynamic composition selection, broker adapter, H1-to-D1
aggregation, behavioral-state resolver, or execution interface here.

## Input contract

`history` is a pandas DataFrame indexed by a unique, ascending
`DatetimeIndex`. Its required columns are numeric `Open`, `High`, `Low`,
`Close`, and `Volume`. Values must be finite; OHLC prices must be positive
and internally consistent, and volume must be non-negative. The index may
be timezone-naive or timezone-aware; it is not localized, converted, sorted,
deduplicated, or stripped of its timezone. The last row is retained as the
latest completed bar. The caller is responsible for submitting only
completed D1 bars and for a consistent D1 cadence.

Forward-looking columns are rejected, including names with the prefixes
`next_`, `fwd_`, `forward`, `future`, `target`, and `label` (case-insensitive).
The runtime deliberately does not accept a `MarketDataPipeline.prepare()`
result: that research method adds forward targets such as `next_return`.

The result fields are timestamp, phase, signal, `sl_pct`, `tp_pct`,
`required_history`, `input_bars`, composition, and the checked-out MPML git
revision (or `MPML_REVISION` when explicitly set; `"unknown"` if git metadata
is unavailable). `warmup_complete` reports whether the supplied bar count
meets the minimum.

## Canonical causal path and research separation

Runtime-specific preparation passes only validated raw OHLCV to
`MarketPhaseDetector.detect_phases()` over the entire supplied history. Its
output then goes directly to the canonical
`PhaseAwareStrategy("TF4", "MR42").generate_signals()`. The runtime neither
reimplements phase classification/strategy logic nor supplies precomputed
indicator columns (notably, TF/MR strategies may consume a preexisting `adx`
column). No forward labels or shifted signal are created.

This path is separate from `MarketDataPipeline.prepare()` and `engineer()`
because those methods serve research/ML workflows: `prepare()` deliberately
adds future targets and drops rows, and `engineer()` adds research indicators
including RSI and trims initial NaNs. Reusing them would violate the
runtime's no-forward-label, preserve-the-latest-bar input contract and would
needlessly impose indicators the fixed runtime does not consume. Research
preparation semantics remain unchanged.

MPML owns this calculation and its runtime revision. A consuming project
(including MRML) may call the interface, but does not own or duplicate
MPML's phase/strategy calculations. This is a Python runtime boundary,
separate from the existing MPML-to-MRML serialized Recommendation artifact;
the interface does not itself transport a recommendation or research
evidence.

## Warm-up and supplied-history start

The current detector and selected strategies imply a minimum of **126
bars**:

| Dependency | Required observations |
|---|---:|
| ADX(14), including its `ta` startup placeholders | 28 |
| ATR(14) | 14 |
| ATR%-median classification | rolling `min_periods` 126 = 126 |
| TF4 LWMA(40) and its previous value | 41 |
| TF4 Stochastic %K(5) | 5 |
| MR42 Bollinger Bands(20) | 20 |
| MR42 previous close | 2 |

The maximum is 126. The volatility rolling window is 252, but its actual
`min_periods` is 126; pandas counts the canonical numeric ATR startup
placeholders toward that threshold, so the 13 ATR startup bars are not added
again. The 252-bar window is not required. RSI is excluded: neither this phase
detector nor TF4/MR42 consumes it. Warm-up counts observations; it does not
truncate or reset state.

TF4/MR42 forward-fill a position-like signal state. Consequently a later
decision can depend on an entry (or exit) earlier in the supplied history.
There is no sliding window or arbitrary warm-up prefix: every call evaluates
the whole history from its first supplied bar. Callers must use a stable,
deterministic history start for comparable decisions. Changing that start
can change the latest signal even where later OHLCV bars are identical.

## Decision semantics

The decision becomes available **at the close of `timestamp`**. `phase`,
`signal`, `sl_pct`, and `tp_pct` all describe that same latest completed bar.
The signal is not shifted. Historical Backtester next-bar execution is a
separate research convention and is not part of this runtime.

Signal sign identifies direction: positive is long, negative is short, zero
is flat. Magnitude is the canonical phase size multiplier, not units,
leverage, a probability, or an account-risk amount:

| Phase | Multiplier |
|---|---:|
| `HV_Trend` | 0.5 |
| `LV_Trend` | 1.5 |
| `HV_Ranging` | 0.5 |
| `LV_Ranging` | 1.0 |

`sl_pct` and `tp_pct` are fractions of entry price: `0.025` represents 2.5%.
TF4 supplies zero for both because its canonical exit is a stochastic
reversal, not fixed SL/TP. MR42 supplies 0.025 SL and 0.0125 TP. These
percentages are **not** `stop_atr_mult`. The latter is a separate
phase-specific ATR multiple used by the research Backtester for stop
distance/position sizing; this runtime returns neither that multiplier nor
position size, and does not change Backtester handling.

## Errors and imports

`InvalidHistory` covers invalid DataFrame schema, timestamps, order,
uniqueness, or values. `ForwardLabelError` specializes it for forbidden
label columns. `InsufficientHistory` reports the supplied and required bar
counts. `UnmatchedPhase` and `NonFiniteOutput` explicitly reject unusable
canonical results. The runtime does not silently drop input rows or return a
success-shaped fallback.

Import with `from mpml.runtime.phaseaware import latest_decision`. Imports
are lazy: importing the entry point itself does not load `main.py`, the
Backtester, walk-forward evaluation, models/training, the research data
pipeline, or behavioral-surface artifact loaders. Execution requires the
existing core Python dependencies pandas, NumPy, and `ta`, as well as this
MPML source package. Git metadata is optional; use `MPML_REVISION` in
packaged deployments when the exact revision must be supplied.

## Tested and unverified

Automated tests cover import isolation and existing `src` imports, schema and
label validation, warm-up thresholds, inclusive prefix parity, future-bar
invariance, same-bar consistency, and a deterministic EURUSD-like synthetic
OHLCV signal/phase fingerprint. The repository checkout contains no frozen
EURUSD raw-bar signal fixture or expected signal artifact, so parity against
an external/historical frozen EURUSD signal fixture is **not verified**.
The runtime does not claim broker/execution parity.
