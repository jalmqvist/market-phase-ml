# MPML-owned PhaseAware runtime boundary

**Status: implemented.** The causal entry point
`mpml.runtime.phaseaware.latest_decision()` is a lightweight, additive API for
the fixed canonical `PhaseAware_TF4_MR42` composition.

## Separate workflows

```text
MPML research workflow
raw history -> MarketDataPipeline.prepare()/engineer()
            -> phases / research features -> evaluation / Backtester
            -> StrategyEvaluation and Recommendation artifacts

MPML causal runtime
completed D1 OHLCV -> validation -> MarketPhaseDetector
                   -> PhaseAwareStrategy(TF4, MR42)
                   -> same-bar immutable decision
```

The two paths intentionally share the canonical phase detector and strategy
implementations, but not their preparation step. Research preparation adds
forward-return targets and drops rows for research/ML use. The runtime
instead rejects forward labels and must preserve the latest completed bar.
Replacing research preparation with a causal runtime transform would risk
changing established research and backtest semantics; reusing research
preparation would violate the runtime input and timing contract. Runtime
preparation therefore remains a small, isolated validation/orchestration
layer that sends raw OHLCV through canonical MPML logic.

## Cross-project responsibilities

MPML remains the owner of phase and TF4/MR42 calculations and publishes their
code-level runtime API and MPML revision. A consumer such as MRML may invoke
that API but should not duplicate those calculations. This boundary does not
replace the existing serialized MPML-to-MRML Recommendation and
StrategyEvaluation artifact contract: those are research evidence, while
this interface evaluates one latest completed bar. Neither interface
implicitly defines broker execution, order submission, or fill behavior.

## Import isolation

`src` retains its existing public exports via module-level lazy attribute
resolution. `src.strategies` no longer eagerly imports the daily DL feature
module just to access its column-name constant; strategy-selector access
resolves that constant on demand, retaining its prior module attribute.
Consequently the canonical strategies and registry can be imported without
loading `main.py`, research data sources, DL surface loaders, models,
visualization, walk-forward evaluation, or training libraries. The canonical
implementation itself is still used.

See [the runtime contract](../runtime/phaseaware_runtime.md) for accepted
history, causal preparation, warm-up, start policy, output semantics and
error behavior.
