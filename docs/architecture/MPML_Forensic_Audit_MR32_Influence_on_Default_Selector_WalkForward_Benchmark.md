# MPML Forensic Audit: MR32 Influence on Default Selector / Walk-Forward Benchmark

## Executive Conclusion

**Root cause: Architectural coupling.**

The default MPML run trains the strategy selector using backtest results for the **full evaluated strategy universe** (MR1, MR2, MR32, MR42, MR5 + TF1–TF5), not only the final evaluation policy pair (TF4 + MR42). Because MR32 is a member of the full MR universe, its equity curve participates in generating selector training labels. Changing the MR32 implementation changes which time-windows are labeled `"MeanReversion"` in the selector training data, which alters the trained selector model, which changes the TF4/MR42 routing in the walk-forward benchmark.

This is **not** a DL/behavioral-surface issue, nor a G3 evaluation-scope issue at the inference level. It is an undocumented dependency in the selector **training** stage that was never covered by the `EvaluationScope` / G3 machinery.

------

## 1. Where MR32 First Enters the Default Pipeline

### Entry point: `run_backtests()` — `src/strategies.py`, line 1590

Invoked from `main.py` section **[4/5]** (Full-Universe Backtests), around line ~3009:

Python

```
results_hardcoded = run_backtests(
    df=df,
    initial_capital=INITIAL_CAPITAL,
    use_atr_sizing=False,
    evaluation_policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
    ...
)
```

Inside `run_backtests()`:

Python

```
tf_strategies, mr_strategies = instantiate_evaluated_strategy_dicts()
```

`instantiate_evaluated_strategy_dicts()` (`src/strategies.py`, line 85):

Python

```
_DEFAULT_EVALUATED_MR_STRATEGY_IDS = ("MR1", "MR2", "MR32", "MR42", "MR5")  # line 20

def instantiate_evaluated_strategy_dicts() -> tuple[dict, dict]:
    strategy_registry = get_default_strategy_registry()
    tf_strategies = {
        strategy_id: strategy_registry.get(strategy_id).instantiate()
        for strategy_id in _DEFAULT_EVALUATED_TF_STRATEGY_IDS
    }
    mr_strategies = {
        strategy_id: strategy_registry.get(strategy_id).instantiate()
        for strategy_id in _DEFAULT_EVALUATED_MR_STRATEGY_IDS
    }
    return tf_strategies, mr_strategies
```

`run_backtests()` then iterates over **all** `mr_strategies` (lines 1664–1668):

Python

```
for name, strategy in sorted(mr_strategies.items()):
    signals, sl_pct, tp_pct = strategy.generate_signals(df)
    results[name] = backtester.run(df, signals, name, sl_pct, tp_pct)
```

**MR32 first appears in the pipeline here.** Its equity curve is written into `results_hardcoded`, which becomes `all_pair_results[pair_name]` and ultimately `hardcoded_results`.

This is also why `results_per_pair__dl_enabled.csv` differs immediately: that file comes from `hardcoded_results` and directly reflects MR32's changed performance.

------

## 2. `_build_selector_reference_results()` Audit

**Location:** `main.py`, line 1976.

Python

```
def _build_selector_reference_results(
    *,
    df_full: pd.DataFrame,
    pair_name: str,
    strategy_registry=None,
    policy_id: str = DEFAULT_PHASEAWARE_POLICY_ID,
) -> dict:
    """Build the minimal full-history results needed for selector labels."""
    strategy_registry = strategy_registry or get_default_strategy_registry()
    baseline_tf, baseline_mr = resolve_phaseaware_strategy_pair(policy_id)
    results = {
        baseline_tf: _run_registry_strategy_backtest(..., strategy_id=baseline_tf, ...),
        baseline_mr: _run_registry_strategy_backtest(..., strategy_id=baseline_mr, ...),
    }
    pa = PhaseAwareStrategy(baseline_tf, baseline_mr)
    results[phaseaware_strategy_name(policy_id)] = backtester.run(...)
    return results
```

**Findings:**

| Question                                                     | Answer                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| What strategies does it evaluate?                            | Only the policy pair: TF4 + MR42 + PhaseAware_TF4_MR42       |
| Does it use the resolved evaluation scope?                   | No — it uses the **policy** via `resolve_phaseaware_strategy_pair()`, which is effectively the same (TF4 + MR42) |
| Does it use the strategy registry directly or all registered strategies? | Only the two policy strategies; correct and narrow           |
| Does it include MR32?                                        | **No.** MR32 is never included here                          |
| What artifact does it produce?                               | A `dict` of `{strategy_id: backtest_result}` with 3 entries (TF4, MR42, PhaseAware) |
| Who consumes it?                                             | `_build_causal_selector_training_data()` in the walk-forward per-fold loop |
| What does the output become?                                 | Input for `StrategyPerformanceTracker.compute_strategy_returns()` → selector training labels |

**Critical finding:** `_build_selector_reference_results()` is **correctly scoped** to the policy pair. It is **not** the source of the MR32 coupling. However, it is only called on the **explicit-strategy path** (`_run_full_universe=False`). On the default path, `pair_results_full` is populated differently (see section 5 below).

------

## 3. Is MR32 Intentionally Part of Selector Training?

**Classification: B — Architectural coupling.**

The code provides no documentation or comment indicating that MR32's inclusion in the selector training universe is intentional. The mechanism is:

1. `run_backtests()` iterates over `_DEFAULT_EVALUATED_MR_STRATEGY_IDS`, a hardcoded tuple defined at the top of `src/strategies.py` (line 20).
2. This constant predates the G3 `EvaluationScope` machinery and is not constrained by it.
3. There is no selector-specific "reference universe" constant. The full evaluation universe is repurposed as selector training data.
4. The documentation in `evaluation_scope.py` and `MPML_Architecture_Roadmap.md` makes no mention of MR32 (or any strategy outside TF4/MR42) participating in selector training.

The coupling exists because the same `hardcoded_results` dict (containing all 10 strategies) is used for **both**:

- Legacy results reporting/aggregation (where the full universe is appropriate), and
- Selector training label generation (where only the policy pair is architecturally correct)

------

## 4. How the Selector Obtains Its Strategy Labels

The label pipeline runs in **two places**:

### Place 1: Global selector training — `main.py` section [4b/5] (~line 3110)

Python

```
pair_backtest = hardcoded_results.get(pair_name, {})   # ← FULL universe (includes MR32)
tracker = StrategyPerformanceTracker(window_days=20)
training_data = tracker.compute_strategy_returns(df, pair_backtest)
selector = StrategySelector(...)
selector.train(training_data, ...)
```

### Place 2: Walk-forward per-fold causal selector — `main.py` ~line 3551

Python

```
if _run_full_universe:
    pair_results_full = hardcoded_results.get(pair_name, {})    # ← FULL universe
else:
    pair_results_full = _build_selector_reference_results(...)   # ← policy-only (TF4+MR42)
```

Then:

Python

```
training_data, _ = _build_causal_selector_training_data(
    pair_results_full=pair_results_full,   # ← still full universe on default path
    ...
)
selector.train(training_data, ...)
```

### Label assignment mechanism

`StrategyPerformanceTracker.compute_strategy_returns()` (`src/models.py`, line 1451):

For each bar `i`, it looks ahead `window_days` bars and determines the `best_strategy`:

Python

```
strategy_returns = {name: float(row[f'{name}_return']) for name in strategy_names}
best_strategy = sorted(strategy_returns.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
row['best_strategy'] = best_strategy   # e.g. "MR32"
```

`StrategySelector._strategy_to_category()` (`src/models.py`, line 1581):

Python

```
@staticmethod
def _strategy_to_category(strategy_name: str) -> str:
    if strategy_name.startswith('TF'):
        return 'TrendFollowing'
    elif strategy_name.startswith('MR'):
        return 'MeanReversion'    # ← MR32 maps here
    elif strategy_name.startswith('PhaseAware'):
        return 'PhaseAware'
```

**Therefore:** Any bar where MR32 outperforms TF1–TF5, MR42, MR1, MR2, MR5, and PhaseAware over the next 20 bars is labeled `"MeanReversion"` in the selector training data. MR32 directly contributes to the `MeanReversion` training population.

This explains the observed population shift:

- old MR32 (MA200-based): MeanReversion 19,389; PhaseAware 32,509; TrendFollowing 8,710
- new MR32 (phase-aware + ADX): MeanReversion 20,119; PhaseAware 32,789; TrendFollowing 7,700

The new MR32 wins more "best strategy" contests in certain market windows. The new MR32 implementation causes approximately 730 additional bars to be labeled `MeanReversion` and approximately 280 additional bars to be labeled `PhaseAware`, while `TrendFollowing` loses approximately 1,010 labels.

------

## 5. The Complete Causal Chain

Code

```
MR32 implementation change
    (new: phase-aware + ADX filter  vs  old: MA(200)-based)
         ↓
src/strategies.py: MR32Strategy.generate_signals()
    produces a different signal/equity curve
         ↓
src/strategies.py: run_backtests()  [called from main.py ~line 3009]
    instantiate_evaluated_strategy_dicts() → _DEFAULT_EVALUATED_MR_STRATEGY_IDS
    contains "MR32" — full MR universe is evaluated
         ↓
main.py: all_pair_results / hardcoded_results
    results_per_pair.csv changes here (first observable divergence)
    [results_ml_backtest unchanged — it doesn't use MR32 directly]
         ↓
main.py section [4b/5] + walk-forward per-fold:
    StrategyPerformanceTracker.compute_strategy_returns(pair_backtest)
    for each bar: which strategy had best 20-bar forward return?
    MR32's changed equity curve → different set of bars labeled "MeanReversion"
         ↓
StrategySelector.train(training_data)
    training labels: "MeanReversion" population changes by ~730 observations
    "TrendFollowing" population decreases by ~1,010 observations
    XGBoost decision boundaries shift
         ↓
selector_state_timeline changes in ~13,380 / 60,608 observations
         ↓
StrategySelector_Dynamic.generate_signals() [walk-forward inference]
    default_tf = "TF4",  default_mr = "MR42"  (from phaseaware_default policy)
    INFERENCE executes TF4 or MR42 signals based on selector prediction
    Different selector routing → different bars execute TF4 vs MR42 vs PhaseAware
         ↓
Different walk-forward signals → different backtester.run() results
         ↓
All 14 pair results change
```

Note that **MR32 is never executed during inference**. The selector predicts a strategy *type* (MeanReversion/TrendFollowing/PhaseAware) and then `StrategySelector_Dynamic` executes `self.default_mr = "MR42"` (line 1841: `mr_sigs, ... = self.mr_strategies[self.default_mr].generate_signals(df)`). MR32 only affects the selector routing probability distribution, not the signal generated when MeanReversion is selected.

------

## 6. Comparison Against Documented Architecture

### G3 documented requirements

From `docs/architecture/MPML_Architecture_Roadmap.md`:

> **The default invocation, without explicit strategy selection, MUST preserve the existing benchmark evaluation scope exactly.**

> Evaluation Scope determines which strategies participate in a particular experiment.

### Compliance analysis

| Requirement                                                  | Status                                                       |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Evaluation Scope constrains which strategies are **evaluated** in the walk-forward | ✅ Satisfied — `_run_full_universe` flag and `filter_strategy_specs` correctly gate strategy execution |
| Default invocation uses `phaseaware_default` → TF4 + MR42    | ✅ Satisfied — `resolve_evaluation_scope` correctly resolves to TF4 + MR42 |
| Explicit `--strategy` MUST affect only the set of strategies evaluated | ✅ Satisfied — explicit path skips full-universe backtests and selector training |
| Default invocation MUST preserve existing benchmark evaluation scope **exactly** | ❌ **Violated** — the selector training uses `_DEFAULT_EVALUATED_MR_STRATEGY_IDS` (MR1, MR2, MR32, MR42, MR5), which is not bounded by the evaluation scope. Any implementation change to any of these strategies can silently change the benchmark |

The documented three-way separation:

> **Strategy Capability ≠ Behavioral Surface / State ≠ Evaluation Scope**

is correctly applied at the **inference** level (which strategies execute during walk-forward). It is **not** applied at the **selector training** level. The selector training uses a broader, undocumented "reference universe" that is larger than the evaluation scope and is not governed by G3.

------

## 7. Why the Explicit `--strategy TF1` Path Is Unaffected

When `--strategy TF1` is passed:

1. `resolve_evaluation_scope()` → `EvaluationScope(strategy_ids=("TF1",), source="explicit")`
2. `should_run_full_universe_backtests(scope)` → `False` (`scope.source == "explicit"`)
3. `_run_full_universe = False`
4. **Section [4/5]** (full-universe backtests): **skipped** — `all_pair_results = {}`, `hardcoded_results = {}`
5. **Section [4b/5]** (global selector training): **skipped** — `print('Skipped: selector training is not required for explicit scope ...')`
6. Walk-forward per-fold: `pair_results_full = _build_selector_reference_results(...)` — **correctly uses only TF4+MR42 (the policy pair)**, MR32 is never included
7. `strategy_execution_timeline` and `strategy_trades` CSVs are byte-identical across MR32 changes because MR32 never enters this path

The explicit-strategy path is **correctly isolated** from the MR32 coupling because `_run_full_universe=False` gates out the entire full-universe backtest and selector training pipeline.

------

## Root-Cause Classification

**Category: Architectural coupling (undocumented design decision that creates a hidden dependency).**

This is not a regression in the G3 implementation. The coupling predates G3. The `_DEFAULT_EVALUATED_MR_STRATEGY_IDS` constant was defined before `EvaluationScope` existed and the G3 gating logic correctly prevents explicit-scope runs from being affected. However, the G3 machinery does not extend its scope-bounding to the selector training reference universe, which remains unbounded.

Whether this coupling is *intentional* cannot be determined from the code alone:

- **Possible intended interpretation**: Using all 5 MR strategies as "diverse experts" for selector training improves the quality of the MeanReversion label (the best-performing MR strategy labels the bar, regardless of which specific MR strategy is ultimately executed). This is a valid "mixture-of-experts" training philosophy.
- **Problem with that interpretation**: It is nowhere documented. It creates a hidden dependency that violates the documented guarantee of benchmark stability. Changing any strategy in the reference universe — even one not in the evaluation scope — silently changes the benchmark.

------

## Exact Files and Functions Responsible

| File                     | Function / Constant                                     | Role                                                         |
| ------------------------ | ------------------------------------------------------- | ------------------------------------------------------------ |
| `src/strategies.py:20`   | `_DEFAULT_EVALUATED_MR_STRATEGY_IDS`                    | Defines full MR evaluation universe (includes MR32)          |
| `src/strategies.py:85`   | `instantiate_evaluated_strategy_dicts()`                | Instantiates all strategies in the full universe             |
| `src/strategies.py:1590` | `run_backtests()`                                       | Runs all strategies in full universe; produces `hardcoded_results` |
| `main.py:~3009`          | Section [4/5]                                           | Calls `run_backtests()` unconditionally for default runs     |
| `src/models.py:1430`     | `StrategyPerformanceTracker.compute_strategy_returns()` | Generates per-bar `best_strategy` labels from full universe results |
| `src/models.py:1581`     | `StrategySelector._strategy_to_category()`              | Maps "MR32" → "MeanReversion"                                |
| `main.py:~3110`          | Section [4b/5]                                          | Trains global selector on `hardcoded_results` (full universe) |
| `main.py:3551–3558`      | Walk-forward per-fold selector                          | Uses `hardcoded_results` (full universe) when `_run_full_universe=True` |
| `main.py:1976`           | `_build_selector_reference_results()`                   | **Correctly scoped** (TF4+MR42 only) but only called on explicit path |

------

## Recommended Next Investigation / Fix Direction

1. **Define a "selector reference universe" constant** separate from `_DEFAULT_EVALUATED_MR_STRATEGY_IDS`. Currently there is only one constant, shared between the legacy reporting universe and the selector training reference. These should be explicitly separated so that the selector training reference can be independently bounded.
2. **Evaluate two possible fixes** (do not implement until design is agreed):
   - **Option A (strict scoping)**: Restrict selector training to the evaluation policy strategies (TF4 + MR42 + PhaseAware). Use `_build_selector_reference_results()` on the default path as well (currently it is used only on the explicit path). This would make the selector training scope match the documented evaluation scope exactly, but may reduce selector training label diversity.
   - **Option B (documented wide reference)**: Explicitly document that selector training uses the full MR/TF universe as a "diverse expert pool", enumerate which strategies are included in the reference universe, and specify that changes to any of these strategies can affect the benchmark. This legitimizes the current behavior but requires adding documentation and potentially pinning the reference strategies.
3. **Verify** that the walk-forward per-fold causal selector training (section [4b/5] per-fold loop) is the dominant driver versus the global selector (section [4b/5] pre-WF loop). The two selector training paths must be aligned on whatever universe is chosen.
4. **Add a regression test** that asserts selector training label distribution does not change when only the MR32 implementation is modified (or conversely, document the expectation that it will change).
