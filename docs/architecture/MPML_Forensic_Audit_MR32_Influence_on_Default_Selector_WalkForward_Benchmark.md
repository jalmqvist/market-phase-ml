# MPML Forensic Audit: MR32 Influence on Default Selector / Walk-Forward Benchmark

## Executive Conclusion

**Root cause: Architectural coupling.**

The default MPML run trains the strategy selector using backtest results for the
**full evaluated strategy universe** (MR1, MR2, MR32, MR42, MR5 + TF1–TF5), not
only the final evaluation policy pair (TF4 + MR42). Because MR32 is a member of
the full MR universe, its equity curve participates in generating selector training
labels. Changing the MR32 implementation changes which time-windows are labeled
`"MeanReversion"` in the selector training data, which alters the trained selector
model, which changes the TF4/MR42 routing in the walk-forward benchmark.

This is **not** a DL/behavioral-surface issue, nor a G3 evaluation-scope issue at
the inference level. It is an undocumented dependency in the selector **training**
stage that was never covered by the `EvaluationScope` / G3 machinery.

---

## 1. Where MR32 First Enters the Default Pipeline

### Entry point: `run_backtests()` — `src/strategies.py`, line 1590

Invoked from `main.py` section **[4/5]** (Full-Universe Backtests), around line ~3009:

```python
results_hardcoded = run_backtests(
    df=df,
    initial_capital=INITIAL_CAPITAL,
    use_atr_sizing=False,
    evaluation_policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
    ...
)
```

Inside `run_backtests()`:

```python
tf_strategies, mr_strategies = instantiate_evaluated_strategy_dicts()
```

`instantiate_evaluated_strategy_dicts()` (`src/strategies.py`, line 85):

```python
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

```python
for name, strategy in sorted(mr_strategies.items()):
    signals, sl_pct, tp_pct = strategy.generate_signals(df)
    results[name] = backtester.run(df, signals, name, sl_pct, tp_pct)
```

**MR32 first appears in the pipeline here.** Its equity curve is written into
`results_hardcoded`, which becomes `all_pair_results[pair_name]` and ultimately
`hardcoded_results`.

This is also why `results_per_pair__dl_enabled.csv` differs immediately: that file
comes from `hardcoded_results` and directly reflects MR32's changed performance.

---

## 2. `_build_selector_reference_results()` Audit

**Location:** `main.py`, line 1976.

```python
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

| Question | Answer |
|---|---|
| What strategies does it evaluate? | Only the policy pair: TF4 + MR42 + PhaseAware_TF4_MR42 |
| Does it use the resolved evaluation scope? | No — it uses the **policy** via `resolve_phaseaware_strategy_pair()`, which is effectively the same (TF4 + MR42) |
| Does it use the strategy registry directly or all registered strategies? | Only the two policy strategies; correct and narrow |
| Does it include MR32? | **No.** MR32 is never included here |
| What artifact does it produce? | A `dict` of `{strategy_id: backtest_result}` with 3 entries (TF4, MR42, PhaseAware) |
| Who consumes it? | `_build_causal_selector_training_data()` in the walk-forward per-fold loop |
| What does the output become? | Input for `StrategyPerformanceTracker.compute_strategy_returns()` → selector training labels |

**Critical finding:** `_build_selector_reference_results()` is **correctly scoped**
to the policy pair. It is **not** the source of the MR32 coupling. However, it is
only called on the **explicit-strategy path** (`_run_full_universe=False`). On the
default path, `pair_results_full` is populated differently (see Section 5 below).

---

## 3. Is MR32 Intentionally Part of Selector Training?

**Classification: B — Architectural coupling.**

The code provides no documentation or comment indicating that MR32's inclusion in
the selector training universe is intentional. The mechanism is:

1. `run_backtests()` iterates over `_DEFAULT_EVALUATED_MR_STRATEGY_IDS`, a hardcoded
   tuple defined at the top of `src/strategies.py` (line 20).
2. This constant predates the G3 `EvaluationScope` machinery and is not constrained
   by it.
3. There is no selector-specific "reference universe" constant. The full evaluation
   universe is repurposed as selector training data.
4. The documentation in `evaluation_scope.py` and `MPML_Architecture_Roadmap.md`
   makes no mention of MR32 (or any strategy outside TF4/MR42) participating in
   selector training.

The coupling exists because the same `hardcoded_results` dict (containing all 10
strategies) is used for **both**:

- Legacy results reporting/aggregation (where the full universe is appropriate), and
- Selector training label generation (where only the policy pair is architecturally
  correct).

---

## 4. How the Selector Obtains Its Strategy Labels

The label pipeline runs in **two places**:

### Place 1: Global selector training — `main.py` section [4b/5] (~line 3110)

```python
pair_backtest = hardcoded_results.get(pair_name, {})   # ← FULL universe (includes MR32)
tracker = StrategyPerformanceTracker(window_days=20)
training_data = tracker.compute_strategy_returns(df, pair_backtest)
selector = StrategySelector(...)
selector.train(training_data, ...)
```

### Place 2: Walk-forward per-fold causal selector — `main.py` ~line 3551

```python
if _run_full_universe:
    pair_results_full = hardcoded_results.get(pair_name, {})    # ← FULL universe
else:
    pair_results_full = _build_selector_reference_results(...)   # ← policy-only (TF4+MR42)
```

Then:

```python
training_data, _ = _build_causal_selector_training_data(
    pair_results_full=pair_results_full,   # ← still full universe on default path
    ...
)
selector.train(training_data, ...)
```

### Label assignment mechanism

`StrategyPerformanceTracker.compute_strategy_returns()` (`src/models.py`, line 1451):

For each bar `i`, it looks ahead `window_days` bars and determines the
`best_strategy`:

```python
strategy_returns = {name: float(row[f'{name}_return']) for name in strategy_names}
best_strategy = sorted(strategy_returns.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
row['best_strategy'] = best_strategy   # e.g. "MR32"
```

`StrategySelector._strategy_to_category()` (`src/models.py`, line 1581):

```python
@staticmethod
def _strategy_to_category(strategy_name: str) -> str:
    if strategy_name.startswith('TF'):
        return 'TrendFollowing'
    elif strategy_name.startswith('MR'):
        return 'MeanReversion'    # ← MR32 maps here
    elif strategy_name.startswith('PhaseAware'):
        return 'PhaseAware'
```

**Therefore:** Any bar where MR32 outperforms TF1–TF5, MR42, MR1, MR2, MR5, and
PhaseAware over the next 20 bars is labeled `"MeanReversion"` in the selector
training data. MR32 directly contributes to the `MeanReversion` training population.

This explains the observed population shift:

- old MR32 (MA200-based): MeanReversion 19,389; PhaseAware 32,509; TrendFollowing 8,710
- new MR32 (phase-aware + ADX): MeanReversion 20,119; PhaseAware 32,789; TrendFollowing 7,700

The new MR32 wins more "best strategy" contests in certain market windows, labeling
~730 additional bars as `"MeanReversion"` and ~1,010 additional bars as
`"PhaseAware"` at the cost of `"TrendFollowing"` labels.

---

## 5. The Complete Causal Chain

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
selector_state_timeline changes in ~13,380 of 60,608 observations
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

Note that **MR32 is never executed during inference**. The selector predicts a
strategy *type* (MeanReversion/TrendFollowing/PhaseAware) and then
`StrategySelector_Dynamic` executes `self.default_mr = "MR42"` (line 1841:
`mr_sigs, ... = self.mr_strategies[self.default_mr].generate_signals(df)`). MR32
only affects the selector routing probability distribution, not the signal generated
when MeanReversion is selected.

---

## 6. Comparison Against Documented Architecture

### G3 documented requirements

From `docs/architecture/MPML_Architecture_Roadmap.md`:

> **The default invocation, without explicit strategy selection, MUST preserve the
> existing benchmark evaluation scope exactly.**

> Evaluation Scope determines which strategies participate in a particular
> experiment.

### Compliance analysis

| Requirement | Status |
|---|---|
| Evaluation Scope constrains which strategies are **evaluated** in the walk-forward | ✅ Satisfied — `_run_full_universe` flag and `filter_strategy_specs` correctly gate strategy execution |
| Default invocation uses `phaseaware_default` → TF4 + MR42 | ✅ Satisfied — `resolve_evaluation_scope` correctly resolves to TF4 + MR42 |
| Explicit `--strategy` MUST affect only the set of strategies evaluated | ✅ Satisfied — explicit path skips full-universe backtests and selector training |
| Default invocation MUST preserve existing benchmark evaluation scope **exactly** | ❌ **Violated** — the selector training uses `_DEFAULT_EVALUATED_MR_STRATEGY_IDS` (MR1, MR2, MR32, MR42, MR5), which is not bounded by the evaluation scope. Any implementation change to any of these strategies can silently change the benchmark |

The documented three-way separation:

> **Strategy Capability ≠ Behavioral Surface / State ≠ Evaluation Scope**

is correctly applied at the **inference** level (which strategies execute during
walk-forward). It is **not** applied at the **selector training** level. The
selector training uses a broader, undocumented "reference universe" that is larger
than the evaluation scope and is not governed by G3.

---

## 7. Why the Explicit `--strategy TF1` Path Is Unaffected

When `--strategy TF1` is passed:

1. `resolve_evaluation_scope()` → `EvaluationScope(strategy_ids=("TF1",), source="explicit")`
2. `should_run_full_universe_backtests(scope)` → `False` (`scope.source == "explicit"`)
3. `_run_full_universe = False`
4. **Section [4/5]** (full-universe backtests): **skipped** — `all_pair_results = {}`,
   `hardcoded_results = {}`
5. **Section [4b/5]** (global selector training): **skipped** — `print('Skipped:
   selector training is not required for explicit scope ...')`
6. Walk-forward per-fold: `pair_results_full = _build_selector_reference_results(...)`
   — **correctly uses only TF4+MR42 (the policy pair)**, MR32 is never included
7. `strategy_execution_timeline` and `strategy_trades` CSVs are byte-identical across
   MR32 changes because MR32 never enters this path

The explicit-strategy path is **correctly isolated** from the MR32 coupling because
`_run_full_universe=False` gates out the entire full-universe backtest and selector
training pipeline.

---

## Root-Cause Classification

**Category: Architectural coupling (undocumented design decision that creates a
hidden dependency).**

This is not a regression in the G3 implementation. The coupling predates G3. The
`_DEFAULT_EVALUATED_MR_STRATEGY_IDS` constant was defined before `EvaluationScope`
existed and the G3 gating logic correctly prevents explicit-scope runs from being
affected. However, the G3 machinery does not extend its scope-bounding to the
selector training reference universe, which remains unbounded.

Whether this coupling is *intentional* cannot be determined from the code alone:

- **Possible intended interpretation**: Using all 5 MR strategies as "diverse
  experts" for selector training improves the quality of the MeanReversion label
  (the best-performing MR strategy labels the bar, regardless of which specific MR
  strategy is ultimately executed). This is a valid "mixture-of-experts" training
  philosophy.
- **Problem with that interpretation**: It is nowhere documented. It creates a hidden
  dependency that violates the documented guarantee of benchmark stability. Changing
  any strategy in the reference universe — even one not in the evaluation scope —
  silently changes the benchmark.

---

## Exact Files and Functions Responsible

| File | Function / Constant | Role |
|---|---|---|
| `src/strategies.py:20` | `_DEFAULT_EVALUATED_MR_STRATEGY_IDS` | Defines full MR evaluation universe (includes MR32) |
| `src/strategies.py:85` | `instantiate_evaluated_strategy_dicts()` | Instantiates all strategies in the full universe |
| `src/strategies.py:1590` | `run_backtests()` | Runs all strategies in full universe; produces `hardcoded_results` |
| `main.py:~3009` | Section [4/5] | Calls `run_backtests()` unconditionally for default runs |
| `src/models.py:1430` | `StrategyPerformanceTracker.compute_strategy_returns()` | Generates per-bar `best_strategy` labels from full universe results |
| `src/models.py:1581` | `StrategySelector._strategy_to_category()` | Maps "MR32" → "MeanReversion" |
| `main.py:~3110` | Section [4b/5] | Trains global selector on `hardcoded_results` (full universe) |
| `main.py:3551–3558` | Walk-forward per-fold selector | Uses `hardcoded_results` (full universe) when `_run_full_universe=True` |
| `main.py:1976` | `_build_selector_reference_results()` | **Correctly scoped** (TF4+MR42 only) but only called on explicit path |

---

## Recommended Next Investigation / Fix Direction

1. **Define a "selector reference universe" constant** separate from
   `_DEFAULT_EVALUATED_MR_STRATEGY_IDS`. Currently there is only one constant,
   shared between the legacy reporting universe and the selector training reference.
   These should be explicitly separated so that the selector training reference can
   be independently bounded.

2. **Evaluate two possible fixes** (do not implement until design is agreed):
   - **Option A (strict scoping)**: Restrict selector training to the evaluation
     policy strategies (TF4 + MR42 + PhaseAware). Use
     `_build_selector_reference_results()` on the default path as well (currently it
     is used only on the explicit path). This would make the selector training scope
     match the documented evaluation scope exactly, but may reduce selector training
     label diversity.
   - **Option B (documented wide reference)**: Explicitly document that selector
     training uses the full MR/TF universe as a "diverse expert pool", enumerate
     which strategies are included in the reference universe, and specify that changes
     to any of these strategies can affect the benchmark. This legitimizes the current
     behavior but requires adding documentation and potentially pinning the reference
     strategies.

3. **Verify** that the walk-forward per-fold causal selector training (section [4b/5]
   per-fold loop) is the dominant driver versus the global selector (section [4b/5]
   pre-WF loop). The two selector training paths must be aligned on whatever universe
   is chosen.

4. **Add a regression test** that asserts selector training label distribution does
   not change when only the MR32 implementation is modified (or conversely, document
   the expectation that it will change).

---

## Selector Reference Universe Design Audit

> **Follow-up design audit.** This section was added after the initial forensic
> audit above. Its purpose is to establish the intended selector reference universe
> from the existing architecture and code, so that the next PR can implement a
> well-defined fix.

### Conceptual Distinction

Three concepts must be held separate throughout this analysis:

**A. Strategy Registry**

The complete set of available strategy implementations and their capabilities.
Currently contains (at minimum): TF1–TF5, MR1, MR2, MR32, MR42, MR5.
This is the authority for strategy identity. It does not determine which strategies
run in any given experiment.
Defined via `get_default_strategy_registry()` / `src/strategy_registry.py`.

**B. Evaluation Scope**

The strategies actually evaluated in a particular experiment.
For the default policy this resolves to `phaseaware_default` → `("TF4", "MR42")`.
Governed by G3 / `resolve_evaluation_scope()` / `EvaluationScope`.
The `_run_full_universe` flag gates large pipeline sections based on this.

**C. Selector Reference / Training Universe**

The strategies whose performance is used to generate `best_strategy` labels, from
which the strategy-type selector (`StrategySelector`) is trained.
This concept is **currently unnamed and undocumented** in the architecture.
It is implicitly defined by whichever `strategy_results` dict is passed to
`StrategyPerformanceTracker.compute_strategy_returns()`.

On the default path today, (C) equals the full research universe (all 10 strategies)
because `hardcoded_results` — which is produced by running every strategy in
`_DEFAULT_EVALUATED_MR_STRATEGY_IDS` and `_DEFAULT_EVALUATED_TF_STRATEGY_IDS` — is
fed directly into the selector trainer without narrowing.

These three concepts are **not** the same and must not be collapsed. The audit
question is: what should (C) be?

---

### What the Selector Actually Learns

Tracing the pipeline from labels to inference:

```
StrategyPerformanceTracker.compute_strategy_returns()
    per bar: look ahead window_days bars
    best_strategy = argmax over all strategies in strategy_results dict
         ↓
StrategySelector._strategy_to_category(best_strategy)
    "TF*"          → "TrendFollowing"
    "MR*"          → "MeanReversion"
    "PhaseAware*"  → "PhaseAware"
         ↓
StrategySelector.train(training_data)
    LabelEncoder on {"TrendFollowing", "MeanReversion", "PhaseAware"}
    XGBoost 3-class classifier: market state → strategy TYPE
    (comment in code: "3-class problem, much more learnable than 31-class")
         ↓
StrategySelector.predict() / predict_proba()
    returns: "TrendFollowing" | "MeanReversion" | "PhaseAware"
         ↓
StrategySelector_Dynamic.generate_signals()
    "TrendFollowing" → execute self.tf_strategies[self.default_tf]  (= TF4)
    "MeanReversion"  → execute self.mr_strategies[self.default_mr]  (= MR42)
    "PhaseAware"     → execute PhaseAwareStrategy(TF4, MR42)
```

**Key finding:** The selector is **not** learning which individual strategy is best.
It is learning which **strategy type** (TrendFollowing / MeanReversion / PhaseAware)
will perform best in the current market regime. At inference time, the selected type
is executed using fixed concrete representatives: TF4 for TrendFollowing, MR42 for
MeanReversion, and PhaseAware(TF4, MR42) for PhaseAware.

This is confirmed by:

- `StrategySelector` docstring: `"Predicts: TrendFollowing vs MeanReversion vs PhaseAware (3-class problem)"`
- `_strategy_to_category()`: collapses any `"MR*"` name, regardless of version, to `"MeanReversion"`
- `StrategySelector_Dynamic.__init__`: resolves `default_tf` and `default_mr` from the policy (`TF4`, `MR42`)
- The executed signals use `self.tf_strategies[self.default_tf]` and `self.mr_strategies[self.default_mr]`

**Implication for the reference universe:** Since the selector maps all MR strategies
to the same class label and then always executes MR42 at inference, the *identity* of
the MR strategy used for training labels directly determines when the classifier
decides "MeanReversion is the best regime". If MR32 wins a bar that MR42 would not
have won, the bar is still labeled "MeanReversion" — but the selector will then
execute MR42, not MR32, on that bar at test time. The label is therefore **not a
faithful oracle** for MR42's future performance. The semantically correct label
source for the MeanReversion class is the strategy that will actually be executed
when the selector picks "MeanReversion" — which is MR42.

---

### Candidate Reference Universe

The candidate proposed by the problem statement is:

```
TF4
MR42
PhaseAware(TF4, MR42)
```

This is also exactly what `_build_selector_reference_results()` currently produces
(see Section 2 of the forensic audit above).

Evaluating this candidate:

| Criterion | Assessment |
|---|---|
| One representative per selector output class? | ✅ TF4 → TrendFollowing; MR42 → MeanReversion; PhaseAware_TF4_MR42 → PhaseAware |
| Matches the strategies actually executed at inference? | ✅ Yes — `StrategySelector_Dynamic` executes exactly TF4, MR42, PhaseAware(TF4,MR42) |
| Labels are faithful oracles for inference-time performance? | ✅ Yes — the label "MeanReversion" on bar i means MR42 was the best over the next N bars, which is exactly what the selector will execute when it selects MeanReversion |
| Consistent with `phaseaware_default` policy? | ✅ Directly derived from the policy pair |
| Avoids hidden dependency on unrelated strategies? | ✅ MR32, MR1, MR2, MR5, TF1–TF3, TF5 do not participate |
| Semantically correct causal training? | ✅ Each label corresponds to the strategy that would actually run in production |
| Preserves historical benchmark? | ✅ When TF4 and MR42 implementations are unchanged, labels and selector are unchanged |

No alternative candidate is supported by the architecture. A wider reference
universe (multiple MR strategies) creates the label-execution mismatch described
above: the selector learns when "some MR strategy performs well" but always executes
MR42, so the training signal is polluted by strategies that are never used at
inference.

---

### Assessment of TF4 + MR42 + PhaseAware(TF4, MR42)

**This is the correct selector reference universe.** The evidence is:

1. **It is what the code already calls for on the explicit-strategy path.**
   `_build_selector_reference_results()` — introduced specifically to build "the
   minimal full-history results needed for selector labels" — produces exactly this
   set. Its docstring and implementation already encode the correct design; the
   function was simply never wired into the default path.

2. **It matches inference semantics exactly.** The selector's three output classes
   map one-to-one to these three concrete strategies. Training labels derived from
   this set are faithful oracles for what the selector will actually execute.

3. **It is derived from the policy, not from a separate constant.** The function uses
   `resolve_phaseaware_strategy_pair(policy_id)` to obtain TF4 and MR42 dynamically.
   If the policy ever changes, the reference universe updates automatically.

4. **It satisfies the G3 invariant.** Changing any strategy outside this set
   (MR32, MR1, etc.) cannot affect the default selector, preserving the documented
   requirement that "the default invocation MUST preserve the existing benchmark
   evaluation scope exactly."

5. **The global selector (section [4b/5]) and the per-fold causal selector currently
   disagree.** On the default path:
   - Global selector uses `hardcoded_results` (full universe, 10 strategies)
   - Per-fold causal selector also uses `hardcoded_results` (full universe)
   - Explicit-strategy path: per-fold uses `_build_selector_reference_results()` (3 strategies)
   
   There is a semantic inconsistency between the default and explicit-strategy paths
   at the per-fold level, which the proposed fix would eliminate.

---

### Global vs Causal Walk-Forward Training

The two selector training paths currently exist in `main.py`:

**Path 1 — Global selector (`[4b/5]`, ~line 3110):**

```python
pair_backtest = hardcoded_results.get(pair_name, {})   # full universe today
tracker = StrategyPerformanceTracker(window_days=20)
training_data = tracker.compute_strategy_returns(df, pair_backtest)
selector = StrategySelector(...)
selector.train(training_data, ...)
selector_trained[pair_name] = selector
```

This trains a global selector using the full historical dataset (no fold boundary).
It is used for the DL-pipeline diagnostic run and for the standalone DL-enabled
evaluation path.

**Path 2 — Per-fold causal selector (walk-forward loop, ~line 3611):**

```python
pair_results_full = hardcoded_results.get(pair_name, {})   # full universe today
training_data, _ = _build_causal_selector_training_data(
    pair_results_full=pair_results_full, ...)
selector.train(training_data, ...)
```

This trains a per-fold selector using only fold-bounded training data (no leakage).
It is the actual selector used for walk-forward evaluation results.

**Both paths currently use the full strategy universe.** They share the same
semantic problem: the training labels are generated from strategies that are never
executed at inference time.

**Should they use the same reference universe?** Yes. The reference universe
conceptually answers the question "which expert should I route to?" — that question
has the same answer regardless of whether we are training a global selector or a
fold-local selector. The only difference between the two paths is the *window* of
data used (full history vs fold-bounded training slice), not the *set of experts*
whose performance is observed.

A partial fix that corrects only one path and not the other would introduce a new
inconsistency: the global selector and the fold selector would disagree on what
"MeanReversion" means, leading to different routing behaviors.

**Recommended architecture:**

```
selector_reference_results = _build_selector_reference_results(
    df_full=df_full,
    pair_name=pair_name,
    policy_id=DEFAULT_PHASEAWARE_POLICY_ID,
)
```

Both paths should compute their training labels from `selector_reference_results`
(TF4 + MR42 + PhaseAware), not from `hardcoded_results` (full universe).

---

### Backward-Compatibility Implications

The empirical facts established by the controlled experiment:

1. Restoring legacy MR32 while keeping current architecture → historical benchmark
   reproduced byte-for-byte.
2. New MR32 implementation → all 14 pairs change.
3. `results_ml_backtest__dl_enabled.csv` is unchanged across MR32 variants.
4. `results_per_pair__dl_enabled.csv` differs (MR32 row changes).
5. `selector_state_timeline__dl_enabled.csv` changes in ~13,380 of 60,608 observations.
6. Explicit `--strategy TF1` artifacts are byte-identical across MR32 variants.

**What the proposed fix achieves with respect to these facts:**

| Artifact | Current behavior with new MR32 | After fix |
|---|---|---|
| `results_per_pair` | MR32 row changes (expected — MR32 ran) | MR32 row still changes (full universe still evaluated for research) |
| `selector_state_timeline` | Changes due to MR32 contamination | **Stable** — selector trained only on TF4+MR42+PhaseAware |
| `walkforward_results_per_pair` | Changes due to contaminated selector | **Stable** — only TF4/MR42 implementation changes affect selector |
| Explicit `--strategy TF1` | Already isolated; unaffected | Unchanged |
| `results_ml_backtest` | Already unchanged; unaffected | Unchanged |

**The fix preserves** all of the following:
- Full strategy universe evaluation (MR32 still runs; `results_per_pair` still contains it)
- Individual strategy research / standalone strategy experiments
- Behavioral-surface functionality (conditioned on the same policy pair)
- Historical default benchmark: with legacy MR32 restored and the fix applied, the
  selector is now derived exclusively from TF4 and MR42; if both those strategies'
  implementations are unchanged, the selector is byte-identical to the pre-MR32-change
  baseline

**The fix does change the default walk-forward result** compared to the current state
(which uses new MR32), because it removes the MR32 contamination. However, this is
the *desired* behavior: the result with the fix applied and legacy MR32 restored
should be byte-identical to the original pre-experiment baseline. The fix + legacy
MR32 restore together achieve full reproducibility.

---

### Recommended Architecture

The clean architecture separates two distinct data flows:

```
Full Strategy Universe
    (TF1–TF5, MR1, MR2, MR32, MR42, MR5)
         │
         ├── run_backtests() → hardcoded_results
         │       │
         │       ├── results_per_pair (research artifact)
         │       ├── legacy aggregation / reporting
         │       └── explicit standalone strategy research
         │
         └── (NOT used for selector training)


Selector Reference Universe
    (TF4, MR42, PhaseAware(TF4, MR42))
         │
         └── _build_selector_reference_results()
                 │
                 ├── [4b/5] global selector training
                 │       StrategyPerformanceTracker
                 │       StrategySelector.train()
                 │       → selector_trained
                 │
                 └── walk-forward per-fold:
                         _build_causal_selector_training_data()
                         StrategySelector.train()
                         → fold selector
```

`_build_selector_reference_results()` becomes the **canonical** mechanism for
constructing selector training data. It is called:
- Once per pair before the global selector training
- Once per pair (full history) before the walk-forward fold loop, with the fold
  slicing applied afterward inside `_build_causal_selector_training_data()`

`hardcoded_results` continues to be built (unchanged) for legacy reporting and
individual strategy research. It is **not** passed to any selector training call.

---

### Recommended PR Scope

**What the next PR should do:**

1. **Add a module-level constant** (or derive it at runtime via
   `resolve_phaseaware_strategy_pair`) that names the selector reference universe.
   This makes the design explicit and auditable. Suggested location:
   `src/strategies.py` alongside `_DEFAULT_EVALUATED_MR_STRATEGY_IDS`, or in
   `src/strategy_registry.py` since it is policy-derived.

2. **Wire `_build_selector_reference_results()` into the default path** for both
   selector training locations:
   - Section [4b/5] global selector: replace `pair_backtest = hardcoded_results.get(...)`
     with `pair_backtest = _build_selector_reference_results(...)` (or a cached
     equivalent computed once per pair)
   - Walk-forward per-fold: the existing `if _run_full_universe:` branch already
     falls through to `hardcoded_results`; it should instead use
     `_build_selector_reference_results()` regardless of `_run_full_universe`

3. **Do not remove or reduce `hardcoded_results` / `run_backtests()`** — the full
   strategy universe must continue to be evaluated for research artifacts and
   `results_per_pair`.

4. **Document the separation** in `src/strategies.py` and/or
   `docs/architecture/selector_architecture.md`:
   - `_DEFAULT_EVALUATED_MR_STRATEGY_IDS` is the research/reporting universe
   - The selector reference universe is the policy pair (TF4 + MR42 + PhaseAware),
     derived from `resolve_phaseaware_strategy_pair()`

5. **Do not change** `_build_selector_reference_results()` itself — it already has
   the correct semantics.

6. **Do not change** `run_backtests()`, `instantiate_evaluated_strategy_dicts()`,
   or `_DEFAULT_EVALUATED_MR_STRATEGY_IDS` — these serve the research/reporting
   purpose and must remain intact.

---

### Required Regression Tests

The following tests should be added or verified to prevent future regressions:

**Test 1 — Selector reference universe is independent of non-policy strategies.**

Assert that `_build_selector_reference_results()` returns exactly the three keys:
`{TF4, MR42, PhaseAware_TF4_MR42}` regardless of the current MR32 implementation.
This can be a unit test on `_build_selector_reference_results()`.

**Test 2 — Default selector training does not include non-policy strategies.**

Assert that the `strategy_names` iterated inside
`StrategyPerformanceTracker.compute_strategy_returns()` during default-path selector
training are exactly `{TF4, MR42, PhaseAware_TF4_MR42}`. This should be validated
via the mock-data integration test for the selector training pipeline.

**Test 3 — Selector label distribution stability.**

Assert that changing the MR32 implementation (mock) does not change the selector
training label distribution (counts of TrendFollowing / MeanReversion / PhaseAware).
This is the direct regression test for the root cause identified in this audit.

**Test 4 — Global and per-fold selector use the same reference universe.**

Assert that the `pair_results_full` dict passed to `_build_causal_selector_training_data()`
and the `pair_backtest` dict passed to the global selector trainer each contain
exactly the same set of strategy keys for a given pair. This prevents the partial-fix
inconsistency described in the Global vs Causal section above.

**Test 5 — `results_per_pair` continues to include all research strategies.**

Assert that `hardcoded_results` (and the resulting `results_per_pair.csv`) still
contains MR32, MR1, MR2, MR5, TF1–TF3, TF5 after the fix, confirming that the
full research universe was not accidentally removed.
