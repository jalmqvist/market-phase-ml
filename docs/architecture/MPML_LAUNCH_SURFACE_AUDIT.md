# MPML Launch-Surface Audit

This is a read-only audit of MPML’s launch surface at the revision listed below. It records observed behavior and technical constraints. Implementation recommendations in this historical audit are not authoritative; use `docs/architecture/MPML_LAUNCH_SURFACE_REFACTOR.md` for the approved PR 1–3 scope, sequence, and acceptance criteria.

**Audit basis:** `jalmqvist/market-phase-ml`, `main`, commit `ef6b1b84c7393a1bb4dbd6f92a1442115c6cbb57`. This was a read-only audit; no repository code was changed. The MSML companion notes below were not re-audited.

## Summary

> **Status:** Audit evidence, not an implementation plan. Statements such as “prefer” and “consider” below record options considered during the audit; they do not override the approved three-PR design in `MPML_LAUNCH_SURFACE_REFACTOR.md`. In particular, injectable configuration and artifact-aware preflight are deferred.


The refactor is feasible, but it is larger than moving the parser and collecting environment variables. `main.py` has import-time cache deletion, configuration reads and output; several parser choices are resolved dynamically; registry construction imports `src.strategies` and therefore pandas; and current environment reads are split across import-time constants and resolver functions that read `os.environ` directly. A validator that accepts an injected `env` mapping will not be authoritative until those production reads share a configuration snapshot or accept explicit inputs.

The highest-value first targets remain parser characterization, side-effect isolation, and a single launch-resolution path. Keep “validate” distinct from “run”: artifact resolution currently reads and processes prediction data, and some proposed errors (missing DL artifact and unknown pairs) would change behavior if made fatal.

## 1. Verified CLI surface

The parser is inline in `_run_cli()` (`main.py:5206-5351`). It currently exposes:

| Argument | Current parse behavior | Later validation / effect |
| --- | --- | --- |
| `--output-dir` | `Path`, default `None` | Selected ahead of `MPML_OUTPUT_DIR`; directory is created with `exist_ok=False` in `_set_run_output_dir` (`main.py:257-265, 2656-2677`). |
| `--experiment-generation` | choices from `_VALID_EXPERIMENT_GENERATIONS`, default `None` | Normalized and compared with the generation implied by the selected variant in `main()`; mismatch raises `ValueError` (`main.py:2637-2649`). |
| `--experiment-variant` | choices from `VALID_EXPERIMENT_VARIANTS`, default `None` | Precedence is CLI, `EXPERIMENT_VARIANT`, then `"A"` (`main.py:2573-2579`). |
| `--experiment-seed` | `int`, default `None` | Precedence is CLI, `EXPERIMENT_SEED`, then `42`; malformed nonempty env input raises `ValueError` (`src/repro.py:34-48`). |
| `--behavioral-surface` | registry-derived choices, default `None` | Precedence is CLI, `BEHAVIORAL_SURFACE`, then `"trend_vol"`; `main()` also checks membership (`main.py:2482-2495`). Current built-in surfaces include `persistent`, `reactive_jpy`, and `trend_vol`; the preliminary sample omits `persistent`. |
| `--recommendations-per-pair` | `store_true`, default `False` | Enables per-pair recommendation output from walk-forward evidence (`main.py:4587-4597`). |
| `--recommendation-top-n` | `int`, default `None` | Positive-value check occurs later in recommendation construction, not in argparse (`src/recommendation.py:260-264, 473-496`). |
| `--strategy` | repeatable `append`, default `None` | Strategy IDs are resolved against the registry; duplicates are de-duplicated in first-seen order (`src/evaluation_scope.py:62-132`). |
| `--phaseaware-tf` | repeatable `append`, default `None` | `_run_cli()` rejects more than one; the resolver requires a registered `TrendFollowing` ID (`main.py:5333-5336`, `src/strategy_registry.py:482-515`). |
| `--phaseaware-mr` | repeatable `append`, default `None` | `_run_cli()` rejects more than one; the resolver requires a registered `MeanReversion` ID. |
| `--debug` | `store_true`, default `False` | Sets diagnostic verbosity when `main()` starts. |
| `--export-dl-conditional-portfolio-timeline` | `store_true`, default `False` | If the dynamic selector is inactive for the scope, raises only after walk-forward execution reaches the export point (`main.py:4543-4549`). |

Two corrections to the preliminary text:

- `MR3` is a valid current strategy ID; `MR4` is not. Current registry IDs are `TF1`–`TF5`, `MR1`, `MR2`, `MR3`, `MR32`, `MR42`, and `MR5` (`src/strategy_registry.py:270-415`).
- The parser already has a late positive-value check for `--recommendation-top-n`; it should be included in launch-rule coverage if the intended validator promises to catch all launch errors.

## 2. Import behavior and parser extraction

`main.py` imports matplotlib, pandas, and `src.models` at module import; `src.models` imports xgboost. It also calls `clear_cache()` and prints the configured date range during import (`main.py:1-17, 120-126, 334-339`). Extracting the parser is therefore a sound boundary.

The registry imports themselves are relatively light, but **building the parser is not necessarily light**: the `--strategy` help calls `get_default_strategy_registry().available()`. Registry construction imports strategy implementations from `src.strategies`, which imports pandas (`src/strategy_registry.py:269-274, 418-420; src/strategies.py:1-13`). `behavioral_registry.available()` uses its built-in registry. Thus:

- A module-only `import mpml.cli` test can pass if `cli.py` has no eager imports of `main` or registries.
- A test that also calls `build_parser()` should expect pandas to be imported unless registry IDs are provided through a genuinely light IDs-only source or the parser avoids eagerly resolving them.
- Importing `src.strategy_registry` and `mpml.behavioral` alone did not load pandas, matplotlib, or xgboost in the audit subprocess; the dynamic strategy-registry construction path is the dependency edge to test.

Preserve the exact argument flags, `dest`, action, defaults, help, types, and choices. In particular, keep `--output-dir` as `Path`; record dynamic choice sources instead of substituting a static list.

## 3. Cache and output-directory side effects

`clear_cache()` is not a general data-cache cleaner. It creates `data/cache` if needed and deletes only matching `*.pkl` files in that directory (`src/cache.py:9-11, 71-83`). It does **not** delete broker CSV caches such as `data/raw_broker_csv` or `data/processed_broker_csv`; those are separate directories created by `MarketDataPipeline` (`src/data.py:164-185`).

No pilot script containing the stated `rm -rf data/raw_broker_csv` / `processed_broker_csv` commands exists in the audited branch. The checked-in `run_v5_full_matrix.sh` has no such cleanup. The external pilot cannot be assessed from this repository snapshot; do not remove its cleanup based on `clear_cache()`.

Moving `clear_cache()` into `main()` removes import-time deletion, but it also means each direct `main()` invocation clears caches (whereas today importing `main.py` clears once per process). Verify the intended call semantics and tests. A CLI-only wrapper could instead clear once for actual runs while keeping library/test imports safe.

The existing output-directory collision check is performed after `MarketDataPipeline` is constructed. Construction creates the source-specific raw and processed cache directories before `_set_run_output_dir()` attempts the exclusive mkdir. A preflight check can report an existing output path earlier, but the exclusive mkdir must remain in the run path to handle races. The check cannot guarantee writability or parent-directory validity without attempting filesystem operations.

## 4. Launch rules and the reusable preflight boundary

Current rules and their real enforcement points:

| Rule | Current behavior |
| --- | --- |
| Repeated `--phaseaware-tf` / `--phaseaware-mr` | Rejected by `_run_cli()` after parsing (`main.py:5333-5336`). |
| Unknown behavioral surface | Rejected by argparse choices and checked again in `main()` (`main.py:5243-5254, 2487-2495`). |
| Unknown `--strategy` ID | Rejected by `resolve_evaluation_scope()` when an explicit strategy scope is used (`src/evaluation_scope.py:112-125`). |
| PhaseAware override family / unknown ID | Rejected by `resolve_phaseaware_configuration()` (`src/strategy_registry.py:482-515`). |
| Variant/generation mismatch | Rejected in `main()` before market-data pipeline construction (`main.py:2637-2649`). |
| Existing run output directory | Rejected by `_set_run_output_dir()` after pipeline construction; `FileExistsError` becomes `ValueError` (`main.py:257-265, 2656-2677`). |
| Timeline export without dynamic selector | `RuntimeError` is raised at the end of the walk-forward export section (`main.py:4543-4549`). |
| Non-positive `--recommendation-top-n` | Rejected during recommendation construction, late in the run (`src/recommendation.py:260-264, 473-496`). |
| Empty walk-forward strategy-spec set | Intended `ValueError` message references undefined `_scope_tf` and `_scope_mr`; this path currently raises `NameError` while formatting the message (`main.py:4570-4577`). Keep the fix separate if preserving behavior is a strict requirement. |

The timeline rule is suitable for genuine preflight: after resolving the scope and PhaseAware composition, `_build_walkforward_execution_plan()` only builds specs and execution flags; it does not load market data (`main.py:1615-1684`). The dynamic selector spec requires the effective PhaseAware TF/MR pair to be in scope. Test default scope, that pair together, either representative alone, and unrelated explicit IDs.

Do not implement `validate_launch(argv, env)` as a parallel set of independent rule logic. Several production functions still read process state directly (`resolve_experiment_seed`, `resolve_market_data_source`, the artifact resolver, and main-module constants). Passing a separate mapping to a validator would not make those calls use that mapping. Prefer a resolved launch/config object built once from injected inputs, and have both preflight and execution consume the same resolved values. Keep the final filesystem mkdir as an atomic run-time guard.

Also avoid promising that every issue can always be collected: argparse normally stops at the first syntax error, and production resolvers can have different failure/side-effect characteristics. Structured issue aggregation is still useful for independent, pure checks.

## 5. Environment-variable inventory and behavior

The launch surface is larger than the variables read directly in `main.py`. The table below includes production reads found in MPML source; tests and the `PYTHONHASHSEED` write are excluded. “Import-time” means the current module snapshots the value during import, so changing `os.environ` afterward does not update that value.

| Variable(s) | Current source / semantics |
| --- | --- |
| `DL_SIGNALS_ENABLED` | Read at import by both `main.py` and `src/dl_config.py`; only `.lower() == "true"` enables it. `"1"`, `"yes"`, `"on"`, empty, and garbage are false. |
| `DL_MODEL`, `DL_FEATURE_SET`, `DL_REGIME` | Read at import by `main.py`; defaults are `"mlp"`, `"price_trend"`, and `""`. If DL is enabled, an explicit artifact path is set, and `DL_FEATURE_SET` is unset/blank, feature set may be inferred from the artifact filename. |
| `MIN_DL_TRAIN_COVERAGE_PCT` | `main.py` import-time `float`, default `"5.0"`; invalid numeric values fail during import. |
| `ACTIVE_PAIRS` | Read and normalized at `main.py` import; comma-separated short pair names (`EURUSD`, not `EURUSD=X`). Empty means all pairs. The post-load filter preserves metadata and silently drops unrecognized names; unknown names are not currently errors. Artifact-overlap reporting and experiment-surface family inference read this env again at runtime. |
| `EXPORT_SELECTOR_STATE_TIMELINE` | Import-time bool in `main.py`; accepts case-insensitive `1/true/yes/on`; other values are false. |
| `MPML_START_DATE`, `MPML_END_DATE` | Import-time strings in `main.py`; defaults `2005-01-01` and `2026-04-01`; import prints both values. |
| `BEHAVIORAL_SURFACE` | Runtime fallback in `main()`, default `"trend_vol"`; CLI wins. |
| `DL_PREDICTION_ARTIFACT_PATH`, `DL_SIGNALS_CUBE_PATH` | `main.py` reads the primary path at runtime. `src/dl_config.py` snapshots path configuration at import and uses `DL_SIGNALS_CUBE_PATH` only if the primary variable is **unset** (an empty primary value suppresses the alias); otherwise it defaults to a sibling-repository artifact directory. |
| `BEHAVIORAL_STATE_ID` | Read at runtime by `main.py`; empty becomes `None`. `src/dl_config.py` also snapshots it at import for canonical-identity helpers. |
| `EXPERIMENT_VARIANT`, `EXPERIMENT_GENERATION` | Runtime fallbacks in `main()`; variant defaults to `"A"`, generation has no default hint. Generation must match variant semantics when supplied. |
| `EXPERIMENT_SEED` | Read at runtime by `resolve_experiment_seed()`; stripped then parsed as integer. Empty uses default `42`; invalid nonempty input raises. CLI seed wins. |
| `MSML_REGIME` | Runtime fallback in `main()`; normalized to uppercase. Fallback varies with the selected behavioral surface and resolved state. |
| `OVERLAP_ONLY` | Runtime bool in `main()`; accepts case-insensitive `1/true/yes/on`; otherwise false. |
| `MPML_OUTPUT_DIR`, `MPML_RUNS_ROOT` | Runtime output selection. CLI `--output-dir` wins over `MPML_OUTPUT_DIR`; absent both, output is under `MPML_RUNS_ROOT` (default run root). `MPML_OUTPUT_DIR` also supplies the run-name hint used by imputation-awareness inference. |
| `MPML_AUTO_ANALYZE` | Read near the end of a run; defaults true and accepts case-insensitive `1/true/yes/on`. |
| `MPML_DATA_SOURCE` | Read by `resolve_market_data_source()`; trimmed/lowercased; default `yfinance`; valid values `yfinance` and `broker_csv`; invalid values raise. |
| `MPML_BROKER_DATA_DIR` | Read when constructing a broker-CSV pipeline; trimmed; overrides the default sibling-repository input directory. It selects broker input data, not the MPML CSV cache directory. |
| `DL_SURFACE_MODEL`, `DL_SURFACE_TARGET_HORIZON`, `DL_SURFACE_FEATURE_SET`, `DL_SURFACE_REGIME` | Import-time legacy selector values in `src/dl_config.py`, defaults `lstm`, `24`, `price_trend`, `HVTF`; target horizon is parsed as `int`. These are distinct from main’s `DL_MODEL` / `DL_FEATURE_SET` / `DL_REGIME`. |
| `BEHAVIORAL_SURFACE_ID`, `BEHAVIORAL_SURFACE_VERSION` | Import-time canonical-identity values in `src/dl_config.py`; exposed through `get_canonical_behavioral_identity()` when all three identity fields are nonempty. |
| `MPML_SENTIMENT_SURFACE`, `MPML_TRAINING_PAIR_FAMILY`, `MPML_EVALUATION_PAIR_FAMILY`, `MPML_FEATURE_SURFACE`, `MPML_ARTIFACT_SOURCE` | Runtime fallbacks used to assemble experiment-surface metadata (`src/experiment_surface_runtime.py`). Bool/string decoding and metadata precedence are specific to that module. |
| `MPML_SELECTOR_TRACE_PATH`, `MPML_SELECTOR_TRACE_PAIR`, `MPML_SELECTOR_TRACE_TIMESTAMP` | Optional selector trace instrumentation in `src/strategies.py`; not ordinary experiment-selection options, but document them as diagnostic env inputs if the spec claims complete repository coverage. |
| `MPML_REVISION` | Read by `mpml/runtime/phaseaware.py` for revision/provenance handling; classify as runtime metadata, not a launch selector. |

Important behavior corrections for the proposed validator:

- `DL_SIGNALS_ENABLED=true` does **not** currently require an explicit artifact path. The resolver can discover a configured/default artifact; if none resolves, it returns a disabled runtime with diagnostics rather than raising. An explicit missing path also disables the runtime. Making either case an error changes behavior and belongs in a behavior-change proposal.
- `ACTIVE_PAIRS` names are the short keys used by `filter_pair_universe`; `src.data.ALL_PAIRS` contains Yahoo ticker strings. Compare configured values against `PAIR_NAMES` values, not `ALL_PAIRS`. Rejecting unknown names would change current silent-filter behavior. The pipeline is constructed and data loading happens before the pair filter is applied.
- The proposal’s `DL_MODEL`-versus-filename-prefix warning is not an established runtime rule. The artifact loader selects by artifact contents/selector; do not report a filename mismatch until the convention and actual loader behavior are verified.
- The output-name inference is real: `infer_imputation_awareness_from_name()` can override the `missing_indicators_enabled` experiment factor. Its hint is the explicit output directory or `MPML_OUTPUT_DIR`; the generated `MPML_RUNS_ROOT` default path is not passed as that hint (`main.py:2596-2615`). Restrict any warning to the paths that actually feed the inference.

A declarative registry should preserve each variable’s source, timing, parser, default, alias precedence, and scope. Moving import-time reads into an injected runtime config is not merely table extraction; characterize the current snapshots and preserve precedence before changing call sites. Do not silently normalize the strict boolean or pair-filter quirks in a behavior-preserving phase.

## 6. Preflight and `--validate-only`

Preflight can safely cover parser errors, registry membership, PhaseAware role constraints, generation/variant consistency, the walk-forward execution plan, positive recommendation limits, and an existing output path—provided the same resolved configuration is used by execution.

The preliminary `--validate-only` description needs to be narrowed:

- `resolve_behavioral_artifact_runtime()` is not metadata-only. With DL enabled and an artifact found, it calls the artifact loader, materializes prediction data, checks for compatible rows, and computes D1 features (`src/behavioral_artifact_resolver.py:34-143`). It may be expensive and loads prediction data even though it does not load market OHLC data.
- A missing/unresolvable artifact currently disables DL rather than failing. `artifact_exists: true/false` alone is not enough to describe the actual runtime; report `enabled`, resolved path/state, and diagnostics.
- A validate-only mode must not call `clear_cache()`, construct `MarketDataPipeline` (its constructor creates cache directories), or `_set_run_output_dir()`. Since importing `main.py` currently clears caches, this requires the import-side-effect change first.
- Checking `Path.exists()` is advisory and race-prone. Keep the exclusive mkdir at real-run time.

Consider separating a fast, pure `validate_launch` from an explicitly opt-in “resolve artifacts” preflight. Do not promise “no data loaded” unless that includes prediction artifacts.

## 7. CLI spec, docs, and tests

The `--dump-spec` and generated-reference goals remain useful. Dynamic registry choices must be labeled by source and resolved deterministically. Include environment timing/precedence, the current rule set, and source-edit-only knobs; do not treat output strings or import-time values as a complete spec by themselves.

In addition to the proposed tests, cover:

- Registry lookup from `build_parser()` and the pandas import edge; do not just test `import mpml.cli`.
- Env snapshot behavior versus runtime reads, including the `DL_SIGNALS_CUBE_PATH` unset-versus-empty distinction.
- Valid and unknown `ACTIVE_PAIRS`, missing artifact discovery, and explicit missing artifact while confirming current non-error behavior.
- `--recommendation-top-n` values `0`, negative, and positive.
- A timeline-export scope with and without the effective PhaseAware pair.
- Existing output directory: ensure preflight reports it without constructing the data pipeline, and actual execution still uses atomic directory creation.
- The empty-spec error path, documenting the existing `NameError` without silently bundling a behavior fix into a “pure” refactor.
- `main()` import side effects and repeated direct `main()` calls if `clear_cache()` is moved.

The import-hygiene assertion should distinguish importing `mpml.cli` from constructing a parser. A strict no-pandas parser-construction guarantee requires a light strategy-ID source or deferred/optional choice resolution; relying on lazy module imports alone does not achieve it.

## 8. Implementation scope and authority

This audit informed the subsequent design, but its earlier suggestions do not prescribe the implementation. The authoritative sequence is defined in `docs/architecture/MPML_LAUNCH_SURFACE_REFACTOR.md`:

1. **PR 1 — Characterize and move:** characterize existing behavior, extract the CLI parser, and isolate import-time cache/configuration side effects with explicitly documented exceptions.
2. **PR 2 — Additive preflight:** add `python -m mpml.preflight` using the real subprocess environment and existing production predicates; do not change the run path.
3. **PR 3 — Spec and reference:** add `python -m mpml.spec` and generated reference documentation, without changing production configuration reads.

An injectable resolved configuration, artifact-aware preflight, and behavior-changing validation policies are deferred. Refer to the refactor document for all acceptance criteria and implementation boundaries.

## 9. Source-edit-only settings and scope notes

The preliminary source-only examples are real (`main.py:135-160, 284-324`): `RUN_IN_SAMPLE_ABLATION`, `RUN_WALKFORWARD`, `RUN_TAU_SWEEP`, `RUN_POLICY_SWEEP`, debug toggles, cost assumptions (`SPREAD_PIPS`, `SLIPPAGE_PIPS`, `COMMISSION_PER_TRADE`), selector thresholds/policy (`WF_TAU`, `DYNAMIC_POLICY_KWARGS`), volatility-guard settings, and selected diagnostic pairs/fold limits are constants, not CLI or environment settings. The generated reference should identify these as source-edit-only, not imply they are part of the env registry.

The refactor can keep strategy, selector, backtest, and evaluation logic unchanged. However, centralizing configuration, moving when cache clearing occurs, and making a formerly late failure happen before data loading are observable changes in timing/side effects; characterize them and state exactly which run behavior remains invariant.