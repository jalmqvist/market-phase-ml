# MPML CLI Reference

<!-- Generated from docs/cli_spec.json by `python scripts/generate_cli_reference.py`; do not edit by hand. -->

This reference describes the existing MPML launch surface. It does not change runtime behavior.
Generate or check the committed files with `python scripts/generate_cli_reference.py` or
`python scripts/generate_cli_reference.py --check`.

## Command-line arguments

| Flags | Destination | Action | Type | Default | Choices | Help |
| --- | --- | --- | --- | --- | --- | --- |
| --output-dir | output_dir | store | Path | - | - | Explicit run output directory (writes all run artifacts here). If omitted, MPML creates results_archive/<gen>_<variant>__<timestamp>/. |
| --experiment-generation | experiment_generation | store | str | - | gen1, gen2 (static) | Optional generation consistency hint (gen1\|gen2). Must match selected variant semantics if provided. |
| --experiment-variant | experiment_variant | store | str | - | A, B, C, D, E, F (static) | Canonical experiment variant (A\|B\|C\|D\|E\|F). Precedence: --experiment-variant > EXPERIMENT_VARIANT env > default A. |
| --experiment-seed | experiment_seed | store | int | - | - | Experiment RNG seed. Precedence: --experiment-seed > EXPERIMENT_SEED env > default (42). |
| --behavioral-surface | behavioral_surface | store | str | - | persistent, reactive_jpy, trend_vol (behavioral_registry) | Behavioral Surface to use for this run. Precedence: --behavioral-surface > BEHAVIORAL_SURFACE env > 'trend_vol'. Available: ['persistent', 'reactive_jpy', 'trend_vol']. Trend/Vol (trend_vol) is the default and preserves existing run behaviour. |
| --recommendations-per-pair | recommendations_per_pair | store_true | - | false | - | Generate recommendations_per_pair.parquet from pair-scoped walk-forward evidence. |
| --recommendation-top-n | recommendation_top_n | store | int | - | - | Return only the top N recommendations ordered by the default recommendation policy (sharpe_rank_v1). If omitted, all recommendations are returned. N must be a positive integer. |
| --strategy | strategy | append | - | - | MR1, MR2, MR3, MR32, MR42, MR5, TF1, TF2, TF3, TF4, TF5 (strategy_registry) | Evaluate only this strategy ID. Repeatable: --strategy TF4 --strategy MR42. When omitted, the default evaluation scope from the active EvaluationPolicy is used (backward-compatible). IDs are validated against the Strategy Registry. Available: ['MR1', 'MR2', 'MR3', 'MR32', 'MR42', 'MR5', 'TF1', 'TF2', 'TF3', 'TF4', 'TF5'] |
| --phaseaware-tf | phaseaware_tf | append | - | - | TF1, TF2, TF3, TF4, TF5 (strategy_registry) | Override the TrendFollowing representative used by PhaseAware for this run. Must be a Strategy Registry TrendFollowing ID, e.g. TF2. May be provided at most once. |
| --phaseaware-mr | phaseaware_mr | append | - | - | MR1, MR2, MR3, MR32, MR42, MR5 (strategy_registry) | Override the MeanReversion representative used by PhaseAware for this run. Must be a Strategy Registry MeanReversion ID, e.g. MR2. May be provided at most once. |
| --debug | debug | store_true | - | false | - | Enable verbose debug output: [AWARENESS], [TRAINING DIAGNOSTICS], [DL FEATURE USAGE], and other detailed per-fold diagnostics that are suppressed by default for concise normal output. |
| --export-dl-conditional-portfolio-timeline | export_dl_conditional_portfolio_timeline | store_true | - | false | - | Opt-in: write dl_conditional_portfolio_timeline.csv (per pair/fold/test-bar realized dynamic-selector portfolio outcomes with dl_available) to the run output directory. Disabled by default; does not change strategy behavior. |

## Registries

| Registry | Identifiers | Source |
| --- | --- | --- |
| Strategies | MR1, MR2, MR3, MR32, MR42, MR5, TF1, TF2, TF3, TF4, TF5 | strategy_registry |
| Behavioral surfaces | persistent, reactive_jpy, trend_vol | behavioral_registry |

Strategy choices on `--strategy`, `--phaseaware-tf`, and `--phaseaware-mr` are validated after parsing.

## Precedence

| Setting | Order (highest priority first) |
| --- | --- |
| behavioral_surface | --behavioral-surface, BEHAVIORAL_SURFACE, trend_vol |
| experiment_seed | --experiment-seed, EXPERIMENT_SEED, 42 |
| experiment_variant | --experiment-variant, EXPERIMENT_VARIANT, A |
| output_directory | --output-dir, MPML_OUTPUT_DIR, MPML_RUNS_ROOT/results_archive default |

## Environment variables

Import-time settings are snapshotted when their module is imported; runtime settings are read during execution.

| Variable | Parser | Default | Timing | Scope | Source files | Semantics |
| --- | --- | --- | --- | --- | --- | --- |
| ACTIVE_PAIRS | csv | all registered pairs | import | launch_selector | main.py, src/behavioral_artifact_resolver.py, src/experiment_surface_runtime.py | Comma-separated pair names; values are stripped and uppercased. Empty selects all pairs. The import-time filter and later metadata inference also read the process environment. |
| BEHAVIORAL_STATE_ID | str | - | import | launch_selector | main.py, src/dl_config.py | Stripped at the runtime read; blank becomes unset. Also snapshotted by src.dl_config for canonical identity helpers. |
| BEHAVIORAL_SURFACE | enum | trend_vol | runtime | launch_selector | main.py | CLI --behavioral-surface takes precedence; otherwise stripped environment value, then trend_vol. |
| BEHAVIORAL_SURFACE_ID | str | - | import | metadata | src/dl_config.py | Empty string becomes unset; part of canonical behavioral identity. |
| BEHAVIORAL_SURFACE_VERSION | str | - | import | metadata | src/dl_config.py | Empty string becomes unset; part of canonical behavioral identity. |
| DL_FEATURE_SET | str | price_trend | import | launch_selector | main.py | Snapshotted in main.DL_SIGNAL_SURFACE; a runtime check also tests whether the environment value is blank when resolving an artifact. |
| DL_MODEL | str | mlp | import | launch_selector | main.py | Snapshotted into main.DL_SIGNAL_SURFACE. |
| DL_PREDICTION_ARTIFACT_PATH | path | sibling market-sentiment-ml artifact directory | import | launch_selector | main.py, src/dl_config.py | src.dl_config snapshots this at import. If unset (not empty), DL_SIGNALS_CUBE_PATH is used; main.py also reads it at runtime. |
| DL_REGIME | str |  | import | launch_selector | main.py | Snapshotted into main.DL_SIGNAL_SURFACE. |
| DL_SIGNALS_CUBE_PATH | path | sibling market-sentiment-ml artifact directory | import | launch_selector | src/dl_config.py | Legacy path alias used only if DL_PREDICTION_ARTIFACT_PATH is unset. An empty primary value suppresses this alias. |
| DL_SIGNALS_ENABLED | bool_strict | false | import | launch_selector | main.py, src/dl_config.py | Only case-insensitive exact text true enables; 1, yes, on, empty, and other values are false. |
| DL_SURFACE_FEATURE_SET | str | price_trend | import | launch_selector | src/dl_config.py | Snapshotted in the legacy DL surface selector. |
| DL_SURFACE_MODEL | str | lstm | import | launch_selector | src/dl_config.py | Snapshotted in the legacy DL surface selector. |
| DL_SURFACE_REGIME | str | HVTF | import | launch_selector | src/dl_config.py | Snapshotted in the legacy DL surface selector. |
| DL_SURFACE_TARGET_HORIZON | int | 24 | import | launch_selector | src/dl_config.py, src/launch_env.py | Parsed with int() when the legacy DL surface is imported; malformed input fails import. |
| EXPERIMENT_GENERATION | enum | - | runtime | launch_selector | main.py | CLI hint takes precedence; when supplied, it must agree with the selected variant. |
| EXPERIMENT_SEED | int | 42 | runtime | launch_selector | src/repro.py | CLI seed wins. Environment text is stripped; blank uses 42 and malformed nonempty text raises ValueError. |
| EXPERIMENT_VARIANT | enum | A | runtime | launch_selector | main.py | CLI value takes precedence; otherwise environment value, then A. |
| EXPORT_SELECTOR_STATE_TIMELINE | bool_loose | false | import | diagnostic | main.py | Case-insensitive 1, true, yes, or on are true; every other value is false. |
| MIN_DL_TRAIN_COVERAGE_PCT | float | 5.0 | import | launch_selector | main.py, src/launch_env.py | Parsed with float() when main is imported; malformed input fails import. Preflight invokes the same converter. |
| MPML_ARTIFACT_SOURCE | str | - | runtime | metadata | src/experiment_surface_runtime.py | Stripped non-empty value supplies artifact-source metadata when artifact metadata is absent. |
| MPML_AUTO_ANALYZE | bool_loose | true | runtime | launch_selector | main.py | Case-insensitive 1, true, yes, or on are true; every other value is false. |
| MPML_BROKER_DATA_DIR | path | sibling market-sentiment-ml input directory | runtime | launch_selector | src/data.py | Trimmed and expanded when constructing a broker_csv pipeline; blank uses the sibling-repository default. |
| MPML_DATA_SOURCE | enum | yfinance | runtime | launch_selector | src/data.py | Trimmed and lowercased; supported values are yfinance and broker_csv. Invalid values raise ValueError. |
| MPML_END_DATE | str | 2026-04-01 | import | launch_selector | main.py | Snapshotted as a string when main is imported. |
| MPML_EVALUATION_PAIR_FAMILY | str | - | runtime | metadata | src/experiment_surface_runtime.py | Stripped non-empty value supplies evaluation-pair-family metadata when artifact metadata is absent. |
| MPML_FEATURE_SURFACE | str | - | runtime | metadata | src/experiment_surface_runtime.py | Stripped non-empty value supplies feature-surface metadata before the DL surface fallback. |
| MPML_OUTPUT_DIR | path | - | runtime | launch_selector | main.py | CLI --output-dir wins. Also supplies the output-name hint used by imputation-awareness inference. |
| MPML_REVISION | str | - | runtime | metadata | mpml/runtime/phaseaware.py | Optional revision/provenance value. |
| MPML_RUNS_ROOT | path | results_archive | runtime | launch_selector | main.py | Used as the parent for generated run output paths when no explicit output directory is selected. |
| MPML_SELECTOR_TRACE_PAIR | str |  | runtime | diagnostic | src/strategies.py | Pair selector for optional single-bar trace output; only used with MPML_SELECTOR_TRACE_PATH. |
| MPML_SELECTOR_TRACE_PATH | path | - | runtime | diagnostic | src/strategies.py | Enables optional selector tracing for the matching pair and timestamp. |
| MPML_SELECTOR_TRACE_TIMESTAMP | str |  | runtime | diagnostic | src/strategies.py | Timestamp selector for optional single-bar trace output. |
| MPML_SENTIMENT_SURFACE | str | - | runtime | metadata | src/experiment_surface_runtime.py | Stripped surface value; supported normalized values are sentiment, no_sentiment, and none. |
| MPML_START_DATE | str | 2005-01-01 | import | launch_selector | main.py | Snapshotted as a string when main is imported. |
| MPML_TRAINING_PAIR_FAMILY | str | - | runtime | metadata | src/experiment_surface_runtime.py | Stripped non-empty value supplies training-pair-family metadata when artifact metadata is absent. |
| MSML_REGIME | str | selected DL regime or unknown | runtime | launch_selector | main.py | Trimmed and uppercased; fallback depends on the selected behavioral surface and resolved state. |
| OVERLAP_ONLY | bool_loose | false | runtime | launch_selector | main.py | Case-insensitive 1, true, yes, or on are true; every other value is false. |

## Preflight issue codes

| Code | Summary |
| --- | --- |
| ACTIVE_PAIRS_UNKNOWN | Configured pair names are not in the supported pair universe. |
| ARGPARSE | Arguments are not accepted by the MPML command-line parser. |
| DL_ALIAS_SUPPRESSED | The primary prediction artifact setting suppresses its legacy alias. |
| DL_ARTIFACT_PATH_MISSING | The explicitly configured prediction artifact does not exist. |
| ENV_BOOL_PARSED_FALSE | A DL boolean value looks truthy but is parsed as false. |
| ENV_INVALID_VALUE | An environment value cannot be parsed or is unsupported. |
| GENERATION_VARIANT_MISMATCH | The requested generation conflicts with the selected variant. |
| OUTPUT_DIR_EXISTS | The selected run output directory already exists. |
| OUTPUT_NAME_INFERS_AWARENESS | The output name supplies an imputation-awareness hint. |
| PHASEAWARE_REPEATED | A PhaseAware representative option was specified more than once. |
| PHASEAWARE_ROLE_INVALID | A PhaseAware representative does not match its required role. |
| STRATEGY_UNKNOWN | A requested strategy ID is not in the strategy registry. |
| SURFACE_UNKNOWN | The requested behavioral surface is not registered. |
| TIMELINE_NEEDS_DYNAMIC_SELECTOR | Timeline export requires an active dynamic selector. |
| TOP_N_NOT_POSITIVE | The recommendation limit must be a positive integer. |

## Source-edit-only settings

These module-level settings in `main.py` are not exposed as command-line arguments or environment variables.
Changing them requires editing source; they are not part of the supported launch configuration surface.

| Setting | Category |
| --- | --- |
| `RUN_IN_SAMPLE_ABLATION` | experiment toggles |
| `RUN_WALKFORWARD` | experiment toggles |
| `RUN_TAU_SWEEP` | experiment toggles |
| `RUN_POLICY_SWEEP` | experiment toggles |
| `DL_DEBUG_VERBOSE` | debug toggles |
| `DEBUG_BASELINE_KEYS` | debug toggles |
| `DEBUG_FEATURE_COLUMNS` | debug toggles |
| `DEBUG_SIGNAL_TYPES` | debug toggles |
| `DEBUG_VOL_GUARD` | debug toggles |
| `DEBUG_SAVE_SELECTED_SERIES` | debug toggles |
| `DEBUG_SAVE_EQUITY_SERIES` | debug toggles |
| `DEBUG_SELECTED_PAIRS` | debug toggles |
| `DEBUG_SELECTED_MAX_FOLDS_PER_PAIR` | debug toggles |
| `SPREAD_PIPS` | transaction costs |
| `SLIPPAGE_PIPS` | transaction costs |
| `COMMISSION_PER_TRADE` | transaction costs |
| `WF_TAU` | walk-forward selector |
| `DYNAMIC_POLICY_KWARGS` | walk-forward selector |
| `USE_VOL_GUARD` | volatility guard |
| `VOL_GUARD_Q` | volatility guard |
| `VOL_GUARD_MODE` | volatility guard |
| `VOL_FEATURE` | volatility guard |
| `USD_QUOTE_VOL_SPIKE_OVERRIDE` | volatility guard |
| `VOL_GUARD_NEAR_MULT` | volatility guard |
