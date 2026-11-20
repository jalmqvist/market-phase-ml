# MPML Launch-Surface Refactor

**Repository:** `jalmqvist/market-phase-ml`  
**Basis:** `main` branch, commit `ef6b1b84c7393a1bb4dbd6f92a1442115c6cbb57`  
**Purpose:** Characterize and improve MPML's CLI launch surface for reliable validation, documentation and external tooling, while preserving normal experiment behavior. The `argus-cli` RAG assistant is an initial consumer.

---

## Principles

1. **Characterize first.** Record current behavior before moving code.
2. **Preserve experiment behavior in PRs 1–3.** PR 1 intentionally changes when cache clearing and configuration output occur. Strict booleans, unknown-pair rejection, required artifacts, and the `_scope_tf` NameError fix are deferred to separate PRs.
3. **Side-effect isolation, not pandas avoidance.** The parser construction imports pandas via the strategy registry. This is acceptable; the goal is no import-time cache deletion, prints or filesystem writes.
4. **Subprocess preflight with real env.** Do not attempt injectable config objects. Import-time snapshots and `os.environ` reads are too pervasive.
5. **Keep "validate" distinct from "run."** Artifact resolution loads prediction data; do not call it a no-data dry run.
6. **Compatibility is defined at the valid CLI experiment boundary.** Preserve CLI syntax,
   argument resolution, strategy/phase selection, calculations, output schemas and results,
   and run-time cache-clearing order for valid CLI runs. PR 1 deliberately changes import,
   `--help`/invalid-argv and direct-Python-call side effects; document those exceptions.
   Do not claim byte-for-byte stdout compatibility where the `[CONFIG]` print moves.
7. **Self-contained repository tests.** All required characterization, preflight and spec
   tests must run from a clean checkout with synthetic or temporary fixtures. The external
   `mpml_run_examples.tgz` is optional contextual evidence, not an implementation dependency.
8. **Facts come from the audit, scope comes from this roadmap.** `docs/architecture/MPML_LAUNCH_SURFACE_AUDIT.md`
   describes what the code does at the audited commit. The audit records observed behavior and historical design options; the PRs below
   define the authoritative implementation scope and acceptance criteria.

---

## Verified Surface (audit summary)

| Argument | Parse | Later validation |
|---|---|---|
| `--output-dir` | `Path`, default `None` | Exclusive mkdir in `_set_run_output_dir` (after pipeline construction) |
| `--experiment-generation` | choices `gen1`, `gen2` | Must match variant-implied generation |
| `--experiment-variant` | choices `A`-`F` | Precedence: CLI > `EXPERIMENT_VARIANT` > `A` |
| `--experiment-seed` | `int`, default `None` | Precedence: CLI > `EXPERIMENT_SEED` > `42` |
| `--behavioral-surface` | choices `persistent`, `reactive_jpy`, `trend_vol` | Precedence: CLI > `BEHAVIORAL_SURFACE` > `trend_vol` |
| `--recommendations-per-pair` | `store_true` | Enables per-pair output |
| `--recommendation-top-n` | `int`, default `None` | Must be positive (late check) |
| `--strategy` | repeatable `append` | Resolved against registry; duplicates de-duplicated |
| `--phaseaware-tf`, `--phaseaware-mr` | repeatable `append` | At most one each; must be registered TF/MR ID |
| `--debug` | `store_true` | Sets verbosity |
| `--export-dl-conditional-portfolio-timeline` | `store_true` | Requires dynamic selector active (late check) |

**Valid strategy IDs:** `TF1`–`TF5`, `MR1`, `MR2`, `MR3`, `MR32`, `MR42`, `MR5`.

**Key env vars (30+ total):** The full inventory (source file, timing, parser, default, alias
precedence) is in `docs/architecture/MPML_LAUNCH_SURFACE_AUDIT.md`, section 5. Critical patterns:

- `DL_SIGNALS_ENABLED`: strict bool (`true` only; `1`/`yes` are false). Import-time in `main.py` and `src/dl_config.py`.
- `EXPORT_SELECTOR_STATE_TIMELINE`, `OVERLAP_ONLY`, `MPML_AUTO_ANALYZE`: loose bool (`1`/`true`/`yes`/`on`).
- `ACTIVE_PAIRS`: comma-separated short names; unknown names silently dropped.
- `DL_PREDICTION_ARTIFACT_PATH` vs `DL_SIGNALS_CUBE_PATH`: the alias is used only when the primary is **unset**. An empty primary suppresses it.
- `MPML_OUTPUT_DIR` / `--output-dir` name can flip `missing_indicators_enabled` via `infer_imputation_awareness_from_name`.

---

## PR 1: Characterize and Move

### Goals

- Record current parser, env parsing, artifact-missing behavior, output-dir collision, and cache effects.
- Move `build_parser()` to `mpml/cli.py` with identical arguments.
- Move `clear_cache()` and `[CONFIG]` print into `_run_cli()`, not `main()`.
- Ensure `import mpml.cli` and `import main` have no side effects.

### Compatibility contract (PRs 1–3)

For valid `python main.py <args>` launches, preserve the complete argument interpretation
(including environment fallbacks and import-time snapshots), effective strategy/PhaseAware
scope, experiment and DL activation semantics, numerical computations, output schemas,
manifest semantics, and the cache-clearing point relative to pipeline/cache reads. These
properties are the backwards-compatibility contract; successful tests do not justify
changing strategy or research logic.

**Documented PR 1 exceptions:** importing `main` or `mpml.cli` no longer clears caches,
creates `data/cache`, or prints configuration; `--help` and rejected CLI arguments no
longer do so either; direct Python `main()` calls no longer inherit an import-triggered
cache clear or config print. Successful CLI runs still clear the same `.pkl` files before
pipeline work, although config-print ordering may change. Do not promise unchanged
stdout/stderr ordering for the exceptions. Verify all known direct callers rather than
silently relying on their previous import side effects.

### Characterization tests (write first)

Import-time snapshots mean **each env case runs in a fresh subprocess** with a temporary
working directory. Never run them against real data or the real `data/` directory.

| Test | Input | Recorded behavior |
|---|---|---|
| Parser equivalence | Table of argv lists (see "Capturing the old parser" below) | Recorded `main()` kwargs, or `SystemExit` code, identical before and after the move |
| `DL_SIGNALS_ENABLED` | unset, `""`, `true`, `True`, `1`, `yes`, `on`, `garbage` | Resolved boolean in `main.py` **and** in `src/dl_config.py`. Only `true` (any case) is true |
| Loose booleans | Same values for `EXPORT_SELECTOR_STATE_TIMELINE`, `OVERLAP_ONLY`, `MPML_AUTO_ANALYZE` | `1/true/yes/on` (any case) are true. Everything else is false |
| `ACTIVE_PAIRS` | Valid short names, unknown names, mixed case, empty, whitespace | Normalized set, silent drop of unknown names, no error |
| DL enabled, no artifact resolvable | `DL_SIGNALS_ENABLED=true`, no path, no default artifact | Runtime disabled with diagnostics, **no exception** |
| DL enabled, explicit missing path | `DL_PREDICTION_ARTIFACT_PATH=/nonexistent.parquet` | Runtime disabled, no exception |
| Artifact alias precedence | Primary unset vs primary set to `""` vs primary set, with `DL_SIGNALS_CUBE_PATH` set | Alias used only when primary is **unset**. Empty primary suppresses it |
| `EXPERIMENT_SEED` | unset, `""`, `42`, `abc` | Default 42 for unset and empty. `ValueError` for `abc`. CLI value wins |
| `MPML_DATA_SOURCE` | unset, `broker_csv`, `YFinance`, `bogus` | Default `yfinance`. Trimmed and lowercased. `ValueError` for `bogus` |
| Output directory collision | `--output-dir` pointing at an existing directory | `ValueError`. Record that `MarketDataPipeline` is constructed (cache dirs created) **before** the check |
| `clear_cache()` scope | Populated `data/cache/*.pkl` plus unrelated files and `data/raw_broker_csv/` | Only matching `.pkl` files in `data/cache` are deleted |
| `--recommendation-top-n` | `0`, `-1`, `3` | Late rejection of `0` and `-1` in recommendation construction. `3` is accepted |
| Timeline rule | `_build_walkforward_execution_plan` for default scope, effective PhaseAware TF+MR together, TF only, MR only, unrelated IDs | Value of `run_dynamic_selector` for each case. Characterize the **function**, not a full run |
| Empty strategy specs | Call the error path directly | Currently raises `NameError` (`_scope_tf`, `_scope_mr`). **Document, do not fix** |
| Direct `main()` caller compatibility | In a fresh subprocess, import `main`, monkeypatch the pipeline/side-effect boundary, invoke `main()` with a minimal valid call | Record invocation semantics before and after; explicitly expect import-triggered cache deletion and `[CONFIG]` printing to disappear, without changing resolved arguments or initiating a real experiment |
| CLI cache-clear ordering | Valid CLI invocation with a seeded `data/cache/*.pkl` and an instrumented pipeline constructor | Same matching files deleted **before** any pipeline constructor/cache read; unrelated files remain |
| Import side effects ("before" snapshot) | `import main` in a fresh subprocess in an empty tmp directory | Currently: prints `[CONFIG] ...`, creates `data/cache`, deletes `*.pkl`. Recorded so PR 1's change is deliberate and visible |

### Capturing the old parser

The old parser is built inline inside `_run_cli()`, so it can't be called directly. Don't extract
it. Observe it through `_run_cli()` itself:

1. Write the test **before** any code is moved. Run it in a fresh subprocess in an empty temporary
   working directory. Importing `main` still clears caches and creates `data/cache` at this
   stage, and that must never touch the real `data/` directory.
2. Replace `main.main` with a recorder that stores the keyword arguments it receives and
   returns. Then call `main._run_cli(argv)`.
3. For each argv in the table, record one of:
   - the kwargs passed to `main()`, with `Path` values converted to `str`, or
   - the `SystemExit` code (argparse errors and the `parser.error()` calls for repeated
     `--phaseaware-*` both exit with code 2).
4. Store the results as a committed snapshot, for example
   `tests/characterization/parser_snapshots.json`. Generate it from the **unchanged** code.
5. After the move, the same test runs against the snapshot without modification. It then checks
   `build_parser()` and the `_run_cli()` wiring together.

Argv table (minimum):

- no arguments
- every flag once with a valid value
- repeated `--strategy` (including duplicates, such as `--strategy TF4 --strategy TF4`)
- `--phaseaware-tf TF2 --phaseaware-tf TF3` (post-parse rejection)
- invalid `--behavioral-surface`, `--experiment-variant`, `--experiment-generation`
- `--experiment-seed abc` and `--recommendation-top-n 0`
- `--output-dir relative/path`
- `--debug` with the other store-true flags

Capture baseline help through `_run_cli(["--help"])` with `COLUMNS=100` and a stable program
name. Record argparse's help text separately from import output. Compare the help text
after extraction, including the dynamic registry text.

`argv=None` (reading `sys.argv`) is not covered by the table. Cover it with a single test that sets
`sys.argv`.

### Changes

1. **`mpml/cli.py`:** `build_parser(parser_class=argparse.ArgumentParser)`. Copy every
   argument verbatim: flags, `dest`, action, type (`--output-dir` stays `Path`), default, help,
   and dynamic `choices` (still resolved from `behavioral_registry` and `VALID_EXPERIMENT_VARIANTS`).
   Read the same strategy and surface registries. Put the generation choices in a module
   both parser construction and runtime validation can import without importing `main.py`.
2. **`main.py::_run_cli()`:** call `build_parser()`. The post-parse checks (`--phaseaware-*`
   at most once) stay in `_run_cli()` for now.
3. **Side effects:** move the module-level `clear_cache()` call and the `[CONFIG] START_DATE=...`
   print into `_run_cli()`, after argparse and the repeated PhaseAware checks, immediately
   before calling `main()`. Help, rejected argv, imports, and direct `main()` calls leave the
   application cache untouched. Test help, syntax rejection, and repeated PhaseAware flags
   with a seeded cache file. Valid CLI invocations still clear the cache before the pipeline runs.
4. **Leave import-time env reads alone.** `DL_SIGNALS_ENABLED`, `ACTIVE_PAIRS`, `START_DATE`,
   `END_DATE` and the others stay as module constants in this PR.

### Check before starting

- `grep` for callers of `main(` and for code that imports `main`: tests, `run_v5_full_matrix.sh`,
  analysis scripts, notebooks. Anything that relied on import-time cache clearing needs a
  decision.
- Confirm that `mpml/__init__.py` does not import `main`.
- Confirm that no test asserts the order of the `[CONFIG]` print relative to other output.
- Characterize direct Python callers as well as CLI entry points: preserve their arguments,
  return values and run semantics as far as possible; treat removal of import-time side
  effects as a documented exception. Do not add cache clearing to `main()` to compensate.

### Acceptance criteria

- All characterization tests pass against **both** the old and the new code, with explicit before/after expectations for the intentional cache and configuration-output changes.
- In a fresh subprocess in an empty directory, `import mpml.cli` and `import main`
  print nothing and create no files or directories.
- Portable parser/env/cache/direct-caller tests pass on a clean checkout without the
  external experiment archive or proprietary data. A direct `main()` smoke test explicitly
  distinguishes intentional import-side-effect changes from argument/dispatch regressions.
- When representative datasets/artifacts are available locally, capture **fresh** pre- and
  post-change baseline and DL-enabled CLI runs at the refactor revision. Compare normalized
  `run_manifest.json` fields (excluding ephemeral timestamps and run IDs), effective resolved
  experiment metadata (including surface, strategy scope, pair set and DL activation), and a
  small set of corresponding output artifact schemas/content or numerical summaries with
  justified deterministic tolerances. Do not require byte identity for inherently
  nondeterministic outputs. Run both cells with the same inputs, environment, cwd and seed.
  These real-run checks are strongly recommended integration evidence, not a dependency on
  `mpml_run_examples.tgz`. If unavailable, explicitly mark **numerical experiment parity
  unverified**, while reporting portable launch-surface compatibility separately; never
  claim full experiment equivalence without evidence.
- Not asserted: whether pandas is imported. Building the parser calls the strategy registry,
  which imports `src.strategies` and pandas. That is accepted.

---

## PR 2: Additive Preflight (existing run paths untouched)

### Goal

Let a caller ask "would this launch be rejected, or would it silently do something
surprising?" without starting a run.

### Design

- New module `mpml/preflight.py`, run as `python -m mpml.preflight <same argv as main.py>`.
  A module entry point is used so that **`build_parser()` and `main.py` stay unchanged**.
- It reads the **real process environment**. There is no injected `env` mapping. Callers
  (including `argus-cli`) set the environment on the subprocess. Import-time snapshots and
  alias precedence then behave as in a real launch.
- Output is JSON on stdout: `{"schema_version": 1, "issues": [...]}`. Each issue has
  `code`, `severity` (`error` or `warning`), `source`, `message`. Exit code is non-zero if any
  `error` exists.
- Nothing in `main()` or `_run_cli()` calls it. Preserve each run check's position and
  rejection behavior when extracting a shared predicate.
- Keep eager launch imports out of the preflight entry point. The module-level float
  conversion for `MIN_DL_TRAIN_COVERAGE_PCT` in `main.py` and int conversion for
  `DL_SURFACE_TARGET_HORIZON` in `src/dl_config.py` can fail even with DL disabled.
  Share the conversions or report their failures with variable-specific context so preflight
  can emit `ENV_INVALID_VALUE` on stdout. Leave unrelated import failures distinguishable
  from invalid environment values.

### What it must not do

- Call `clear_cache()`.
- Construct `MarketDataPipeline` (its constructor creates cache directories).
- Call `_set_run_output_dir()` or create any directory.
- Resolve behavioral artifacts. That loads prediction data and is out of scope here.

### Checks

Each check calls the production function and converts exceptions into issues. No rule text
is duplicated. When a rule currently lives inline in a run path, extract the smallest
shared predicate only if it can be called without executing a pipeline; retain the original
production call location, exception class/message and error timing. Do not introduce a
second independently maintained rule implementation. If a required production dependency
would trigger experiment work, expensive artifact resolution or filesystem mutations,
report the obstruction before widening the refactor or copying the rule.

| Code | Severity | Production function or source |
|---|---|---|
| `ARGPARSE` | error | `build_parser(parser_class=_RaisingParser)` (raises instead of `sys.exit`). Reports the first syntax error only |
| `PHASEAWARE_REPEATED` | error | Shared predicate used by `_run_cli()` after parsing |
| `SURFACE_UNKNOWN` | error | `behavioral_registry`, including the env fallback `BEHAVIORAL_SURFACE` |
| `STRATEGY_UNKNOWN` | error | `resolve_evaluation_scope(...)` |
| `PHASEAWARE_ROLE_INVALID` | error | `resolve_phaseaware_configuration(...)` |
| `GENERATION_VARIANT_MISMATCH` | error | `normalize_variant` and `build_experiment_metadata_from_variant`, including `EXPERIMENT_VARIANT` and `EXPERIMENT_GENERATION` fallbacks |
| `TOP_N_NOT_POSITIVE` | error | Shared predicate used by recommendation construction |
| `TIMELINE_NEEDS_DYNAMIC_SELECTOR` | error | `_build_walkforward_execution_plan(scope, ...)["run_dynamic_selector"]` |
| `OUTPUT_DIR_EXISTS` | error (advisory) | `Path.exists()` on the selected output path. The exclusive mkdir at run time remains the real guard |
| `ENV_INVALID_VALUE` | error | `EXPERIMENT_SEED` not an int, `MPML_DATA_SOURCE` not `yfinance`/`broker_csv`, `MIN_DL_TRAIN_COVERAGE_PCT` not a float, `DL_SURFACE_TARGET_HORIZON` not an int |
| `ENV_BOOL_PARSED_FALSE` | warning | A strict-bool variable (`DL_SIGNALS_ENABLED`) set to `1`, `yes` or `on`: parsed as false, so the run is a baseline run |
| `ACTIVE_PAIRS_UNKNOWN` | warning | Names not in the short names derived from `PAIR_NAMES` (not `ALL_PAIRS`, which holds Yahoo tickers). Unknown names are silently dropped today |
| `DL_ARTIFACT_PATH_MISSING` | warning | DL requested and the explicit path, after stripping whitespace and expanding `~`, does not exist. The explicit path branch disables DL without discovery |
| `DL_ALIAS_SUPPRESSED` | warning | `DL_PREDICTION_ARTIFACT_PATH` set (even to `""`) together with `DL_SIGNALS_CUBE_PATH` |
| `OUTPUT_NAME_INFERS_AWARENESS` | warning | The name of `--output-dir` or `MPML_OUTPUT_DIR` makes `infer_imputation_awareness_from_name` return a non-`None` value. The generated default under `MPML_RUNS_ROOT` is **not** passed as that hint, so don't warn for it |

Deliberately **no** warning for `DL_SIGNALS_ENABLED=true` without any artifact path. The
resolver can discover a default artifact. The configuration helper can choose the newest
parquet in a discovery directory. The explicit path branch only checks existence before
loading. A resolved path can still fail state selection or strict loading. The fast check
reports path and alias facts without claiming DL activation. Likewise
no `DL_MODEL` versus filename-prefix check, which isn't an established runtime rule.

### Limits to state in the module docstring

- Argparse stops at the first syntax error. Aggregation applies only to the independent checks after a successful parse.
- Existence checks are advisory and race-prone.
- Artifact resolution (loading prediction data) is **not** covered.

### Acceptance criteria

- At least one test per issue code that triggers it, and one that confirms a valid launch yields no issues.
- Timeline cases: default scope, effective PhaseAware TF+MR together, TF alone, MR alone,
  unrelated IDs. The execution plan derives eligibility from scope and composition without
  loading prediction data. The CLI portfolio timeline flag can raise when the selector is
  inactive. `EXPORT_SELECTOR_STATE_TIMELINE` produces a skip diagnostic in that case.
- In fresh subprocesses, each malformed import-time numeric variable yields one JSON document,
  `ENV_INVALID_VALUE` naming the variable, and a nonzero exit. Cover runtime-value precedence
  separately, including a valid CLI seed overriding an invalid environment seed.
- A test that runs preflight in an empty temp directory and asserts that **nothing was created** and no caches were touched.
- A test that `MarketDataPipeline` is never constructed (patch it to raise).
- A regression test for each extracted shared predicate demonstrates that the production
  run path still calls it at its original point and preserves its old failure behavior;
  preflight produces the corresponding issue without duplicating the rule.
- Preflight exits 0 and prints `"issues": []` for a known-good baseline cell and a known-good DL cell.

---

## PR 3: Spec Generator (`python -m mpml.spec`) and Generated Reference

### Goal

A deterministic, machine-readable description of the launch surface, usable as the primary
knowledge source for `argus-cli`.

### Design

- `python -m mpml.spec` prints JSON. Sorted keys, stable ordering, **no timestamps, no commit SHA**
  (the committed file would otherwise be stale after every commit).
- Contents:
  - `arguments`: flags, `dest`, action, type name, default, `nargs`, help text, and resolved
    `choices` labeled with `choices_source` (`static`, `behavioral_registry`, `strategy_registry`, ...).
  - Record registry values enforced after parsing separately from argparse `choices`.
  - `precedence`: the rules stated in help strings and the audit (`--behavioral-surface` > `BEHAVIORAL_SURFACE` > `trend_vol`; seed; variant; output directory).
  - `env`: one entry per variable with `name`, `parser` (`bool_strict`, `bool_loose`, `int`, `float`, `str`, `path`, `csv`, `enum`), `default`, `timing` (`import` or `runtime`), `read_in` (files), `alias_of`, and `scope` (`launch_selector`, `diagnostic`, `metadata`).
  - `rules`: the preflight codes from PR 2 with one-line summaries.
  - `source_edit_only`: `RUN_IN_SAMPLE_ABLATION`, `RUN_WALKFORWARD`, `RUN_TAU_SWEEP`, `RUN_POLICY_SWEEP`, debug toggles, `SPREAD_PIPS`, `SLIPPAGE_PIPS`, `COMMISSION_PER_TRADE`, `WF_TAU`, `DYNAMIC_POLICY_KWARGS`, vol-guard settings, `DEBUG_SELECTED_PAIRS`, `DEBUG_SELECTED_MAX_FOLDS_PER_PAIR`.
  - `registries`: valid strategy IDs (`TF1`–`TF5`, `MR1`, `MR2`, `MR3`, `MR32`, `MR42`, `MR5`) and surfaces (`persistent`, `reactive_jpy`, `trend_vol`), read from the registries, not hardcoded.
- The env table is a literal table inside `mpml/spec.py`. It is documentation, **not** wired into
  production reads (that is deferred). To keep it from drifting, add a test that AST-scans MPML
  source for `os.environ` / `os.getenv` names and fails if any name is missing from the table
  or from an explicit, commented ignore list.
- Generate `docs/CLI_REFERENCE.md` from the same JSON. It includes the "source-edit-only"
  table and states clearly what cannot be set from the CLI or env.
- Commit both generated files. CI regenerates them and fails on a diff.

### Acceptance criteria

- Two consecutive runs of `python -m mpml.spec` produce byte-identical output. The output contains no timestamps and no commit SHA.
- The spec's argument list is generated from `build_parser()`, not copied. A test adds a dummy argument to the parser and confirms it appears in the spec.
- `choices` for `--behavioral-surface` and the strategy IDs are read from the registries and labeled with `choices_source`. No hardcoded lists. A test asserts the current values: surfaces `persistent`, `reactive_jpy` and `trend_vol`, and strategies `TF1` to `TF5`, `MR1`, `MR2`, `MR3`, `MR32`, `MR42` and `MR5`.
- The env-drift test passes on the current table. It fails when a variable read via `os.environ` or `os.getenv` in MPML source is missing from both the table and the commented ignore list. Verify this by temporarily removing one entry.
- In addition to name coverage, a few targeted subprocess tests verify the documented
  **semantics** of critical env entries: strict `DL_SIGNALS_ENABLED` truth values (including
  `1`/`yes` being false), `DL_PREDICTION_ARTIFACT_PATH` unset-versus-empty alias precedence,
  CLI-over-env experiment seed and surface precedence, and import-time-vs-runtime timing.
  These are consistency tests for the spec, not a new production config registry; AST
  discovery alone cannot verify defaults, parsers or precedence.
- The `rules` section lists exactly the issue codes implemented in PR 2. A test compares the two sets, so they can't diverge.
- `source_edit_only` lists the constants named in this document, and a test confirms each name exists as a module-level constant in `main.py`.
- `docs/CLI_REFERENCE.md` is generated from the spec JSON by a script, not edited by hand. It states which settings cannot be set from the CLI or env.
- Both generated files (the spec JSON and `CLI_REFERENCE.md`) are committed. CI regenerates them and fails on any diff.
- Running the spec generator in an empty temporary directory creates no files, directories or caches, and does not call `clear_cache()`.
- No production code path changes. `main.py` behavior and run outputs are identical before and after this PR.

## Deferred (not part of PRs 1 to 3)

| Item | Why it waits |
|---|---|
| A resolved launch/config object that production consumes | The audit shows import-time snapshots and direct `os.environ` reads in several modules. This is a real project of its own. The assistant doesn't need it, because it sets the environment of a subprocess |
| A declarative env registry wired into production reads | Same reason. PR 3's env table is documentation only |
| Artifact-aware preflight (`--resolve-artifacts`) | It loads prediction data and computes D1 features. Make it opt-in later and call it "resolve artifacts", not a dry run |
| Moving `_build_walkforward_execution_plan` to a lighter module | Only if importing `main` proves too heavy for preflight |
| A `--validate-only` flag on `main.py` | PR 2's module entry point covers the need without touching `_run_cli()` |

## Separate behavior-change PRs

One PR each. Each needs a compatibility note describing how existing results are affected.

| Change | Compatibility concern |
|---|---|
| `DL_SIGNALS_ENABLED` accepts `1/yes/on`, or raises on unrecognized values | Past runs launched with `=1` were baseline runs. Audit existing scripts and results first |
| `ACTIVE_PAIRS` rejects unknown names | Typos currently shrink the universe silently |
| Require a resolvable artifact when DL is enabled | Today the runtime is disabled with diagnostics, not an error |
| Stop inferring `missing_indicators_enabled` from the output directory name | Can change experiment semantics for existing run names |
| Define or rewrite `_scope_tf` / `_scope_mr` in the empty-spec error | Currently a `NameError`. Low risk |
| Call the PR 2 checks (`top_n`, timeline rule) from `_run_cli()` | Moves failures from late to early. Observable in timing, not in results |

## How `argus-cli` uses this

- **Tier-1 knowledge:** the PR 3 spec JSON, indexed together with the MPML commit SHA recorded at index time (the spec itself contains no SHA).
- **Layer-3 validation:** `python -m mpml.preflight <argv>` in a subprocess, with the candidate environment set on that process. It runs in MPML's own venv.
- **Rule ownership:** use preflight for its checks on supported MPML revisions. Keep assistant-side compatibility rules scoped to explicitly supported older revisions and flag stale profiles as unverified. Render and validate the same ordered argv, working directory, and environment overrides.
- **Golden set negatives:** one case per issue code in PR 2.
- **Not covered:** artifact resolution. The assistant must not claim a DL launch will actually enable DL, only that preflight found no problems.

## Run evidence for acceptance cases

The separate RAG project's `PLAN.md` describes the **external, unversioned**
`mpml_run_examples.tgz` archive. It is useful for RAG golden questions and optional local
integration recipes, but **is not a prerequisite for PR 1, 2 or 3**. Do not assume implementation tooling
can access it. Do not commit the bulk logs, model artifacts or historical market data to
MPML simply to satisfy this roadmap.

Commit small, self-contained characterization fixtures/snapshots produced from the unchanged
code before PR 1. Use temporary directories and synthetic data wherever possible. If the
local environment has representative market data and DL artifacts, the completed MR32
baseline and TF1 MLP Reactive-JPY consensus-young recipes described in the external RAG `PLAN.md` are
reasonable **optional** fresh before/after integration cells. Run these against the same
refactor revision; archived outcomes from older commits are not a parity oracle.

Record separately (a) required portable launch-interface tests and (b) optional real-run
numerical/manifest comparisons. If (b) cannot run, call it unverified, not passed. The
LSTM consensus-mature MR32 USDJPY-only case lacks exact-run evidence and is not a mandatory
MPML refactor fixture.
