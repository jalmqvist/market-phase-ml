"""Deterministic machine-readable description of the MPML launch surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mpml.cli import build_parser
from mpml.preflight import ISSUE_CODE_SUMMARIES
from mpml.behavioral import registry as behavioral_registry
from src.strategy_registry import get_default_strategy_registry


# This inventory documents production reads; it is deliberately not used to
# resolve configuration. Keep parser/default/timing details aligned with source.
ENVIRONMENT: tuple[dict[str, Any], ...] = (
    {
        "name": "ACTIVE_PAIRS",
        "parser": "csv",
        "default": "all registered pairs",
        "timing": "import",
        "read_in": ["main.py", "src/behavioral_artifact_resolver.py", "src/experiment_surface_runtime.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Comma-separated pair names; values are stripped and uppercased. Empty selects all pairs. The import-time filter and later metadata inference also read the process environment.",
    },
    {
        "name": "BEHAVIORAL_STATE_ID",
        "parser": "str",
        "default": None,
        "timing": "import",
        "read_in": ["main.py", "src/dl_config.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Stripped at the runtime read; blank becomes unset. Also snapshotted by src.dl_config for canonical identity helpers.",
    },
    {
        "name": "BEHAVIORAL_SURFACE",
        "parser": "enum",
        "default": "trend_vol",
        "timing": "runtime",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "CLI --behavioral-surface takes precedence; otherwise stripped environment value, then trend_vol.",
    },
    {
        "name": "BEHAVIORAL_SURFACE_ID",
        "parser": "str",
        "default": None,
        "timing": "import",
        "read_in": ["src/dl_config.py"],
        "alias_of": None,
        "scope": "metadata",
        "semantics": "Empty string becomes unset; part of canonical behavioral identity.",
    },
    {
        "name": "BEHAVIORAL_SURFACE_VERSION",
        "parser": "str",
        "default": None,
        "timing": "import",
        "read_in": ["src/dl_config.py"],
        "alias_of": None,
        "scope": "metadata",
        "semantics": "Empty string becomes unset; part of canonical behavioral identity.",
    },
    {
        "name": "DL_FEATURE_SET",
        "parser": "str",
        "default": "price_trend",
        "timing": "import",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Snapshotted in main.DL_SIGNAL_SURFACE; a runtime check also tests whether the environment value is blank when resolving an artifact.",
    },
    {
        "name": "DL_MODEL",
        "parser": "str",
        "default": "mlp",
        "timing": "import",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Snapshotted into main.DL_SIGNAL_SURFACE.",
    },
    {
        "name": "DL_PREDICTION_ARTIFACT_PATH",
        "parser": "path",
        "default": "sibling market-sentiment-ml artifact directory",
        "timing": "import",
        "read_in": ["main.py", "src/dl_config.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "src.dl_config snapshots this at import. If unset (not empty), DL_SIGNALS_CUBE_PATH is used; main.py also reads it at runtime.",
    },
    {
        "name": "DL_REGIME",
        "parser": "str",
        "default": "",
        "timing": "import",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Snapshotted into main.DL_SIGNAL_SURFACE.",
    },
    {
        "name": "DL_SIGNALS_CUBE_PATH",
        "parser": "path",
        "default": "sibling market-sentiment-ml artifact directory",
        "timing": "import",
        "read_in": ["src/dl_config.py"],
        "alias_of": "DL_PREDICTION_ARTIFACT_PATH",
        "scope": "launch_selector",
        "semantics": "Legacy path alias used only if DL_PREDICTION_ARTIFACT_PATH is unset. An empty primary value suppresses this alias.",
    },
    {
        "name": "DL_SIGNALS_ENABLED",
        "parser": "bool_strict",
        "default": False,
        "timing": "import",
        "read_in": ["main.py", "src/dl_config.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Only case-insensitive exact text true enables; 1, yes, on, empty, and other values are false.",
    },
    {
        "name": "DL_SURFACE_FEATURE_SET",
        "parser": "str",
        "default": "price_trend",
        "timing": "import",
        "read_in": ["src/dl_config.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Snapshotted in the legacy DL surface selector.",
    },
    {
        "name": "DL_SURFACE_MODEL",
        "parser": "str",
        "default": "lstm",
        "timing": "import",
        "read_in": ["src/dl_config.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Snapshotted in the legacy DL surface selector.",
    },
    {
        "name": "DL_SURFACE_REGIME",
        "parser": "str",
        "default": "HVTF",
        "timing": "import",
        "read_in": ["src/dl_config.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Snapshotted in the legacy DL surface selector.",
    },
    {
        "name": "DL_SURFACE_TARGET_HORIZON",
        "parser": "int",
        "default": 24,
        "timing": "import",
        "read_in": ["src/dl_config.py", "src/launch_env.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Parsed with int() when the legacy DL surface is imported; malformed input fails import.",
    },
    {
        "name": "EXPERIMENT_GENERATION",
        "parser": "enum",
        "default": None,
        "timing": "runtime",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "CLI hint takes precedence; when supplied, it must agree with the selected variant.",
    },
    {
        "name": "EXPERIMENT_SEED",
        "parser": "int",
        "default": 42,
        "timing": "runtime",
        "read_in": ["src/repro.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "CLI seed wins. Environment text is stripped; blank uses 42 and malformed nonempty text raises ValueError.",
    },
    {
        "name": "EXPERIMENT_VARIANT",
        "parser": "enum",
        "default": "A",
        "timing": "runtime",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "CLI value takes precedence; otherwise environment value, then A.",
    },
    {
        "name": "EXPORT_SELECTOR_STATE_TIMELINE",
        "parser": "bool_loose",
        "default": False,
        "timing": "import",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "diagnostic",
        "semantics": "Case-insensitive 1, true, yes, or on are true; every other value is false.",
    },
    {
        "name": "MIN_DL_TRAIN_COVERAGE_PCT",
        "parser": "float",
        "default": 5.0,
        "timing": "import",
        "read_in": ["main.py", "src/launch_env.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Parsed with float() when main is imported; malformed input fails import. Preflight invokes the same converter.",
    },
    {
        "name": "MPML_ARTIFACT_SOURCE",
        "parser": "str",
        "default": None,
        "timing": "runtime",
        "read_in": ["src/experiment_surface_runtime.py"],
        "alias_of": None,
        "scope": "metadata",
        "semantics": "Stripped non-empty value supplies artifact-source metadata when artifact metadata is absent.",
    },
    {
        "name": "MPML_AUTO_ANALYZE",
        "parser": "bool_loose",
        "default": True,
        "timing": "runtime",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Case-insensitive 1, true, yes, or on are true; every other value is false.",
    },
    {
        "name": "MPML_BROKER_DATA_DIR",
        "parser": "path",
        "default": "sibling market-sentiment-ml input directory",
        "timing": "runtime",
        "read_in": ["src/data.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Trimmed and expanded when constructing a broker_csv pipeline; blank uses the sibling-repository default.",
    },
    {
        "name": "MPML_DATA_SOURCE",
        "parser": "enum",
        "default": "yfinance",
        "timing": "runtime",
        "read_in": ["src/data.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Trimmed and lowercased; supported values are yfinance and broker_csv. Invalid values raise ValueError.",
    },
    {
        "name": "MPML_END_DATE",
        "parser": "str",
        "default": "2026-04-01",
        "timing": "import",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Snapshotted as a string when main is imported.",
    },
    {
        "name": "MPML_EVALUATION_PAIR_FAMILY",
        "parser": "str",
        "default": None,
        "timing": "runtime",
        "read_in": ["src/experiment_surface_runtime.py"],
        "alias_of": None,
        "scope": "metadata",
        "semantics": "Stripped non-empty value supplies evaluation-pair-family metadata when artifact metadata is absent.",
    },
    {
        "name": "MPML_FEATURE_SURFACE",
        "parser": "str",
        "default": None,
        "timing": "runtime",
        "read_in": ["src/experiment_surface_runtime.py"],
        "alias_of": None,
        "scope": "metadata",
        "semantics": "Stripped non-empty value supplies feature-surface metadata before the DL surface fallback.",
    },
    {
        "name": "MPML_OUTPUT_DIR",
        "parser": "path",
        "default": None,
        "timing": "runtime",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "CLI --output-dir wins. Also supplies the output-name hint used by imputation-awareness inference.",
    },
    {
        "name": "MPML_REVISION",
        "parser": "str",
        "default": None,
        "timing": "runtime",
        "read_in": ["mpml/runtime/phaseaware.py"],
        "alias_of": None,
        "scope": "metadata",
        "semantics": "Optional revision/provenance value.",
    },
    {
        "name": "MPML_RUNS_ROOT",
        "parser": "path",
        "default": "results_archive",
        "timing": "runtime",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Used as the parent for generated run output paths when no explicit output directory is selected.",
    },
    {
        "name": "MPML_SELECTOR_TRACE_PAIR",
        "parser": "str",
        "default": "",
        "timing": "runtime",
        "read_in": ["src/strategies.py"],
        "alias_of": None,
        "scope": "diagnostic",
        "semantics": "Pair selector for optional single-bar trace output; only used with MPML_SELECTOR_TRACE_PATH.",
    },
    {
        "name": "MPML_SELECTOR_TRACE_PATH",
        "parser": "path",
        "default": None,
        "timing": "runtime",
        "read_in": ["src/strategies.py"],
        "alias_of": None,
        "scope": "diagnostic",
        "semantics": "Enables optional selector tracing for the matching pair and timestamp.",
    },
    {
        "name": "MPML_SELECTOR_TRACE_TIMESTAMP",
        "parser": "str",
        "default": "",
        "timing": "runtime",
        "read_in": ["src/strategies.py"],
        "alias_of": None,
        "scope": "diagnostic",
        "semantics": "Timestamp selector for optional single-bar trace output.",
    },
    {
        "name": "MPML_SENTIMENT_SURFACE",
        "parser": "str",
        "default": None,
        "timing": "runtime",
        "read_in": ["src/experiment_surface_runtime.py"],
        "alias_of": None,
        "scope": "metadata",
        "semantics": "Stripped surface value; supported normalized values are sentiment, no_sentiment, and none.",
    },
    {
        "name": "MPML_START_DATE",
        "parser": "str",
        "default": "2005-01-01",
        "timing": "import",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Snapshotted as a string when main is imported.",
    },
    {
        "name": "MPML_TRAINING_PAIR_FAMILY",
        "parser": "str",
        "default": None,
        "timing": "runtime",
        "read_in": ["src/experiment_surface_runtime.py"],
        "alias_of": None,
        "scope": "metadata",
        "semantics": "Stripped non-empty value supplies training-pair-family metadata when artifact metadata is absent.",
    },
    {
        "name": "MSML_REGIME",
        "parser": "str",
        "default": "selected DL regime or unknown",
        "timing": "runtime",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Trimmed and uppercased; fallback depends on the selected behavioral surface and resolved state.",
    },
    {
        "name": "OVERLAP_ONLY",
        "parser": "bool_loose",
        "default": False,
        "timing": "runtime",
        "read_in": ["main.py"],
        "alias_of": None,
        "scope": "launch_selector",
        "semantics": "Case-insensitive 1, true, yes, or on are true; every other value is false.",
    },
)


SOURCE_EDIT_ONLY = (
    ("RUN_IN_SAMPLE_ABLATION", "experiment toggles"),
    ("RUN_WALKFORWARD", "experiment toggles"),
    ("RUN_TAU_SWEEP", "experiment toggles"),
    ("RUN_POLICY_SWEEP", "experiment toggles"),
    ("DL_DEBUG_VERBOSE", "debug toggles"),
    ("DEBUG_BASELINE_KEYS", "debug toggles"),
    ("DEBUG_FEATURE_COLUMNS", "debug toggles"),
    ("DEBUG_SIGNAL_TYPES", "debug toggles"),
    ("DEBUG_VOL_GUARD", "debug toggles"),
    ("DEBUG_SAVE_SELECTED_SERIES", "debug toggles"),
    ("DEBUG_SAVE_EQUITY_SERIES", "debug toggles"),
    ("DEBUG_SELECTED_PAIRS", "debug toggles"),
    ("DEBUG_SELECTED_MAX_FOLDS_PER_PAIR", "debug toggles"),
    ("SPREAD_PIPS", "transaction costs"),
    ("SLIPPAGE_PIPS", "transaction costs"),
    ("COMMISSION_PER_TRADE", "transaction costs"),
    ("WF_TAU", "walk-forward selector"),
    ("DYNAMIC_POLICY_KWARGS", "walk-forward selector"),
    ("USE_VOL_GUARD", "volatility guard"),
    ("VOL_GUARD_Q", "volatility guard"),
    ("VOL_GUARD_MODE", "volatility guard"),
    ("VOL_FEATURE", "volatility guard"),
    ("USD_QUOTE_VOL_SPIKE_OVERRIDE", "volatility guard"),
    ("VOL_GUARD_NEAR_MULT", "volatility guard"),
)

PRECEDENCE = (
    {
        "setting": "behavioral_surface",
        "order": ["--behavioral-surface", "BEHAVIORAL_SURFACE", "trend_vol"],
    },
    {
        "setting": "experiment_seed",
        "order": ["--experiment-seed", "EXPERIMENT_SEED", "42"],
    },
    {
        "setting": "experiment_variant",
        "order": ["--experiment-variant", "EXPERIMENT_VARIANT", "A"],
    },
    {
        "setting": "output_directory",
        "order": ["--output-dir", "MPML_OUTPUT_DIR", "MPML_RUNS_ROOT/results_archive default"],
    },
)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return str(value)


def _action_name(action: argparse.Action) -> str:
    known_names = {
        "_AppendAction": "append",
        "_CountAction": "count",
        "_StoreAction": "store",
        "_StoreConstAction": "store_const",
        "_StoreFalseAction": "store_false",
        "_StoreTrueAction": "store_true",
    }
    return known_names.get(type(action).__name__, type(action).__name__)


def _argument_spec(action: argparse.Action, strategy_ids: list[str]) -> dict[str, Any]:
    flags = list(action.option_strings)
    argument: dict[str, Any] = {
        "flags": flags,
        "dest": action.dest,
        "action": _action_name(action),
        "type": getattr(action.type, "__name__", None),
        "default": _json_value(action.default),
        "nargs": action.nargs,
        "help": action.help,
        "choices": (
            sorted(action.choices, key=str)
            if action.choices is not None
            else None
        ),
        "choices_source": "static" if action.choices is not None else None,
    }
    if "--behavioral-surface" in flags:
        argument["choices_source"] = "behavioral_registry"
    if action.dest == "strategy":
        argument["post_parse_choices"] = strategy_ids
        argument["post_parse_choices_source"] = "strategy_registry"
    elif action.dest == "phaseaware_tf":
        argument["post_parse_choices"] = [
            item
            for item in strategy_ids
            if get_default_strategy_registry().get(item).family.casefold()
            == "trendfollowing".casefold()
        ]
        argument["post_parse_choices_source"] = "strategy_registry"
    elif action.dest == "phaseaware_mr":
        argument["post_parse_choices"] = [
            item
            for item in strategy_ids
            if get_default_strategy_registry().get(item).family.casefold()
            == "meanreversion".casefold()
        ]
        argument["post_parse_choices_source"] = "strategy_registry"
    return argument


def build_spec(parser: argparse.ArgumentParser | None = None) -> dict[str, Any]:
    """Build the launch specification from the parser and current registries."""
    strategy_registry = get_default_strategy_registry()
    strategy_ids = strategy_registry.available()
    surfaces = behavioral_registry.available()
    parser = parser or build_parser()
    return {
        "schema_version": 1,
        "arguments": [
            _argument_spec(action, strategy_ids)
            for action in parser._actions
            if action.dest != "help"
        ],
        "registries": {
            "behavioral_surfaces": {
                "ids": surfaces,
                "choices_source": "behavioral_registry",
            },
            "strategies": {
                "ids": strategy_ids,
                "choices_source": "strategy_registry",
            },
        },
        "precedence": list(PRECEDENCE),
        "env": sorted(ENVIRONMENT, key=lambda entry: entry["name"]),
        "rules": [
            {"code": code, "summary": summary}
            for code, summary in sorted(ISSUE_CODE_SUMMARIES.items())
        ],
        "source_edit_only": [
            {"name": name, "category": category}
            for name, category in SOURCE_EDIT_ONLY
        ],
    }


def spec_json(spec: dict[str, Any] | None = None) -> str:
    """Serialize a specification in its canonical committed format."""
    return json.dumps(spec if spec is not None else build_spec(), indent=2, sort_keys=True) + "\n"


def _cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) if value else "—"
    return str(value).replace("|", "\\|").replace("\n", " ")


def render_reference(spec: dict[str, Any] | None = None) -> str:
    """Render the human-readable reference from the machine specification."""
    spec = spec if spec is not None else build_spec()
    rows = [
        "# MPML CLI Reference",
        "",
        "<!-- Generated from docs/cli_spec.json by `python scripts/generate_cli_reference.py`; do not edit by hand. -->",
        "",
        "This reference describes the existing MPML launch surface. It does not change runtime behavior.",
        "Generate or check the committed files with `python scripts/generate_cli_reference.py` or",
        "`python scripts/generate_cli_reference.py --check`.",
        "",
        "## Command-line arguments",
        "",
        "| Flags | Destination | Action | Type | Default | Choices | Help |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for argument in spec["arguments"]:
        choices = argument["choices"]
        if choices is None:
            choices = argument.get("post_parse_choices")
        rows.append(
            "| "
            + " | ".join(
                _cell(argument[key])
                for key in ("flags", "dest", "action", "type", "default")
            )
            + f" | {_cell(choices)}"
            + (
                f" ({_cell(argument.get('choices_source') or argument.get('post_parse_choices_source'))})"
                if choices is not None
                else ""
            )
            + f" | {_cell(argument['help'])} |"
        )
    rows.extend(
        [
            "",
            "## Registries",
            "",
            "| Registry | Identifiers | Source |",
            "| --- | --- | --- |",
        ]
    )
    for key, label in (
        ("strategies", "Strategies"),
        ("behavioral_surfaces", "Behavioral surfaces"),
    ):
        registry = spec["registries"][key]
        rows.append(
            f"| {label} | {_cell(registry['ids'])} | {_cell(registry['choices_source'])} |"
        )
    rows.extend(
        [
            "",
            "Strategy choices on `--strategy`, `--phaseaware-tf`, and `--phaseaware-mr` are validated after parsing.",
            "",
            "## Precedence",
            "",
            "| Setting | Order (highest priority first) |",
            "| --- | --- |",
        ]
    )
    for rule in spec["precedence"]:
        rows.append(f"| {_cell(rule['setting'])} | {_cell(rule['order'])} |")
    rows.extend(
        [
            "",
            "## Environment variables",
            "",
            "Import-time settings are snapshotted when their module is imported; runtime settings are read during execution.",
            "",
            "| Variable | Parser | Default | Timing | Scope | Source files | Semantics |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for entry in spec["env"]:
        rows.append(
            "| "
            + " | ".join(
                _cell(entry[key])
                for key in ("name", "parser", "default", "timing", "scope", "read_in")
            )
            + f" | {_cell(entry['semantics'])} |"
        )
    rows.extend(
        [
            "",
            "## Preflight issue codes",
            "",
            "| Code | Summary |",
            "| --- | --- |",
        ]
    )
    rows.extend(f"| {rule['code']} | {rule['summary']} |" for rule in spec["rules"])
    rows.extend(
        [
            "",
            "## Source-edit-only settings",
            "",
            "These module-level settings in `main.py` are not exposed as command-line arguments or environment variables.",
            "Changing them requires editing source; they are not part of the supported launch configuration surface.",
            "",
            "| Setting | Category |",
            "| --- | --- |",
        ]
    )
    rows.extend(
        f"| `{setting['name']}` | {setting['category']} |"
        for setting in spec["source_edit_only"]
    )
    return "\n".join(rows) + "\n"


def main() -> None:
    print(spec_json(), end="")


if __name__ == "__main__":
    main()
