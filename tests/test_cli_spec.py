from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from mpml.spec import build_spec


ROOT = Path(__file__).resolve().parents[1]

# Dynamic getenv(name) calls in experiment_surface_runtime.py receive only the
# five MPML metadata keys documented in ENVIRONMENT; static reads are still scanned.
ENV_READ_SCAN_IGNORES = {
    "src/experiment_surface_runtime.py": "dynamic _env_bool/_env_str lookup; concrete callers are inventoried",
    "src/repro.py": "resolve_experiment_seed receives env_var; its launch call uses EXPERIMENT_SEED",
}


def _environment_reads(path: Path) -> tuple[set[str], bool]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    has_dynamic_read = False

    def is_os_environ(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr == "environ"
            and isinstance(node.value, ast.Name)
            and node.value.id == "os"
        )

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            is_getenv = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "getenv"
            )
            is_mapping_get = (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and is_os_environ(node.func.value)
            )
            if is_getenv or is_mapping_get:
                if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                    names.add(node.args[0].value)
                else:
                    has_dynamic_read = True
        elif (
            isinstance(node, ast.Subscript)
            and isinstance(node.ctx, ast.Load)
            and is_os_environ(node.value)
        ):
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                names.add(node.slice.value)
        elif isinstance(node, ast.Compare) and any(is_os_environ(item) for item in node.comparators):
            if node.left and isinstance(node.left, ast.Constant) and isinstance(node.left.value, str):
                names.add(node.left.value)

    return names, has_dynamic_read


def _undocumented_environment_reads(
    source_root: Path,
    documented_names: set[str],
    ignored_names: set[str] | None = None,
) -> set[str]:
    ignored_names = ignored_names or set()
    discovered: set[str] = set()
    for source_dir in (source_root / "mpml", source_root / "src"):
        for path in source_dir.rglob("*.py"):
            names, dynamic = _environment_reads(path)
            if dynamic:
                relative = path.relative_to(source_root).as_posix()
                assert relative in ENV_READ_SCAN_IGNORES, f"Unreviewed dynamic environment read in {relative}"
            discovered.update(names)
    names, _ = _environment_reads(source_root / "main.py")
    discovered.update(names)
    return discovered - documented_names - ignored_names


def _subprocess(
    tmp_path: Path,
    code: str,
    env_updates: dict[str, str] | None = None,
    *,
    module: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("MPML_", "DL_", "BEHAVIORAL_", "EXPERIMENT_")) or key in {
            "ACTIVE_PAIRS",
            "EXPORT_SELECTOR_STATE_TIMELINE",
            "MIN_DL_TRAIN_COVERAGE_PCT",
            "MSML_REGIME",
            "OVERLAP_ONLY",
        }:
            env.pop(key)
    env["PYTHONPATH"] = str(ROOT)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path.parent / f"{tmp_path.name}-mplconfig")
    env.update(env_updates or {})
    command = [sys.executable, "-m", module] if module else [sys.executable, "-c", code]
    if module:
        command.extend(code.split())
    return subprocess.run(
        command,
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _env_entry(name: str) -> dict:
    return next(entry for entry in build_spec()["env"] if entry["name"] == name)


def test_spec_is_deterministic_and_has_no_filesystem_side_effects(tmp_path: Path) -> None:
    first = _subprocess(tmp_path, "", module="mpml.spec")
    second = _subprocess(tmp_path, "", module="mpml.spec")
    assert first.returncode == second.returncode == 0
    assert first.stdout == second.stdout
    document = json.loads(first.stdout)
    assert document["schema_version"] == 1
    assert not list(tmp_path.iterdir())
    assert "main" not in sys.modules


def test_arguments_are_built_from_parser_and_registries() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--injected-option", type=int, default=7, help="Test-only option.")
    spec = build_spec(parser)
    injected = next(arg for arg in spec["arguments"] if "--injected-option" in arg["flags"])
    assert injected["dest"] == "injected_option"
    assert injected["type"] == "int"
    assert injected["default"] == 7
    assert injected["help"] == "Test-only option."

    assert spec["registries"]["behavioral_surfaces"] == {
        "ids": ["persistent", "reactive_jpy", "trend_vol"],
        "choices_source": "behavioral_registry",
    }
    assert spec["registries"]["strategies"] == {
        "ids": ["MR1", "MR2", "MR3", "MR32", "MR42", "MR5", "TF1", "TF2", "TF3", "TF4", "TF5"],
        "choices_source": "strategy_registry",
    }
    surface = next(arg for arg in build_spec()["arguments"] if "--behavioral-surface" in arg["flags"])
    assert surface["choices_source"] == "behavioral_registry"
    strategy = next(arg for arg in build_spec()["arguments"] if "--strategy" in arg["flags"])
    assert strategy["choices_source"] is None
    assert strategy["post_parse_choices_source"] == "strategy_registry"
    assert strategy["post_parse_choices"] == spec["registries"]["strategies"]["ids"]


def test_environment_inventory_covers_static_reads_and_documents_dynamic_lookup() -> None:
    documented = {entry["name"] for entry in build_spec()["env"]}
    assert _undocumented_environment_reads(ROOT, documented) == set()

    # Verify that removing a statically read variable is detected.
    missing = _undocumented_environment_reads(
        ROOT, documented - {"DL_SIGNALS_ENABLED"}
    )
    assert missing == {"DL_SIGNALS_ENABLED"}

    # EXPERIMENT_SEED is read dynamically through resolve_experiment_seed()
    # in src/repro.py, so the AST scanner cannot detect its literal name.
    # Ensure that it remains explicitly documented in the inventory.
    assert "EXPERIMENT_SEED" in documented


def test_preflight_issue_code_inventory_matches_implemented_codes() -> None:
    preflight_path = ROOT / "mpml" / "preflight.py"
    tree = ast.parse(preflight_path.read_text(encoding="utf-8"))
    implemented = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_issue"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    spec_codes = {rule["code"] for rule in build_spec()["rules"]}
    assert spec_codes == implemented


def test_source_edit_only_settings_exist_as_main_module_constants() -> None:
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    assignments = {
        target.id
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        for target in (
            node.targets if isinstance(node, ast.Assign) else [node.target]
        )
        if isinstance(target, ast.Name)
    }
    listed = {entry["name"] for entry in build_spec()["source_edit_only"]}
    assert listed
    assert listed <= assignments


def test_generated_spec_and_reference_are_synchronized() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/generate_cli_reference.py", "--check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("true", True),
        ("TRUE", True),
        ("1", False),
        ("yes", False),
        ("on", False),
        ("false", False),
        ("", False),
    ],
)
def test_dl_signals_enabled_uses_strict_truth_semantics(
    tmp_path: Path, value: str, expected: bool
) -> None:
    result = _subprocess(
        tmp_path,
        "import json, main; print(json.dumps(main.DL_SIGNALS_ENABLED))",
        {"DL_SIGNALS_ENABLED": value},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) is expected
    assert _env_entry("DL_SIGNALS_ENABLED")["parser"] == "bool_strict"


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        (None, "/tmp/legacy-cube.parquet"),
        ("", "."),
    ],
)
def test_prediction_path_alias_only_applies_when_primary_is_unset(
    tmp_path: Path, primary: str | None, expected: str
) -> None:
    updates = {"DL_SIGNALS_CUBE_PATH": "/tmp/legacy-cube.parquet"}
    if primary is not None:
        updates["DL_PREDICTION_ARTIFACT_PATH"] = primary
    result = _subprocess(
        tmp_path,
        "import json; from src.dl_config import DL_PREDICTION_ARTIFACT_PATH; print(json.dumps(str(DL_PREDICTION_ARTIFACT_PATH)))",
        updates,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == expected
    assert _env_entry("DL_SIGNALS_CUBE_PATH")["alias_of"] == "DL_PREDICTION_ARTIFACT_PATH"


def test_cli_values_override_environment_seed_and_surface(tmp_path: Path) -> None:
    result = _subprocess(
        tmp_path,
        'from mpml.preflight import run_preflight; import json; print(json.dumps(run_preflight(["--experiment-seed", "123", "--behavioral-surface", "trend_vol"])))',
        {"EXPERIMENT_SEED": "not-an-int", "BEHAVIORAL_SURFACE": "not-registered"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [0, {"schema_version": 1, "issues": []}]


def test_environment_timing_distinguishes_import_snapshots_from_runtime_reads(
    tmp_path: Path,
) -> None:
    result = _subprocess(
        tmp_path,
        (
            "import json, os, main; "
            "from src.repro import resolve_experiment_seed; "
            "before = main.DL_SIGNALS_ENABLED; "
            "os.environ['DL_SIGNALS_ENABLED'] = 'true'; "
            "os.environ['EXPERIMENT_SEED'] = '19'; "
            "print(json.dumps([before, main.DL_SIGNALS_ENABLED, resolve_experiment_seed()]))"
        ),
        {"DL_SIGNALS_ENABLED": "false", "EXPERIMENT_SEED": "7"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == [False, False, 19]
    assert _env_entry("DL_SIGNALS_ENABLED")["timing"] == "import"
    assert _env_entry("EXPERIMENT_SEED")["timing"] == "runtime"
