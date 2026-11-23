from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _preflight(
    tmp_path: Path,
    *argv: str,
    env_updates: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("MPML_", "DL_", "BEHAVIORAL_", "EXPERIMENT_")) or key in {
            "ACTIVE_PAIRS",
            "EXPORT_SELECTOR_STATE_TIMELINE",
            "MIN_DL_TRAIN_COVERAGE_PCT",
        }:
            env.pop(key)
    env["PYTHONPATH"] = str(ROOT)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path.parent / f"{tmp_path.name}-mplconfig")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    env.update(env_updates or {})
    result = subprocess.run(
        [sys.executable, "-m", "mpml.preflight", *argv],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, json.loads(result.stdout)


def _codes(document: dict) -> set[str]:
    return {issue["code"] for issue in document["issues"]}


@pytest.mark.parametrize(
    ("argv", "env", "code"),
    [
        (("--unknown-option",), {}, "ARGPARSE"),
        (
            ("--phaseaware-tf", "TF2", "--phaseaware-tf", "TF3"),
            {},
            "PHASEAWARE_REPEATED",
        ),
        ((), {"BEHAVIORAL_SURFACE": "unknown"}, "SURFACE_UNKNOWN"),
        (("--strategy", "UNKNOWN"), {}, "STRATEGY_UNKNOWN"),
        (("--phaseaware-tf", "MR2"), {}, "PHASEAWARE_ROLE_INVALID"),
        ((), {"EXPERIMENT_VARIANT": "A", "EXPERIMENT_GENERATION": "gen2"}, "GENERATION_VARIANT_MISMATCH"),
        (("--recommendation-top-n", "0"), {}, "TOP_N_NOT_POSITIVE"),
        (
            (
                "--export-dl-conditional-portfolio-timeline",
                "--strategy",
                "TF3",
            ),
            {},
            "TIMELINE_NEEDS_DYNAMIC_SELECTOR",
        ),
        ((), {"MPML_DATA_SOURCE": "bad-source"}, "ENV_INVALID_VALUE"),
        ((), {"DL_SIGNALS_ENABLED": "yes"}, "ENV_BOOL_PARSED_FALSE"),
        ((), {"ACTIVE_PAIRS": "USDJPY,UNKNOWN"}, "ACTIVE_PAIRS_UNKNOWN"),
        (
            (),
            {
                "DL_SIGNALS_ENABLED": "true",
                "DL_PREDICTION_ARTIFACT_PATH": "/missing/preflight.parquet",
            },
            "DL_ARTIFACT_PATH_MISSING",
        ),
        (
            (),
            {
                "DL_PREDICTION_ARTIFACT_PATH": "",
                "DL_SIGNALS_CUBE_PATH": "/unused/alias.parquet",
            },
            "DL_ALIAS_SUPPRESSED",
        ),
        (("--output-dir", "already-aware"), {}, "OUTPUT_NAME_INFERS_AWARENESS"),
    ],
)
def test_preflight_reports_each_launch_issue(
    tmp_path: Path,
    argv: tuple[str, ...],
    env: dict[str, str],
    code: str,
) -> None:
    result, document = _preflight(tmp_path, *argv, env_updates=env)
    should_fail = code not in {
        "ENV_BOOL_PARSED_FALSE",
        "ACTIVE_PAIRS_UNKNOWN",
        "DL_ARTIFACT_PATH_MISSING",
        "DL_ALIAS_SUPPRESSED",
        "OUTPUT_NAME_INFERS_AWARENESS",
    }
    assert (result.returncode != 0) is should_fail
    assert code in _codes(document)


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("EXPERIMENT_SEED", "bad"),
        ("MIN_DL_TRAIN_COVERAGE_PCT", "bad"),
        ("DL_SURFACE_TARGET_HORIZON", "bad"),
        ("MPML_DATA_SOURCE", "unsupported"),
    ],
)
def test_invalid_environment_values_are_structured(
    tmp_path: Path,
    variable: str,
    value: str,
) -> None:
    result, document = _preflight(
        tmp_path,
        env_updates={variable: value},
    )
    assert result.returncode != 0
    issues = [
        issue
        for issue in document["issues"]
        if issue["code"] == "ENV_INVALID_VALUE"
    ]
    assert len(issues) == 1
    assert variable in issues[0]["source"]


def test_cli_seed_overrides_invalid_environment_seed(tmp_path: Path) -> None:
    result, document = _preflight(
        tmp_path,
        "--experiment-seed",
        "123",
        env_updates={"EXPERIMENT_SEED": "not-an-int"},
    )
    assert result.returncode == 0
    assert document["issues"] == []


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (("--export-dl-conditional-portfolio-timeline",), False),
        (
            (
                "--export-dl-conditional-portfolio-timeline",
                "--phaseaware-tf",
                "TF2",
                "--phaseaware-mr",
                "MR2",
                "--strategy",
                "TF2",
                "--strategy",
                "MR2",
            ),
            False,
        ),
        (
            (
                "--export-dl-conditional-portfolio-timeline",
                "--strategy",
                "TF4",
            ),
            True,
        ),
        (
            (
                "--export-dl-conditional-portfolio-timeline",
                "--strategy",
                "MR42",
            ),
            True,
        ),
        (
            (
                "--export-dl-conditional-portfolio-timeline",
                "--strategy",
                "TF3",
            ),
            True,
        ),
    ],
)
def test_timeline_preflight_uses_effective_scope(
    tmp_path: Path,
    argv: tuple[str, ...],
    expected: bool,
) -> None:
    result, document = _preflight(tmp_path, *argv)
    assert ("TIMELINE_NEEDS_DYNAMIC_SELECTOR" in _codes(document)) is expected
    assert result.returncode == (1 if expected else 0)


def test_passive_timeline_environment_flag_does_not_require_selector(
    tmp_path: Path,
) -> None:
    result, document = _preflight(
        tmp_path,
        "--strategy",
        "TF4",
        env_updates={"EXPORT_SELECTOR_STATE_TIMELINE": "true"},
    )
    assert result.returncode == 0
    assert "TIMELINE_NEEDS_DYNAMIC_SELECTOR" not in _codes(document)


def test_output_directory_collision_is_advisory(tmp_path: Path) -> None:
    output = tmp_path / "existing-output"
    output.mkdir()
    result, document = _preflight(tmp_path, "--output-dir", str(output))
    assert result.returncode != 0
    issue = next(issue for issue in document["issues"] if issue["code"] == "OUTPUT_DIR_EXISTS")
    assert issue["severity"] == "error"
    assert output.is_dir()


def test_valid_baseline_and_dl_launches_have_no_issues(tmp_path: Path) -> None:
    baseline, baseline_document = _preflight(tmp_path)
    assert baseline.returncode == 0
    assert baseline_document["issues"] == []

    artifact = tmp_path / "available.parquet"
    artifact.touch()
    dl_result, dl_document = _preflight(
        tmp_path,
        env_updates={
            "DL_SIGNALS_ENABLED": "true",
            "DL_PREDICTION_ARTIFACT_PATH": str(artifact),
        },
    )
    assert dl_result.returncode == 0
    assert dl_document["issues"] == []


def test_dl_enabled_without_explicit_path_is_not_warned(tmp_path: Path) -> None:
    result, document = _preflight(
        tmp_path,
        env_updates={"DL_SIGNALS_ENABLED": "true"},
    )
    assert result.returncode == 0
    assert document["issues"] == []


def test_preflight_does_not_create_files_or_construct_pipeline(tmp_path: Path) -> None:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("MPML_", "DL_", "BEHAVIORAL_", "EXPERIMENT_")) or key in {
            "ACTIVE_PAIRS",
            "EXPORT_SELECTOR_STATE_TIMELINE",
            "MIN_DL_TRAIN_COVERAGE_PCT",
        }:
            env.pop(key)
    env["PYTHONPATH"] = str(ROOT)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path.parent / f"{tmp_path.name}-mplconfig")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    code = """
        import json
        import main
        from unittest.mock import patch
        from mpml.preflight import run_preflight
        with patch.object(main, "MarketDataPipeline", side_effect=AssertionError("pipeline constructed")):
            status, result = run_preflight([])
        print(json.dumps({"status": status, **result}))
    """
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["issues"] == []
    assert list(tmp_path.iterdir()) == []
    assert not (tmp_path / "data" / "cache").exists()


def test_shared_cli_repetition_predicate_keeps_argparse_exit_behavior() -> None:
    import main

    with (
        patch.object(main, "main") as run_main,
        patch(
            "main.validate_phaseaware_repetitions",
            wraps=main.validate_phaseaware_repetitions,
        ) as validator,
        pytest.raises(SystemExit) as exc_info,
    ):
        main._run_cli(["--phaseaware-tf", "TF2", "--phaseaware-tf", "TF3"])
    assert exc_info.value.code == 2
    run_main.assert_not_called()
    validator.assert_called_once()


def test_recommendation_builder_uses_shared_top_n_predicate() -> None:
    from src import recommendation

    with patch.object(
        recommendation,
        "validate_recommendation_top_n",
        side_effect=ValueError("sentinel"),
    ) as validator:
        with pytest.raises(ValueError, match="sentinel"):
            recommendation.recommendations_from_evaluations([], top_n=1)
    validator.assert_called_once_with(1)


def test_main_keeps_shared_surface_and_generation_checks_in_run_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import main

    for name in (
        "BEHAVIORAL_SURFACE",
        "EXPERIMENT_GENERATION",
        "EXPERIMENT_SEED",
        "EXPERIMENT_VARIANT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(main, "DL_SIGNALS_ENABLED", False)
    monkeypatch.setattr(main, "set_global_seed", lambda _seed: {})
    monkeypatch.setattr(main, "_configure_debug", lambda _debug: None)
    with patch(
        "main._resolve_behavioral_surface",
        wraps=main._resolve_behavioral_surface,
    ) as surface_check:
        with pytest.raises(ValueError, match="Unknown --behavioral-surface"):
            main.main(behavioral_surface="unknown")
    surface_check.assert_called_once_with("unknown")

    runtime = SimpleNamespace(
        enabled=False,
        artifact_path=None,
        surface_selector=main.DL_SIGNAL_SURFACE,
        state_id=None,
        d1_predictions=None,
        diagnostics=[],
    )
    with (
        patch.object(main, "resolve_behavioral_artifact_runtime", return_value=runtime),
        patch(
            "main._validate_generation_for_variant",
            wraps=main._validate_generation_for_variant,
        ) as generation_check,
        pytest.raises(ValueError, match="Conflicting experiment semantic inputs"),
    ):
        main.main(experiment_variant="A", experiment_generation="gen2")
    generation_check.assert_called_once()


def test_preflight_in_empty_directory_leaves_it_empty(tmp_path: Path) -> None:
    result, document = _preflight(tmp_path)
    assert result.returncode == 0
    assert document["issues"] == []
    assert list(tmp_path.iterdir()) == []
