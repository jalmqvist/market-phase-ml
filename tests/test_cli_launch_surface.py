"""Data-independent characterization for the MPML command-line boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest


ROOT = Path(__file__).resolve().parents[1]
CHARACTERIZATION = ROOT / "tests" / "characterization"


def _isolated_python(
    tmp_path: Path,
    code: str,
    *,
    env_updates: dict[str, str] | None = None,
    unset: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("MPML_", "DL_", "BEHAVIORAL_", "EXPERIMENT_")) or key in {
            "ACTIVE_PAIRS",
            "EXPORT_SELECTOR_STATE_TIMELINE",
            "OVERLAP_ONLY",
            "MSML_REGIME",
            "MIN_DL_TRAIN_COVERAGE_PCT",
        }:
            env.pop(key)
    env["PYTHONPATH"] = str(ROOT)
    env["MPLBACKEND"] = "Agg"
    env["MPLCONFIGDIR"] = str(tmp_path.parent / f"{tmp_path.name}-mplconfig")
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    for key in unset:
        env.pop(key, None)
    env.update(env_updates or {})
    return subprocess.run(
        [sys.executable, "-c", dedent(code)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


PARSER_ARGV = {
    "no_arguments": [],
    "all_flags": [
        "--output-dir", "relative/path",
        "--experiment-generation", "gen1",
        "--experiment-variant", "A",
        "--experiment-seed", "123",
        "--behavioral-surface", "trend_vol",
        "--recommendations-per-pair",
        "--recommendation-top-n", "3",
        "--strategy", "TF4",
        "--phaseaware-tf", "TF2",
        "--phaseaware-mr", "MR2",
        "--debug",
        "--export-dl-conditional-portfolio-timeline",
    ],
    "repeated_strategy_duplicates": ["--strategy", "TF4", "--strategy", "TF4"],
    "repeated_phaseaware_tf": ["--phaseaware-tf", "TF2", "--phaseaware-tf", "TF3"],
    "repeated_phaseaware_mr": ["--phaseaware-mr", "MR2", "--phaseaware-mr", "MR3"],
    "invalid_behavioral_surface": ["--behavioral-surface", "bogus"],
    "invalid_experiment_variant": ["--experiment-variant", "Z"],
    "invalid_experiment_generation": ["--experiment-generation", "gen3"],
    "invalid_experiment_seed": ["--experiment-seed", "abc"],
    "recommendation_top_n_zero": ["--recommendation-top-n", "0"],
    "output_dir_relative": ["--output-dir", "relative/path"],
    "store_true_flags": [
        "--debug",
        "--recommendations-per-pair",
        "--export-dl-conditional-portfolio-timeline",
    ],
}


def test_cli_arguments_match_committed_pre_refactor_snapshot(tmp_path: Path) -> None:
    code = """
        import contextlib
        import io
        import json
        import main
        from pathlib import Path

        cases = __PARSER_ARGV__
        outcomes = []
        captured = {}
        def record(**kwargs):
            captured.update(kwargs)
        main.main = record
        for name, argv in cases.items():
            captured.clear()
            try:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    main._run_cli(argv)
                kwargs = dict(captured)
                if kwargs["output_dir"] is not None:
                    kwargs["output_dir"] = str(kwargs["output_dir"])
                outcome = {"kwargs": kwargs}
            except SystemExit as exc:
                outcome = {"system_exit": exc.code}
            outcomes.append({"name": name, "result": outcome})
        print(json.dumps(outcomes))
        """
    result = _isolated_python(
        tmp_path,
        code.replace("__PARSER_ARGV__", json.dumps(PARSER_ARGV)),
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    expected = json.loads((CHARACTERIZATION / "parser_snapshots.json").read_text())
    assert json.loads(result.stdout) == expected


def test_help_matches_committed_snapshot_including_registry_text(tmp_path: Path) -> None:
    result = _isolated_python(
        tmp_path,
        """
        import sys
        sys.argv[0] = "mpml"
        import main
        main._run_cli(["--help"])
        """,
        env_updates={"COLUMNS": "100", "DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0
    assert result.stdout == (CHARACTERIZATION / "parser_help.txt").read_text()


def test_cli_without_argv_uses_sys_argv(tmp_path: Path) -> None:
    result = _isolated_python(
        tmp_path,
        """
        import contextlib
        import io
        import json
        import sys
        import main
        sys.argv = ["mpml", "--experiment-seed", "17"]
        captured = {}
        main.main = lambda **kwargs: captured.update(kwargs)
        with contextlib.redirect_stdout(io.StringIO()):
            main._run_cli()
        print(json.dumps({"seed": captured["experiment_seed"]}))
        """,
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"seed": 17}


def test_parser_choices_follow_runtime_registries() -> None:
    from mpml.behavioral import registry as behavioral_registry
    from mpml.cli import build_parser
    from src.strategy_registry import get_default_strategy_registry

    parser = build_parser()
    actions = {action.dest: action for action in parser._actions}
    assert actions["behavioral_surface"].choices == behavioral_registry.available()
    assert actions["strategy"].choices is None
    assert "Strategy Registry" in actions["strategy"].help
    assert str(get_default_strategy_registry().available()) in actions["strategy"].help


@pytest.mark.parametrize("module_name", ["mpml.cli", "main"])
def test_imports_are_silent_and_do_not_create_files(
    tmp_path: Path, module_name: str
) -> None:
    result = _isolated_python(
        tmp_path,
        f"import {module_name}",
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert result.stderr == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("argv", "expected_exit"),
    [
        (["--help"], 0),
        (["--behavioral-surface", "bogus"], 2),
        (["--phaseaware-tf", "TF2", "--phaseaware-tf", "TF3"], 2),
    ],
)
def test_help_and_rejected_cli_arguments_preserve_cache(
    tmp_path: Path, argv: list[str], expected_exit: int
) -> None:
    (tmp_path / "data" / "cache").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "keep.pkl").write_text("keep")
    (tmp_path / "data" / "cache" / "keep.txt").write_text("keep")
    code = """
        import contextlib
        import io
        import json
        import main
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                main._run_cli(__CLI_ARGV__)
        except SystemExit as exc:
            print(json.dumps({"exit": exc.code}))
        """
    result = _isolated_python(
        tmp_path,
        code.replace("__CLI_ARGV__", json.dumps(argv)),
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"exit": expected_exit}
    assert (tmp_path / "data" / "cache" / "keep.pkl").read_text() == "keep"
    assert (tmp_path / "data" / "cache" / "keep.txt").read_text() == "keep"


def test_valid_cli_clears_only_cache_pickles_before_main_dispatch(tmp_path: Path) -> None:
    (tmp_path / "data" / "cache").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "remove.pkl").write_text("remove")
    (tmp_path / "data" / "cache" / "keep.txt").write_text("keep")
    result = _isolated_python(
        tmp_path,
        """
        import contextlib
        import io
        import json
        from pathlib import Path
        import main

        seen = {}
        def record(**kwargs):
            cache = Path("data/cache")
            seen["kwargs"] = kwargs
            seen["cache"] = sorted(path.name for path in cache.iterdir())
        main.main = record
        with contextlib.redirect_stdout(io.StringIO()):
            main._run_cli(["--strategy", "TF4"])
        print(json.dumps(seen))
        """,
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    seen = json.loads(result.stdout)
    assert seen["cache"] == ["keep.txt"]
    assert seen["kwargs"]["strategy"] == ["TF4"]


def test_import_side_effect_snapshot_records_intentional_behavior_change(
    tmp_path: Path,
) -> None:
    before = json.loads(
        (CHARACTERIZATION / "legacy_import_side_effects.json").read_text()
    )
    assert "[CONFIG]" in before["import_stdout"]
    assert before["probe_pkl_removed"] is True
    (tmp_path / "data" / "cache").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "keep.pkl").write_text("keep")
    (tmp_path / "data" / "cache" / "keep.txt").write_text("keep")
    result = _isolated_python(
        tmp_path,
        """
        import json
        import main
        from pathlib import Path
        cache = Path("data/cache")
        print(json.dumps({
            "cache_after_import": sorted(path.name for path in cache.iterdir()),
            "probe_pkl_removed": not (cache / "keep.pkl").exists(),
            "non_pkl_preserved": (cache / "keep.txt").exists(),
        }))
        """,
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "cache_after_import": ["keep.pkl", "keep.txt"],
        "probe_pkl_removed": False,
        "non_pkl_preserved": True,
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("true", True),
        ("True", True),
        ("1", False),
        ("yes", False),
        ("on", False),
        ("garbage", False),
    ],
)
def test_dl_strict_boolean_is_characterized_in_both_importers(
    tmp_path: Path, value: str | None, expected: bool
) -> None:
    updates = {"DL_SIGNALS_ENABLED": value} if value is not None else {}
    loose_updates = (
        {"EXPORT_SELECTOR_STATE_TIMELINE": value} if value is not None else {}
    )
    unset = (
        ()
        if value is not None
        else ("DL_SIGNALS_ENABLED", "EXPORT_SELECTOR_STATE_TIMELINE")
    )
    result = _isolated_python(
        tmp_path,
        """
        import json
        import main
        import src.dl_config
        print(json.dumps([
            main.DL_SIGNALS_ENABLED,
            src.dl_config.DL_SIGNALS_ENABLED,
            main.EXPORT_SELECTOR_STATE_TIMELINE,
        ]))
        """,
        env_updates={**updates, **loose_updates},
        unset=unset,
    )
    assert result.returncode == 0, result.stderr
    loose_expected = bool(value and value.lower() in {"1", "true", "yes", "on"})
    assert json.loads(result.stdout) == [expected, expected, loose_expected]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("   ", None),
        ("EURUSD, unknown, gbpusd", ["EURUSD", "GBPUSD", "UNKNOWN"]),
        ("eurusd,,  ", ["EURUSD"]),
    ],
)
def test_active_pairs_are_trimmed_uppercased_and_unknown_names_retained(
    tmp_path: Path, value: str | None, expected: list[str] | None
) -> None:
    updates = {"ACTIVE_PAIRS": value} if value is not None else {}
    unset = () if value is not None else ("ACTIVE_PAIRS",)
    result = _isolated_python(
        tmp_path,
        """
        import json
        import main
        print(json.dumps(sorted(main.ACTIVE_PAIRS) if main.ACTIVE_PAIRS is not None else None))
        """,
        env_updates={**updates, "DL_SIGNALS_ENABLED": "false"},
        unset=unset,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == expected


def test_artifact_alias_uses_cube_path_only_when_primary_is_unset(
    tmp_path: Path,
) -> None:
    result = _isolated_python(
        tmp_path,
        """
        import json
        import os
        from pathlib import Path
        import src.dl_config as config
        print(json.dumps(str(config.DL_PREDICTION_ARTIFACT_PATH)))
        """,
        env_updates={"DL_SIGNALS_CUBE_PATH": "alias.parquet"},
        unset=("DL_PREDICTION_ARTIFACT_PATH",),
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == "alias.parquet"
    for primary, expected in (("", "."), ("primary.parquet", "primary.parquet")):
        case_dir = tmp_path / ("empty-primary" if not primary else "explicit-primary")
        case_dir.mkdir()
        case = _isolated_python(
            case_dir,
            """
            import json
            import src.dl_config as config
            print(json.dumps(str(config.DL_PREDICTION_ARTIFACT_PATH)))
            """,
            env_updates={
                "DL_PREDICTION_ARTIFACT_PATH": primary,
                "DL_SIGNALS_CUBE_PATH": "alias.parquet",
            },
            unset=("DL_PREDICTION_ARTIFACT_PATH", "DL_SIGNALS_CUBE_PATH"),
        )
        assert case.returncode == 0, case.stderr
        assert json.loads(case.stdout) == expected


def test_dl_enabled_without_resolvable_or_existing_explicit_artifact_disables_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import behavioral_artifact_resolver as resolver

    common = {
        "dl_runtime_enabled": True,
        "behavioral_surface_id": "trend_vol",
        "behavioral_surface_version": "1",
        "dl_surface": {"model": "mlp", "target_horizon": 24},
    }
    with monkeypatch.context() as patcher:
        patcher.setattr(resolver, "_resolve_artifact_path", lambda _path: None)
        undiscovered = resolver.resolve_behavioral_artifact_runtime(**common)
    assert not undiscovered.enabled
    assert any("no prediction artifact was resolved" in line for line in undiscovered.diagnostics)

    explicit_missing = resolver.resolve_behavioral_artifact_runtime(
        **common,
        explicit_artifact_path=str(tmp_path / "missing.parquet"),
    )
    assert not explicit_missing.enabled
    assert any("no prediction artifact was resolved" in line for line in explicit_missing.diagnostics)


def test_timeline_eligibility_follows_effective_walkforward_scope() -> None:
    import main
    from src.strategy_registry import (
        DEFAULT_PHASEAWARE_POLICY_ID,
        get_default_policy_registry,
        get_default_strategy_registry,
        resolve_phaseaware_configuration,
    )

    strategies = get_default_strategy_registry()
    policies = get_default_policy_registry()
    composition = resolve_phaseaware_configuration(
        DEFAULT_PHASEAWARE_POLICY_ID,
        strategy_registry=strategies,
        policy_registry=policies,
    )

    def eligibility(ids: list[str] | None) -> bool:
        scope = main.resolve_evaluation_scope(
            requested_strategy_ids=ids,
            registry=strategies,
            policy_registry=policies,
            surface_id="trend_vol",
        )
        return main._build_walkforward_execution_plan(
            scope,
            strategy_registry=strategies,
            phaseaware_configuration=composition,
        )["run_dynamic_selector"]

    assert eligibility(None) is True
    assert eligibility([composition.tf_strategy_id, composition.mr_strategy_id]) is True
    assert eligibility([composition.tf_strategy_id]) is False
    assert eligibility([composition.mr_strategy_id]) is False
    assert eligibility(["MR3"]) is False


def test_seed_and_market_data_source_environment_rules() -> None:
    from src.data import resolve_market_data_source
    from src.repro import resolve_experiment_seed

    original_seed = os.environ.get("EXPERIMENT_SEED")
    original_source = os.environ.get("MPML_DATA_SOURCE")
    try:
        for raw, expected in ((None, 42), ("", 42), ("42", 42)):
            if raw is None:
                os.environ.pop("EXPERIMENT_SEED", None)
            else:
                os.environ["EXPERIMENT_SEED"] = raw
            assert resolve_experiment_seed() == expected
        os.environ["EXPERIMENT_SEED"] = "abc"
        with pytest.raises(ValueError, match="Invalid EXPERIMENT_SEED"):
            resolve_experiment_seed()
        os.environ["EXPERIMENT_SEED"] = "abc"
        assert resolve_experiment_seed(cli_seed=9) == 9

        for raw, expected in (
            (None, "yfinance"),
            ("broker_csv", "broker_csv"),
            (" YFinance ", "yfinance"),
        ):
            if raw is None:
                os.environ.pop("MPML_DATA_SOURCE", None)
            else:
                os.environ["MPML_DATA_SOURCE"] = raw
            assert resolve_market_data_source() == expected
        os.environ["MPML_DATA_SOURCE"] = "bogus"
        with pytest.raises(ValueError, match="Unsupported market data source"):
            resolve_market_data_source()
    finally:
        for name, original in (
            ("EXPERIMENT_SEED", original_seed),
            ("MPML_DATA_SOURCE", original_source),
        ):
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original


def test_overlap_only_loose_boolean_is_resolved_at_call_time(tmp_path: Path) -> None:
    result = _isolated_python(
        tmp_path,
        """
        import contextlib
        import io
        import json
        import os
        from unittest.mock import patch
        os.environ["DL_SIGNALS_ENABLED"] = "false"
        import main

        class StopBeforeLoading(Exception):
            pass
        def stop_pipeline(**_kwargs):
            raise StopBeforeLoading
        original_build = main._build_experiment_metadata
        observed = {}
        values = [None, "", "true", "True", "1", "yes", "on", "garbage"]
        for value in values:
            if value is None:
                os.environ.pop("OVERLAP_ONLY", None)
            else:
                os.environ["OVERLAP_ONLY"] = value
            calls = []
            def capture(**kwargs):
                result = original_build(**kwargs)
                calls.append(kwargs.get("factor_overrides"))
                return result
            try:
                with patch.object(main, "_build_experiment_metadata", side_effect=capture):
                    with patch.object(main, "MarketDataPipeline", side_effect=stop_pipeline):
                        with contextlib.redirect_stdout(io.StringIO()):
                            main.main(experiment_seed=19)
            except StopBeforeLoading:
                pass
            observed[str(value)] = calls[-1]["overlap_only"]
        print(json.dumps(observed))
        """,
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    expected = {
        "None": False,
        "": False,
        "true": True,
        "True": True,
        "1": True,
        "yes": True,
        "on": True,
        "garbage": False,
    }
    assert json.loads(result.stdout) == expected


def test_clear_cache_removes_only_cache_pickle_files(tmp_path: Path) -> None:
    cache_dir = tmp_path / "data" / "cache"
    cache_dir.mkdir(parents=True)
    (cache_dir / "one.pkl").write_text("cache")
    (cache_dir / "keep.csv").write_text("not a pickle")
    broker_cache = tmp_path / "data" / "raw_broker_csv"
    broker_cache.mkdir(parents=True)
    (broker_cache / "one.pkl").write_text("broker data")
    result = _isolated_python(
        tmp_path,
        """
        import json
        from pathlib import Path
        from src.cache import clear_cache
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            clear_cache()
        print(json.dumps({
            "cache": sorted(path.name for path in Path("data/cache").iterdir()),
            "broker": (Path("data/raw_broker_csv/one.pkl").exists()),
        }))
        """,
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"cache": ["keep.csv"], "broker": True}


def test_recommendation_top_n_validation_is_late_and_accepts_positive_values() -> None:
    from src.recommendation import recommendations_from_evaluations

    assert recommendations_from_evaluations([], top_n=3) == []
    for top_n in (0, -1):
        with pytest.raises(ValueError, match="top_n must be a positive integer"):
            recommendations_from_evaluations([], top_n=top_n)


def test_direct_main_call_does_not_inherit_import_cache_or_config_side_effects(
    tmp_path: Path,
) -> None:
    (tmp_path / "data" / "cache").mkdir(parents=True)
    (tmp_path / "data" / "cache" / "keep.pkl").write_text("keep")
    result = _isolated_python(
        tmp_path,
        """
        import contextlib
        import io
        import json
        from pathlib import Path
        from unittest.mock import patch
        import_output = io.StringIO()
        with contextlib.redirect_stdout(import_output):
            import main
        class StopBeforeLoading(Exception):
            pass
        def stop_pipeline(**kwargs):
            raise StopBeforeLoading
        run_output = io.StringIO()
        try:
            with patch.object(main, "MarketDataPipeline", side_effect=stop_pipeline):
                with contextlib.redirect_stdout(run_output):
                    main.main(experiment_seed=19)
        except StopBeforeLoading:
            pass
        print(json.dumps({
            "import_output": import_output.getvalue(),
            "config_in_run_output": "[CONFIG]" in run_output.getvalue(),
            "cache_preserved": Path("data/cache/keep.pkl").exists(),
        }))
        """,
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "import_output": "",
        "config_in_run_output": False,
        "cache_preserved": True,
    }


def test_output_directory_collision_occurs_after_pipeline_construction(
    tmp_path: Path,
) -> None:
    existing = tmp_path / "already-there"
    existing.mkdir()
    result = _isolated_python(
        tmp_path,
        """
        import json
        from pathlib import Path
        from unittest.mock import patch
        import main

        calls = []
        def construct(**kwargs):
            calls.append(kwargs)
            return object()
        try:
            with patch.object(main, "MarketDataPipeline", side_effect=construct):
                main.main(output_dir=Path("already-there"), experiment_seed=19)
        except ValueError as exc:
            print(json.dumps({"error": str(exc), "pipeline_calls": len(calls)}))
        """,
        env_updates={"DL_SIGNALS_ENABLED": "false"},
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.splitlines()[-1])
    assert "Run output directory already exists" in outcome["error"]
    assert outcome["pipeline_calls"] == 1
