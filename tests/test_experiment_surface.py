"""
tests/test_experiment_surface.py
==================================
Regression tests for the v5 experiment_surface ontology migration.

These tests verify that:

1. experiment_surface is parsed and propagated correctly from manifests.
2. Sentiment attribution comes from sentiment_surface (parquet-level), NOT from variant.
3. evaluation_pair_family is tracked separately from training_pair_family.
4. surface_semantics_version is preserved.
5. Legacy manifests (no experiment_surface) fall back gracefully.
6. Anti-corruption assertions fire correctly.
7. Reproducibility checks scope to (seed, feature_surface) groups.
8. Sentiment / imputation-awareness comparisons use surface factors.

Run with:
    python -m unittest tests/test_experiment_surface.py -v
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from experiment_semantics import (
    EXPERIMENT_SURFACE_VERSION,
    normalize_experiment_surface,
    is_v5_surface,
)
from analysis.parsers.manifest_parser import parse_manifest, _parse_experiment_surface
from analysis.parsers.run_identity import infer_run_identity, _build_surface_run_meaning
from analysis.comparisons.factors import (
    summary_surface,
    summary_surface_source,
    is_v5_summary,
    filter_summaries,
    factor_crosstab,
)
from analysis.comparisons.sentiment import compare_sentiment_variants
from analysis.comparisons.factor_comparison import (
    compare_imputation_awareness_effect,
    compare_training_family_effect,
)
from analysis.validation import validate_summaries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(run_dir: Path, data: dict) -> None:
    (run_dir / "run_manifest.json").write_text(json.dumps(data))


def _make_run_dir() -> Path:
    return Path(tempfile.mkdtemp())


def _v5_surface(**kwargs) -> dict:
    """Build a minimal valid v5 experiment_surface block."""
    base = {
        "surface_semantics_version": EXPERIMENT_SURFACE_VERSION,
        "sentiment_surface": True,
        "training_pair_family": "persistent",
        "evaluation_pair_family": "persistent",
        "feature_surface": "trend_vol_only",
        "artifact_source": "artifacts_v5/persistent_dl_sentiment/model.pth",
    }
    base.update(kwargs)
    return base


def _make_summary_with_surface(
    run_id: str,
    surface: dict | None = None,
    missing_indicators_enabled: bool = False,
    dl_enabled: bool = True,
    msml_regime: str = "LVTF",
    sharpe_delta: float | None = None,
    pair: str = "EURUSD",
) -> dict:
    """Build a minimal v5 summary dict with experiment_surface."""
    surface = surface if surface is not None else _v5_surface()
    wf_row: dict = {"Pair": pair}
    if sharpe_delta is not None:
        wf_row["Sharpe_Delta"] = sharpe_delta
    return {
        "run_id": run_id,
        "meta": {
            "dl_enabled": dl_enabled,
            "experiment_gen": "gen1",
            "run_variant": "A",
            "sentiment_enabled": True,
            "missing_indicators_enabled": missing_indicators_enabled,
            "experiment_surface": surface,
            "surface_source": "manifest",
            "experiment": {
                "generation": "gen1",
                "variant": "A",
                "factors": {
                    "dl_enabled": dl_enabled,
                    "sentiment_enabled": True,
                    "missing_indicators_enabled": missing_indicators_enabled,
                    "msml_regime": msml_regime,
                    "overlap_only": False,
                    "selector_enabled": True,
                },
                "semantic_label": "Gen1_A",
            },
        },
        "csvs": {
            "walkforward_summary": [wf_row],
            "selector_comparison": None,
            "ml_accuracy": None,
            "backtest": None,
            "walkforward_per_fold": None,
            "ablation_aggregate": None,
            "ablation_per_pair": None,
            "vol_guard_summary": None,
            "vol_guard_per_fold": None,
            "results_summary": None,
            "results_per_pair": None,
        },
        "log": {},
        "coverage": {},
        "warnings": [],
    }


# ---------------------------------------------------------------------------
# normalize_experiment_surface tests
# ---------------------------------------------------------------------------


class TestNormalizeExperimentSurface(unittest.TestCase):

    def test_basic_v5_surface(self):
        raw = _v5_surface()
        result = normalize_experiment_surface(raw)
        self.assertEqual(result["sentiment_surface"], True)
        self.assertEqual(result["training_pair_family"], "persistent")
        self.assertEqual(result["evaluation_pair_family"], "persistent")
        self.assertEqual(result["feature_surface"], "trend_vol_only")
        self.assertEqual(result["surface_semantics_version"], EXPERIMENT_SURFACE_VERSION)

    def test_nosentiment_surface(self):
        raw = _v5_surface(sentiment_surface=False)
        result = normalize_experiment_surface(raw)
        self.assertIs(result["sentiment_surface"], False)

    def test_none_input_returns_all_none(self):
        result = normalize_experiment_surface(None)
        for key, val in result.items():
            self.assertIsNone(val, f"Expected None for key={key!r}, got {val!r}")

    def test_string_version_coerced_to_int(self):
        raw = _v5_surface(surface_semantics_version="5")
        result = normalize_experiment_surface(raw)
        self.assertEqual(result["surface_semantics_version"], 5)

    def test_evaluation_pair_family_distinct_from_training(self):
        raw = _v5_surface(
            training_pair_family="persistent",
            evaluation_pair_family="reactive",
        )
        result = normalize_experiment_surface(raw)
        self.assertEqual(result["training_pair_family"], "persistent")
        self.assertEqual(result["evaluation_pair_family"], "reactive")

    def test_is_v5_surface_true(self):
        surface = normalize_experiment_surface(_v5_surface())
        self.assertTrue(is_v5_surface(surface))

    def test_is_v5_surface_false_when_missing_version(self):
        surface = normalize_experiment_surface(
            {k: v for k, v in _v5_surface().items() if k != "surface_semantics_version"}
        )
        self.assertFalse(is_v5_surface(surface))

    def test_is_v5_surface_false_when_missing_sentiment(self):
        surface = normalize_experiment_surface(
            {k: v for k, v in _v5_surface().items() if k != "sentiment_surface"}
        )
        self.assertFalse(is_v5_surface(surface))


# ---------------------------------------------------------------------------
# manifest_parser tests
# ---------------------------------------------------------------------------


class TestManifestParserExperimentSurface(unittest.TestCase):

    def _parse(self, data: dict) -> dict:
        run_dir = _make_run_dir()
        _write_manifest(run_dir, data)
        result = parse_manifest(run_dir)
        self.assertIsNotNone(result)
        return result

    def test_v5_surface_parsed(self):
        data = {
            "run": {"run_id": "gen1_A__20240101T120000Z", "timestamp_utc": "20240101T120000Z"},
            "experiment": {
                "generation": "gen1", "variant": "A",
                "sentiment_enabled": True, "missing_indicators_enabled": False,
                "dl_enabled": True, "msml_regime": "LVTF",
                "factors": {
                    "dl_enabled": True, "sentiment_enabled": True,
                    "missing_indicators_enabled": False, "msml_regime": "LVTF",
                    "overlap_only": False, "selector_enabled": True,
                },
                "semantic_label": "Gen1_A",
            },
            "experiment_surface": _v5_surface(sentiment_surface=False),
        }
        result = self._parse(data)
        surface = result["experiment_surface"]
        self.assertIs(surface["sentiment_surface"], False,
                      "persistent_dl_nosentiment_blind: sentiment_surface must be False "
                      "(not inferred from variant=A)")
        self.assertEqual(surface["training_pair_family"], "persistent")
        self.assertEqual(surface["surface_semantics_version"], EXPERIMENT_SURFACE_VERSION)
        self.assertEqual(result["surface_source"], "manifest")

    def test_surface_source_legacy_when_no_surface(self):
        data = {
            "run": {"run_id": "gen1_A__20240101T120000Z", "timestamp_utc": "20240101T120000Z"},
            "experiment": {
                "generation": "gen1", "variant": "A",
                "sentiment_enabled": True, "missing_indicators_enabled": False,
                "dl_enabled": True,
                "factors": {
                    "dl_enabled": True, "sentiment_enabled": True,
                    "missing_indicators_enabled": False, "msml_regime": "LVTF",
                    "overlap_only": False, "selector_enabled": True,
                },
                "semantic_label": "Gen1_A",
            },
        }
        result = self._parse(data)
        self.assertEqual(result["surface_source"], "legacy_variant_fallback")

    def test_evaluation_pair_family_in_parsed_surface(self):
        data = {
            "run": {"run_id": "gen1_A__20240101T120000Z", "timestamp_utc": "20240101T120000Z"},
            "experiment": {"generation": "gen1", "variant": "A", "semantic_label": "Gen1_A",
                           "factors": {"dl_enabled": True, "sentiment_enabled": True,
                                       "missing_indicators_enabled": False, "msml_regime": "LVTF",
                                       "overlap_only": False, "selector_enabled": True}},
            "experiment_surface": _v5_surface(
                training_pair_family="persistent",
                evaluation_pair_family="reactive",
            ),
        }
        result = self._parse(data)
        surface = result["experiment_surface"]
        self.assertEqual(surface["training_pair_family"], "persistent")
        self.assertEqual(surface["evaluation_pair_family"], "reactive")

    def test_surface_semantics_version_preserved(self):
        data = {
            "run": {"run_id": "gen1_A__20240101T120000Z", "timestamp_utc": "20240101T120000Z"},
            "experiment": {"generation": "gen1", "variant": "A", "semantic_label": "Gen1_A",
                           "factors": {"dl_enabled": True, "sentiment_enabled": True,
                                       "missing_indicators_enabled": False, "msml_regime": "LVTF",
                                       "overlap_only": False, "selector_enabled": True}},
            "experiment_surface": _v5_surface(surface_semantics_version=5),
        }
        result = self._parse(data)
        self.assertEqual(result["experiment_surface"]["surface_semantics_version"], 5)


# ---------------------------------------------------------------------------
# Anti-corruption regression tests
# ---------------------------------------------------------------------------


class TestAntiCorruptionSentimentSurface(unittest.TestCase):
    """
    The root corruption being fixed:
    persistent_dl_nosentiment_blind with variant=A was reported as sentiment_enabled=True.
    The fix: sentiment_surface is read from experiment_surface, NOT from variant.
    """

    def test_persistent_dl_nosentiment_blind_has_sentiment_surface_false(self):
        """CRITICAL: persistent_dl_nosentiment_blind → sentiment_surface=False."""
        surface = normalize_experiment_surface(
            _v5_surface(sentiment_surface=False, training_pair_family="persistent")
        )
        self.assertIs(surface["sentiment_surface"], False,
                      "persistent_dl_nosentiment_blind must have sentiment_surface=False")

    def test_persistent_dl_sentiment_blind_has_sentiment_surface_true(self):
        """persistent_dl_sentiment_blind → sentiment_surface=True."""
        surface = normalize_experiment_surface(
            _v5_surface(sentiment_surface=True, training_pair_family="persistent")
        )
        self.assertIs(surface["sentiment_surface"], True)
        self.assertEqual(surface["training_pair_family"], "persistent")

    def test_reactive_dl_sentiment_aware_correct_family_and_awareness(self):
        """reactive_dl_sentiment_aware → training_pair_family=reactive, imputation awareness=True."""
        summary = _make_summary_with_surface(
            "reactive_dl_sentiment_aware",
            surface=_v5_surface(
                sentiment_surface=True,
                training_pair_family="reactive",
                evaluation_pair_family="reactive",
            ),
            missing_indicators_enabled=True,
        )
        surface = summary_surface(summary)
        self.assertEqual(surface["training_pair_family"], "reactive")
        # Imputation awareness is stored in runtime factors, not surface.
        meta = summary.get("meta") or {}
        factors = (meta.get("experiment") or {}).get("factors") or {}
        self.assertIs(factors["missing_indicators_enabled"], True)
        self.assertIs(surface["sentiment_surface"], True)

    def test_variant_A_does_not_imply_sentiment_surface_true(self):
        """
        Anti-corruption: variant=A must NOT imply sentiment_surface=True when
        experiment_surface is absent.  The run should be flagged as legacy_variant_fallback.
        """
        run_dir = _make_run_dir()
        _write_manifest(run_dir, {
            "run": {"run_id": "gen1_A__20240101T120000Z", "timestamp_utc": "20240101T120000Z"},
            "experiment": {
                "generation": "gen1", "variant": "A",
                "sentiment_enabled": True, "missing_indicators_enabled": False,
                "dl_enabled": True,
                "factors": {
                    "dl_enabled": True, "sentiment_enabled": True,
                    "missing_indicators_enabled": False, "msml_regime": "LVTF",
                    "overlap_only": False, "selector_enabled": True,
                },
                "semantic_label": "Gen1_A",
            },
            # No experiment_surface block.
        })
        result = parse_manifest(run_dir)
        # Surface block must not infer sentiment_surface from variant A.
        self.assertEqual(result["surface_source"], "legacy_variant_fallback")
        surface = result["experiment_surface"]
        self.assertIsNone(surface["sentiment_surface"],
                          "Without experiment_surface, sentiment_surface must be None "
                          "(not inferred from variant=A)")


# ---------------------------------------------------------------------------
# run_identity tests
# ---------------------------------------------------------------------------


class TestRunIdentityWithSurface(unittest.TestCase):

    def _build_identity(self, surface: dict | None = None) -> dict:
        run_dir = _make_run_dir()
        archive_root = run_dir.parent
        manifest = {
            "run_id": "gen1_A__20240101T120000Z",
            "timestamp_utc": "20240101T120000Z",
            "experiment": {
                "generation": "gen1", "variant": "A",
                "sentiment_enabled": True, "missing_indicators_enabled": False,
                "dl_enabled": True, "msml_regime": "LVTF",
                "factors": {
                    "dl_enabled": True, "sentiment_enabled": True,
                    "missing_indicators_enabled": False, "msml_regime": "LVTF",
                    "overlap_only": False, "selector_enabled": True,
                },
                "semantic_label": "Gen1_A",
                "legacy_semantics": False,
            },
            "experiment_surface": surface or _v5_surface(sentiment_surface=False),
            "surface_source": "manifest" if surface is not None or True else "legacy_variant_fallback",
        }
        return infer_run_identity(
            archive_root=archive_root,
            run_dir=run_dir,
            experiment_gen="gen1",
            manifest=manifest,
        )

    def test_v5_run_meaning_is_surface_derived(self):
        identity = self._build_identity(
            _v5_surface(
                sentiment_surface=False,
                training_pair_family="persistent",
                evaluation_pair_family="persistent",
            )
        )
        meaning = identity["run_meaning"]
        self.assertIn("persistent", meaning.lower())
        self.assertIn("no-sentiment", meaning.lower())

    def test_surface_propagated_through_identity(self):
        identity = self._build_identity()
        surface = identity["experiment_surface"]
        self.assertIsNotNone(surface)
        self.assertEqual(identity["surface_source"], "manifest")

    def test_legacy_warning_emitted_without_surface(self):
        run_dir = _make_run_dir()
        archive_root = run_dir.parent
        manifest = {
            "run_id": "gen1_A__20240101T120000Z",
            "timestamp_utc": "20240101T120000Z",
            "experiment": {
                "generation": "gen1", "variant": "A",
                "sentiment_enabled": True, "missing_indicators_enabled": False,
                "dl_enabled": True, "msml_regime": "LVTF",
                "factors": {
                    "dl_enabled": True, "sentiment_enabled": True,
                    "missing_indicators_enabled": False, "msml_regime": "LVTF",
                    "overlap_only": False, "selector_enabled": True,
                },
                "semantic_label": "Gen1_A",
            },
            "surface_source": "legacy_variant_fallback",
        }
        identity = infer_run_identity(
            archive_root=archive_root,
            run_dir=run_dir,
            experiment_gen="gen1",
            manifest=manifest,
        )
        # Should warn about missing experiment_surface.
        warnings = identity["identity_warnings"]
        self.assertTrue(
            any("experiment_surface" in w.lower() for w in warnings),
            f"Expected experiment_surface warning, got: {warnings}"
        )

    def test_surface_run_meaning_with_transfer_learning(self):
        meaning = _build_surface_run_meaning({
            "sentiment_surface": True,
            "training_pair_family": "persistent",
            "evaluation_pair_family": "reactive",
            "feature_surface": "trend_vol_only",
        })
        self.assertIn("persistent", meaning.lower())
        self.assertIn("reactive", meaning.lower())
        self.assertIn("sentiment surface", meaning.lower())


# ---------------------------------------------------------------------------
# Sentiment comparison with v5 surface
# ---------------------------------------------------------------------------


class TestSentimentComparisonV5Surface(unittest.TestCase):

    def test_nosentiment_run_NOT_in_sentiment_on_cohort(self):
        """
        CRITICAL anti-corruption: persistent_dl_nosentiment_blind must NOT appear
        in the sentiment_on cohort even if variant=A (which has sentiment_enabled=True).
        """
        summaries = [
            _make_summary_with_surface(
                "persistent_sentiment",
                surface=_v5_surface(sentiment_surface=True, training_pair_family="persistent"),
                sharpe_delta=0.15,
            ),
            _make_summary_with_surface(
                "persistent_nosentiment",
                surface=_v5_surface(sentiment_surface=False, training_pair_family="persistent"),
                sharpe_delta=0.05,
            ),
        ]
        result = compare_sentiment_variants(summaries)
        self.assertIn("persistent_sentiment", result["sentiment_on"])
        self.assertNotIn("persistent_nosentiment", result["sentiment_on"])
        self.assertIn("persistent_nosentiment", result["sentiment_off"])

    def test_v5_sentiment_comparison_produces_valid_delta(self):
        summaries = [
            _make_summary_with_surface(
                "run_sentiment_on",
                surface=_v5_surface(sentiment_surface=True, training_pair_family="persistent"),
                sharpe_delta=0.20,
            ),
            _make_summary_with_surface(
                "run_sentiment_off",
                surface=_v5_surface(sentiment_surface=False, training_pair_family="persistent"),
                sharpe_delta=0.10,
            ),
        ]
        result = compare_sentiment_variants(summaries)
        self.assertGreater(len(result["valid_comparisons"]), 0)
        self.assertGreater(len(result["delta_table"]), 0)
        row = next(
            (r for r in result["delta_table"] if r.get("metric") == "sharpe_delta"),
            None,
        )
        self.assertIsNotNone(row)
        self.assertAlmostEqual(row["delta_on_minus_off"], 0.10, places=5)

    def test_mixed_training_families_not_compared(self):
        """
        Persistent and reactive runs with different sentiment must NOT be
        compared against each other (they differ in training_pair_family).
        """
        summaries = [
            _make_summary_with_surface(
                "persistent_sentiment_on",
                surface=_v5_surface(sentiment_surface=True, training_pair_family="persistent"),
                sharpe_delta=0.20,
            ),
            _make_summary_with_surface(
                "reactive_sentiment_off",
                surface=_v5_surface(sentiment_surface=False, training_pair_family="reactive"),
                sharpe_delta=0.05,
            ),
        ]
        result = compare_sentiment_variants(summaries)
        # They are in different families → no valid cross-family comparison.
        self.assertEqual(result["delta_table"], [])
        self.assertEqual(result["valid_comparisons"], [])


# ---------------------------------------------------------------------------
# Imputation Awareness effect (v5)
# ---------------------------------------------------------------------------


class TestImputationAwarenessEffect(unittest.TestCase):

    def test_imputation_awareness_effect_valid(self):
        summaries = [
            _make_summary_with_surface(
                "aware_run",
                surface=_v5_surface(sentiment_surface=True, training_pair_family="persistent"),
                missing_indicators_enabled=True,
                sharpe_delta=0.12,
            ),
            _make_summary_with_surface(
                "blind_run",
                surface=_v5_surface(sentiment_surface=True, training_pair_family="persistent"),
                missing_indicators_enabled=False,
                sharpe_delta=0.08,
            ),
        ]
        result = compare_imputation_awareness_effect(summaries)
        valid_comps = [c for c in result["comparisons"] if c["valid"]]
        self.assertGreater(len(valid_comps), 0)
        comp = valid_comps[0]
        self.assertIn("aware_run", comp["imputation_aware"])
        self.assertIn("blind_run", comp["imputation_blind"])

    def test_same_feature_surface_different_awareness_not_reproducibility_error(self):
        """
        Two runs with same seed but different imputation awareness but same feature_surface
        should be in the same reproducibility group.
        """
        summaries = [
            _make_summary_with_surface(
                "aware_run",
                surface=_v5_surface(feature_surface="trend_vol_only"),
                missing_indicators_enabled=True,
            ),
            _make_summary_with_surface(
                "blind_run",
                surface=_v5_surface(feature_surface="trend_vol_only"),
                missing_indicators_enabled=False,
            ),
        ]
        # Add same seed to both.
        for s in summaries:
            s["meta"]["reproducibility"] = {"experiment_seed": 42, "numpy_seed": 42,
                                             "python_random_seed": 42, "torch_seed": 42}
        result = validate_summaries(summaries)
        repro_warnings = [
            w for w in result["warnings"] if "feature column order" in w.lower() or
            "differing reproducibility" in w.lower()
        ]
        # Same seed + same feature_surface → only checked within that group, not globally.
        # With no actual feature_ordering metadata, there should be no feature-column warnings.
        self.assertEqual(repro_warnings, [])


# ---------------------------------------------------------------------------
# Training-family effect (v5)
# ---------------------------------------------------------------------------


class TestTrainingFamilyEffect(unittest.TestCase):

    def test_training_family_effect_valid(self):
        summaries = [
            _make_summary_with_surface(
                "persistent_run",
                surface=_v5_surface(sentiment_surface=True, training_pair_family="persistent"),
                sharpe_delta=0.15,
            ),
            _make_summary_with_surface(
                "reactive_run",
                surface=_v5_surface(sentiment_surface=True, training_pair_family="reactive"),
                sharpe_delta=0.10,
            ),
        ]
        result = compare_training_family_effect(summaries)
        valid_comps = [c for c in result["comparisons"] if c["valid"]]
        self.assertGreater(len(valid_comps), 0)

    def test_training_family_effect_incomplete_when_only_one_family(self):
        summaries = [
            _make_summary_with_surface(
                "persistent_run",
                surface=_v5_surface(sentiment_surface=True, training_pair_family="persistent"),
            ),
        ]
        result = compare_training_family_effect(summaries)
        # Only one family → no pairs to compare.
        valid_comps = [c for c in result["comparisons"] if c["valid"]]
        self.assertEqual(valid_comps, [])


# ---------------------------------------------------------------------------
# factor_crosstab surface keys
# ---------------------------------------------------------------------------


class TestFactorCrosstabSurfaceKeys(unittest.TestCase):

    def test_crosstab_includes_sentiment_surface(self):
        summaries = [
            _make_summary_with_surface(
                "run_on",
                surface=_v5_surface(sentiment_surface=True),
            ),
            _make_summary_with_surface(
                "run_off",
                surface=_v5_surface(sentiment_surface=False),
            ),
        ]
        crosstab = factor_crosstab(summaries)
        self.assertIn("sentiment_surface", crosstab)
        self.assertIn("True", crosstab["sentiment_surface"])
        self.assertIn("False", crosstab["sentiment_surface"])

    def test_crosstab_includes_training_pair_family(self):
        summaries = [
            _make_summary_with_surface(
                "run_persistent",
                surface=_v5_surface(training_pair_family="persistent"),
            ),
            _make_summary_with_surface(
                "run_reactive",
                surface=_v5_surface(training_pair_family="reactive"),
            ),
        ]
        crosstab = factor_crosstab(summaries)
        self.assertIn("training_pair_family", crosstab)
        self.assertIn("persistent", crosstab["training_pair_family"])
        self.assertIn("reactive", crosstab["training_pair_family"])

    def test_crosstab_includes_evaluation_pair_family(self):
        summaries = [
            _make_summary_with_surface(
                "run_transfer",
                surface=_v5_surface(
                    training_pair_family="persistent",
                    evaluation_pair_family="reactive",
                ),
            ),
        ]
        crosstab = factor_crosstab(summaries)
        self.assertIn("evaluation_pair_family", crosstab)
        self.assertIn("reactive", crosstab["evaluation_pair_family"])


# ---------------------------------------------------------------------------
# Reproducibility: feature-surface scoped
# ---------------------------------------------------------------------------


class TestReproducibilityFeatureSurface(unittest.TestCase):

    def _make_seeded(self, run_id: str, feature_surface: str, seed: int = 42) -> dict:
        s = _make_summary_with_surface(
            run_id,
            surface=_v5_surface(feature_surface=feature_surface),
        )
        s["meta"]["manifest_present"] = True
        s["meta"]["reproducibility"] = {
            "experiment_seed": seed,
            "numpy_seed": seed,
            "python_random_seed": seed,
            "torch_seed": seed,
        }
        return s

    def test_different_feature_surfaces_same_seed_no_repro_warning(self):
        """
        Runs sharing a seed but with different feature_surface values are in
        DIFFERENT reproducibility groups and must NOT produce cross-surface warnings.
        """
        summaries = [
            self._make_seeded("run_trend", "trend_vol_only", seed=42),
            self._make_seeded("run_price", "price_trend", seed=42),
        ]
        result = validate_summaries(summaries)
        repro_warnings = [
            w for w in result["warnings"]
            if "differing reproducibility" in w.lower() or "feature column order" in w.lower()
        ]
        self.assertEqual(repro_warnings, [])

    def test_same_feature_surface_same_seed_checked_together(self):
        """
        Runs sharing seed AND feature_surface are in the same group and
        get their reproducibility metadata checked together.
        """
        s1 = self._make_seeded("run_a", "trend_vol_only", seed=42)
        s2 = self._make_seeded("run_b", "trend_vol_only", seed=42)
        # Mutate seed to force mismatch.
        s2["meta"]["reproducibility"]["numpy_seed"] = 99
        result = validate_summaries([s1, s2])
        repro_warnings = [
            w for w in result["warnings"]
            if "differing reproducibility" in w.lower()
        ]
        self.assertGreater(len(repro_warnings), 0)


# ---------------------------------------------------------------------------
# Validation anti-corruption
# ---------------------------------------------------------------------------


class TestValidationAntiCorruption(unittest.TestCase):

    def test_non_legacy_manifest_without_surface_emits_semantic_warning(self):
        """A manifest without experiment_surface should emit a semantic warning."""
        run_dir = _make_run_dir()
        _write_manifest(run_dir, {
            "run": {"run_id": "gen1_A__20240101T120000Z", "timestamp_utc": "20240101T120000Z"},
            "experiment": {
                "generation": "gen1", "variant": "A",
                "sentiment_enabled": True, "missing_indicators_enabled": False,
                "dl_enabled": True,
                "factors": {"dl_enabled": True, "sentiment_enabled": True,
                            "missing_indicators_enabled": False, "msml_regime": "LVTF",
                            "overlap_only": False, "selector_enabled": True},
                "semantic_label": "Gen1_A",
                "legacy_semantics": False,
            },
            # No experiment_surface.
        })
        manifest = parse_manifest(run_dir)
        identity = infer_run_identity(
            archive_root=run_dir.parent,
            run_dir=run_dir,
            experiment_gen="gen1",
            manifest=manifest,
        )
        summary = {
            "run_id": identity["run_id"],
            "meta": {
                **identity,
                "manifest_present": True,
                "legacy_mode": False,
                "legacy_semantics": identity["legacy_semantics"],
                "manifest_diagnostics": {"manifest_count": 1},
                "experiment": manifest["experiment"],
                "experiment_surface": identity["experiment_surface"],
                "surface_source": identity["surface_source"],
            },
            "csvs": {},
            "log": {},
            "warnings": [],
        }
        result = validate_summaries([summary])
        semantic_warnings = result["sections"]["semantic_integrity"]["warnings"]
        self.assertTrue(
            any("experiment_surface" in w.lower() for w in semantic_warnings),
            f"Expected experiment_surface warning, got: {semantic_warnings}"
        )

    def test_is_v5_summary_false_for_legacy(self):
        from tests.test_comparisons import _make_summary
        legacy = _make_summary("legacy_run", dl_enabled=True, experiment_gen="gen1")
        self.assertFalse(is_v5_summary(legacy))

    def test_is_v5_summary_true_for_surface_run(self):
        v5 = _make_summary_with_surface("v5_run")
        self.assertTrue(is_v5_summary(v5))


if __name__ == "__main__":
    unittest.main()
