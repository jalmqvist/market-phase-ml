"""
tests/test_behavioral_surface.py
==================================
Tests for the mpml.behavioral package (Phase A — Behavioral Surface Registry).

Covers:
- BehavioralState construction and immutability
- TrendVolSurface states, get_state(), alias resolution, metadata
- ReactiveJPYSurface states, get_state(), metadata
- BehavioralSurfaceRegistry register/load/available/get_state
- Compatibility helpers (dl_regime_to_state, phase_label_to_state,
  build_behavioral_surface_manifest_block)
- Default registry contents

Run with:
    python -m pytest tests/test_behavioral_surface.py -v
"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from mpml.behavioral.base import BehavioralState, BehavioralSurface
from mpml.behavioral.registry import BehavioralSurfaceRegistry, default_registry
from mpml.behavioral.trend_vol import TrendVolSurface
from mpml.behavioral.reactive_jpy import ReactiveJPYSurface
from mpml.behavioral.compat import (
    dl_regime_to_state,
    phase_label_to_state,
    build_behavioral_surface_manifest_block,
)
from mpml.behavioral import registry as global_registry


# ---------------------------------------------------------------------------
# BehavioralState tests
# ---------------------------------------------------------------------------

class TestBehavioralState(unittest.TestCase):

    def test_basic_construction(self):
        state = BehavioralState(
            state_id="TEST",
            display_name="Test State",
            surface_id="test_surface",
        )
        self.assertEqual(state.state_id, "TEST")
        self.assertEqual(state.display_name, "Test State")
        self.assertEqual(state.surface_id, "test_surface")
        self.assertEqual(state.description, "")
        self.assertEqual(state.metadata, {})

    def test_str_returns_state_id(self):
        state = BehavioralState(
            state_id="LVTF",
            display_name="Low-Vol Trend",
            surface_id="trend_vol",
        )
        self.assertEqual(str(state), "LVTF")

    def test_frozen_immutability(self):
        state = BehavioralState(
            state_id="LVTF",
            display_name="LV Trend",
            surface_id="trend_vol",
        )
        with self.assertRaises((AttributeError, TypeError)):
            state.state_id = "CHANGED"  # type: ignore[misc]

    def test_empty_state_id_raises(self):
        with self.assertRaises(ValueError):
            BehavioralState(state_id="", display_name="x", surface_id="s")

    def test_empty_surface_id_raises(self):
        with self.assertRaises(ValueError):
            BehavioralState(state_id="X", display_name="x", surface_id="")

    def test_metadata_field(self):
        state = BehavioralState(
            state_id="X",
            display_name="X",
            surface_id="s",
            metadata={"aliases": ["Y"]},
        )
        self.assertEqual(state.metadata["aliases"], ["Y"])


# ---------------------------------------------------------------------------
# TrendVolSurface tests
# ---------------------------------------------------------------------------

class TestTrendVolSurface(unittest.TestCase):

    def setUp(self):
        self.surface = TrendVolSurface()

    def test_surface_id(self):
        self.assertEqual(self.surface.surface_id, "trend_vol")

    def test_surface_version(self):
        self.assertIsNotNone(self.surface.surface_version)

    def test_display_name(self):
        self.assertIn("Trend", self.surface.display_name)

    def test_states_returns_four(self):
        states = self.surface.states()
        self.assertEqual(len(states), 4)

    def test_state_ids_are_canonical(self):
        ids = self.surface.state_ids()
        for expected in ("LVTF", "HVTF", "LVR", "HVR"):
            self.assertIn(expected, ids)

    def test_get_state_canonical(self):
        for state_id in ("LVTF", "HVTF", "LVR", "HVR"):
            state = self.surface.get_state(state_id)
            self.assertEqual(state.state_id, state_id)
            self.assertEqual(state.surface_id, "trend_vol")

    def test_get_state_alias_hvmr(self):
        state = self.surface.get_state("HVMR")
        self.assertEqual(state.state_id, "HVR")

    def test_get_state_alias_lvmr(self):
        state = self.surface.get_state("LVMR")
        self.assertEqual(state.state_id, "LVR")

    def test_get_state_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.surface.get_state("UNKNOWN_STATE")

    def test_metadata_shape(self):
        meta = self.surface.metadata()
        for key in ("surface_id", "surface_version", "display_name", "description",
                    "state_ids", "aliases"):
            self.assertIn(key, meta)

    def test_metadata_aliases_present(self):
        meta = self.surface.metadata()
        aliases = meta["aliases"]
        self.assertIn("HVMR", aliases)
        self.assertIn("LVMR", aliases)

    def test_repr(self):
        r = repr(self.surface)
        self.assertIn("TrendVolSurface", r)
        self.assertIn("trend_vol", r)


# ---------------------------------------------------------------------------
# ReactiveJPYSurface tests
# ---------------------------------------------------------------------------

class TestReactiveJPYSurface(unittest.TestCase):

    def setUp(self):
        self.surface = ReactiveJPYSurface()

    def test_surface_id(self):
        self.assertEqual(self.surface.surface_id, "reactive_jpy")

    def test_states_returns_four(self):
        self.assertEqual(len(self.surface.states()), 4)

    def test_state_ids(self):
        ids = self.surface.state_ids()
        expected = {
            "JPY_NON_EXTREME",
            "JPY_CONSENSUS_YOUNG",
            "JPY_CONSENSUS_MATURING",
            "JPY_CONSENSUS_MATURE",
        }
        self.assertEqual(set(ids), expected)

    def test_get_state_all_valid(self):
        for state_id in (
            "JPY_NON_EXTREME",
            "JPY_CONSENSUS_YOUNG",
            "JPY_CONSENSUS_MATURING",
            "JPY_CONSENSUS_MATURE",
        ):
            state = self.surface.get_state(state_id)
            self.assertEqual(state.state_id, state_id)
            self.assertEqual(state.surface_id, "reactive_jpy")

    def test_get_state_unknown_raises(self):
        with self.assertRaises(KeyError):
            self.surface.get_state("LVTF")  # belongs to a different surface

    def test_metadata_shape(self):
        meta = self.surface.metadata()
        for key in ("surface_id", "surface_version", "display_name", "description"):
            self.assertIn(key, meta)

    def test_all_states_have_surface_id(self):
        for state in self.surface.states():
            self.assertEqual(state.surface_id, "reactive_jpy")


# ---------------------------------------------------------------------------
# BehavioralSurfaceRegistry tests
# ---------------------------------------------------------------------------

class TestBehavioralSurfaceRegistry(unittest.TestCase):

    def _fresh_registry(self):
        reg = BehavioralSurfaceRegistry()
        reg.register(TrendVolSurface())
        reg.register(ReactiveJPYSurface())
        return reg

    def test_available_surfaces(self):
        reg = self._fresh_registry()
        available = reg.available()
        self.assertIn("trend_vol", available)
        self.assertIn("reactive_jpy", available)

    def test_load_trend_vol(self):
        reg = self._fresh_registry()
        surface = reg.load("trend_vol")
        self.assertIsInstance(surface, TrendVolSurface)

    def test_load_reactive_jpy(self):
        reg = self._fresh_registry()
        surface = reg.load("reactive_jpy")
        self.assertIsInstance(surface, ReactiveJPYSurface)

    def test_load_unknown_raises(self):
        reg = self._fresh_registry()
        with self.assertRaises(KeyError):
            reg.load("does_not_exist")

    def test_get_state_shortcut(self):
        reg = self._fresh_registry()
        state = reg.get_state("trend_vol", "LVTF")
        self.assertEqual(state.state_id, "LVTF")

    def test_get_state_alias_via_registry(self):
        reg = self._fresh_registry()
        state = reg.get_state("trend_vol", "HVMR")
        self.assertEqual(state.state_id, "HVR")

    def test_contains(self):
        reg = self._fresh_registry()
        self.assertIn("trend_vol", reg)
        self.assertNotIn("does_not_exist", reg)

    def test_register_non_surface_raises(self):
        reg = BehavioralSurfaceRegistry()
        with self.assertRaises(TypeError):
            reg.register("not_a_surface")  # type: ignore[arg-type]

    def test_register_replaces_existing(self):
        reg = BehavioralSurfaceRegistry()
        reg.register(TrendVolSurface())
        reg.register(TrendVolSurface())  # should not raise; silently replaces
        self.assertIn("trend_vol", reg)

    def test_all_metadata_keys(self):
        reg = self._fresh_registry()
        all_meta = reg.all_metadata()
        self.assertIn("trend_vol", all_meta)
        self.assertIn("reactive_jpy", all_meta)

    def test_repr(self):
        reg = self._fresh_registry()
        self.assertIn("BehavioralSurfaceRegistry", repr(reg))


# ---------------------------------------------------------------------------
# Default registry tests
# ---------------------------------------------------------------------------

class TestDefaultRegistry(unittest.TestCase):

    def test_default_registry_has_trend_vol(self):
        self.assertIn("trend_vol", default_registry)

    def test_default_registry_has_reactive_jpy(self):
        self.assertIn("reactive_jpy", default_registry)

    def test_global_registry_alias(self):
        # mpml.behavioral.registry (imported as global_registry) exposes
        # default_registry through the module attribute
        self.assertIs(global_registry, default_registry)


# ---------------------------------------------------------------------------
# Compatibility helper tests
# ---------------------------------------------------------------------------

class TestCompatHelpers(unittest.TestCase):

    def test_dl_regime_to_state_canonical(self):
        for regime, expected_id in (
            ("LVTF", "LVTF"),
            ("HVTF", "HVTF"),
            ("LVR", "LVR"),
            ("HVR", "HVR"),
        ):
            state = dl_regime_to_state(regime)
            self.assertEqual(state.state_id, expected_id)
            self.assertEqual(state.surface_id, "trend_vol")

    def test_dl_regime_to_state_alias_hvmr(self):
        state = dl_regime_to_state("HVMR")
        self.assertEqual(state.state_id, "HVR")

    def test_dl_regime_to_state_alias_lvmr(self):
        state = dl_regime_to_state("LVMR")
        self.assertEqual(state.state_id, "LVR")

    def test_dl_regime_to_state_unknown_raises(self):
        with self.assertRaises(KeyError):
            dl_regime_to_state("UNKNOWN")

    def test_phase_label_to_state(self):
        for label, expected_id in (
            ("HV_Trend",   "HVTF"),
            ("LV_Trend",   "LVTF"),
            ("HV_Ranging", "HVR"),
            ("LV_Ranging", "LVR"),
        ):
            state = phase_label_to_state(label)
            self.assertEqual(state.state_id, expected_id)

    def test_phase_label_to_state_unknown_raises(self):
        with self.assertRaises(KeyError):
            phase_label_to_state("Unknown_Label")

    def test_build_manifest_block_no_state(self):
        block = build_behavioral_surface_manifest_block("trend_vol")
        self.assertEqual(block["surface_id"], "trend_vol")
        self.assertIn("surface_version", block)
        self.assertIn("display_name", block)
        self.assertNotIn("behavioral_state", block)

    def test_build_manifest_block_with_state(self):
        block = build_behavioral_surface_manifest_block("trend_vol", "LVTF")
        self.assertEqual(block["surface_id"], "trend_vol")
        self.assertIn("behavioral_state", block)
        self.assertEqual(block["behavioral_state"]["state_id"], "LVTF")
        self.assertIn("display_name", block["behavioral_state"])
        self.assertIn("description", block["behavioral_state"])

    def test_build_manifest_block_with_alias(self):
        block = build_behavioral_surface_manifest_block("trend_vol", "HVMR")
        self.assertEqual(block["behavioral_state"]["state_id"], "HVR")

    def test_build_manifest_block_jpy_surface(self):
        block = build_behavioral_surface_manifest_block(
            "reactive_jpy", "JPY_CONSENSUS_MATURE"
        )
        self.assertEqual(block["surface_id"], "reactive_jpy")
        self.assertEqual(
            block["behavioral_state"]["state_id"], "JPY_CONSENSUS_MATURE"
        )

    def test_build_manifest_block_unknown_surface_raises(self):
        with self.assertRaises(KeyError):
            build_behavioral_surface_manifest_block("does_not_exist")

    def test_build_manifest_block_unknown_state_raises(self):
        with self.assertRaises(KeyError):
            build_behavioral_surface_manifest_block("trend_vol", "JPY_CONSENSUS_MATURE")


if __name__ == "__main__":
    unittest.main(verbosity=2)
