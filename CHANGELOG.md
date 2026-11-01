# Changelog

## Unreleased

- Added the fixed-composition causal runtime
  `mpml.runtime.phaseaware.latest_decision()` for same-bar
  `PhaseAware_TF4_MR42` decisions, with OHLCV validation, forward-label
  rejection, 126-bar derived warm-up and immutable outputs.
- Kept runtime imports out of the research/model/data-loading stack by
  lazily resolving existing `src` public exports and the strategy module's
  daily-DL feature-column constant. Existing public names and canonical
  strategy/registry behavior are retained.
- Documented the runtime, MPML/MRML responsibility boundary and its
  separation from research preparation and the existing artifact interface.
