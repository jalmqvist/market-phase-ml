# src/models.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

# ---------------------------------------------------------------------------
# DL daily feature constants
# ---------------------------------------------------------------------------

# Import D1_FEATURE_COLS from the authoritative source so that models.py
# stays in sync with dl_daily_features.py automatically.
# dl_daily_features -> dl_surface_loader -> pandas only; no circular risk.
from src.dl_daily_features import D1_FEATURE_COLS as _D1_FEATURE_COLS  # noqa: E402

#: Feature columns produced by src.dl_daily_features.compute_d1_features().
#: When DL_SIGNALS_ENABLED is True and these columns are present in the D1
#: DataFrame they are included as model features; otherwise they are excluded
#: to avoid NaN-only columns in non-DL runs.
DL_D1_FEATURE_COLS: frozenset[str] = frozenset(_D1_FEATURE_COLS)
OPTIONAL_DL_FEATURE_COLS: tuple[str, ...] = tuple(sorted(DL_D1_FEATURE_COLS))

# Core market features that are required for dynamic selector training.
REQUIRED_FEATURE_COLS: tuple[str, ...] = (
    "adx",
    "atr_pct",
    "plus_di",
    "minus_di",
    "rsi",
    "returns_recent",
    "volatility_recent",
)

#: Columns that must NEVER be used as ML features regardless of dtype.
#: These carry provenance / regime metadata and would introduce leakage.
_DL_LEAKAGE_GUARD_COLS: frozenset[str] = frozenset(
    {
        "dl_regime",
        "mpml_regime_equiv",
        "prediction_timestamp",
        "dl_prediction_timestamp",
    }
)

# Single source of truth: read DL_SIGNALS_ENABLED from src.dl_config so that
# the env-var parsing logic lives in one place and models.py stays in sync
# automatically.  dl_config only imports os and pathlib — no circular imports.
from src.dl_config import DL_SIGNALS_ENABLED  # noqa: E402


# ---------------------------------------------------------------------------
# Schema-tolerant helpers
# ---------------------------------------------------------------------------

def safe_existing_columns(df: pd.DataFrame, cols: list) -> list:
    """Return only the columns from *cols* that are present in *df*.

    Use this instead of ``df[expected_cols]`` wherever downstream pipeline
    stages must tolerate DL feature columns being present or absent depending
    on DL enabled/disabled state, per-pair artifact coverage, or model family.

    For truly required columns (e.g. 'phase', 'adx') raise explicitly with an
    informative error rather than silently dropping them.

    Example::

        safe_cols = safe_existing_columns(df, selector.feature_cols)
        X = df[safe_cols]  # never raises KeyError for optional DL columns
    """
    return [c for c in cols if c in df.columns]


def apply_optional_feature_imputation(
    X: pd.DataFrame,
    optional_feature_cols: list[str],
    *,
    fill_value: float = 0.0,
    add_missing_indicators: bool = True,
) -> pd.DataFrame:
    """Impute optional feature columns deterministically."""
    X_out = X.copy()
    optional_cols_present = [c for c in optional_feature_cols if c in X_out.columns]

    for col in optional_cols_present:
        if add_missing_indicators:
            indicator_col = f"{col}_missing"
            if indicator_col not in X_out.columns:
                X_out[indicator_col] = X_out[col].isna().astype("int8")
        X_out[col] = X_out[col].fillna(fill_value)

    return X_out


def build_training_matrix(
    X_raw: pd.DataFrame,
    y_raw: pd.Series,
    feature_cols: list[str],
    *,
    required_feature_cols: list[str] | None = None,
    optional_feature_cols: list[str] | None = None,
    diagnostics_label: str | None = None,
    add_optional_missing_indicators: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Build robust training matrices with required-only masking and optional DL imputation."""
    feature_cols = list(dict.fromkeys(feature_cols))
    X_source = X_raw.reindex(columns=feature_cols).copy()
    y_source = y_raw.copy()

    optional_set = set(
        optional_feature_cols
        if optional_feature_cols is not None
        else [c for c in feature_cols if c in OPTIONAL_DL_FEATURE_COLS]
    )
    optional_cols = [c for c in feature_cols if c in optional_set]

    if required_feature_cols is None:
        required_cols = [c for c in feature_cols if c not in optional_set]
    else:
        required_set = set(required_feature_cols)
        required_cols = [c for c in feature_cols if c in required_set]

    rows_before_mask = len(X_source)
    missingness_stats = {}
    required_mask = y_source.notna()
    if required_cols:
        required_mask &= X_source[required_cols].notna().all(axis=1)

    rows_after_required_mask = int(required_mask.sum())
    X_required = X_source.loc[required_mask].copy()
    y_required = y_source.loc[required_mask].copy()

    optional_cols_present = [c for c in optional_cols if c in X_required.columns]
    if optional_cols_present and len(X_required):
        missingness_stats = {
            col: float(X_required[col].isna().mean() * 100.0)
            for col in optional_cols_present
        }
        dl_coverage_pct = float(
            X_required[optional_cols_present].notna().any(axis=1).mean() * 100.0
        )
    else:
        dl_coverage_pct = 0.0

    rows_if_optional_required = rows_after_required_mask
    if optional_cols_present and rows_after_required_mask > 0:
        rows_if_optional_required = int(
            X_required[optional_cols_present].notna().all(axis=1).sum()
        )

    optional_collapse_pct = (
        float((rows_after_required_mask - rows_if_optional_required) / rows_after_required_mask * 100.0)
        if rows_after_required_mask > 0
        else 0.0
    )

    if optional_collapse_pct > 50.0:
        label = diagnostics_label or "training"
        print(
            f"  ⚠️  [{label}] optional DL features would collapse "
            f"{optional_collapse_pct:.2f}% of rows under full-feature masking"
        )

    X_final = apply_optional_feature_imputation(
        X_required,
        optional_cols_present,
        fill_value=0.0,
        add_missing_indicators=add_optional_missing_indicators,
    )

    rows_after_optional_imputation = len(X_final)
    effective_training_samples = rows_after_optional_imputation
    if effective_training_samples < 100:
        print(
            f"  ⚠️  [{diagnostics_label}] "
            f"very small effective sample size: "
            f"{effective_training_samples}"
        )

    if rows_after_required_mask > 0:
        min_expected_rows = int(np.floor(rows_after_required_mask * 0.5))
        assert (
            rows_after_optional_imputation >= min_expected_rows
        ), "Training rows collapsed >50% due to optional feature handling"

    diagnostics = {
        "rows_before_mask": rows_before_mask,
        "rows_after_required_mask": rows_after_required_mask,
        "rows_after_optional_imputation": rows_after_optional_imputation,
        "dl_coverage_pct": dl_coverage_pct,
        "effective_training_samples": effective_training_samples,
        "missingness_stats": missingness_stats,
    }

    if diagnostics_label is not None:
        print(
            f"  [TRAINING DIAGNOSTICS] {diagnostics_label}: "
            f"rows_before_mask={rows_before_mask} "
            f"rows_after_required_mask={rows_after_required_mask} "
            f"rows_after_optional_imputation={rows_after_optional_imputation} "
            f"dl_coverage_pct={dl_coverage_pct:.2f} "
            f"effective_training_samples={effective_training_samples}"
        )

    return X_final, y_required, diagnostics


class PhaseMLPredictor:
    """
    Walk-forward ML phase predictor.

    Predicts next bar's market phase using a rolling training
    window XGBoost classifier. Predictions are generated
    without lookahead bias — each prediction uses only data
    available up to that bar.

    Walk-forward scheme:
        - Warmup:    First train_window bars — no predictions,
                     fallback to rule-based phase labels
        - Training:  Rolling window of train_window bars
        - Retraining: Every retrain_freq bars
        - Target:    Next bar's phase (4-class classification)
        - Smoothing: Optional phase label smoothing applied
                     to training targets only

    Args:
        train_window:      Rolling training window in bars (default 504 = 2yr)
        retrain_freq:      Retraining frequency in bars (default 21 = 1 month)
        confirmation_bars: Phase smoothing confirmation period (default 5)
        smooth_labels:     If True, apply phase smoothing to training targets
        random_state:      Random seed for reproducibility
    """


    def __init__(self,
                 train_window:      int  = 504,
                 retrain_freq:      int  = 21,
                 confirmation_bars: int  = 5,
                 smooth_labels:     bool = True,
                 random_state:      int  = 42,
                 seed:              int = 42):
        self.train_window      = train_window
        self.retrain_freq      = retrain_freq
        self.confirmation_bars = confirmation_bars
        self.smooth_labels     = smooth_labels
        self.random_state      = random_state
        self.seed              = int(seed)
        self._exclude_cols     = PhaseMLExperiment.EXCLUDE_COLS
        self._label_encoder    = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """Auto-detect usable feature columns."""
        cols = [
            col for col in df.columns
            if col not in self._exclude_cols
            and df[col].dtype in ['float64', 'int64', 'float32']
            and df[col].dtype != bool
        ]
        # Gate DL D1 daily feature columns behind DL_SIGNALS_ENABLED.
        if not DL_SIGNALS_ENABLED:
            cols = [c for c in cols if c not in DL_D1_FEATURE_COLS]
        return cols

    def _build_model(self) -> xgb.XGBClassifier:
        """Build a fresh XGBoost classifier."""
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            n_jobs=1,
            eval_metric='mlogloss',    # multi-class log loss
            verbosity=0
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Run walk-forward phase prediction on full DataFrame.

        Key robustness change vs previous version:
        - Mask out rows with NaNs BEFORE fitting StandardScaler.
          This prevents sklearn RuntimeWarnings from StandardScaler.fit() when
          sparse DL features introduce NaNs in training windows.

        Training/eval semantics otherwise unchanged.
        """
        feature_cols = self._get_feature_cols(df)
        required_feature_cols = [c for c in feature_cols if c not in OPTIONAL_DL_FEATURE_COLS]
        optional_dl_feature_cols = [c for c in feature_cols if c in OPTIONAL_DL_FEATURE_COLS]
        X = df[feature_cols].copy()

        # Get phase target — optionally smoothed
        raw_phase = df["phase"].copy()
        if self.smooth_labels:
            train_phase = smooth_phase_labels(
                raw_phase,
                confirmation_bars=self.confirmation_bars,
            )
        else:
            train_phase = raw_phase.copy()

        # Next bar's phase as target
        y = train_phase.shift(-1)

        # Fixed global mapping — avoids LabelEncoder gap bug
        PHASE_MAP = {
            "HV_Ranging": 0,
            "HV_Trend": 1,
            "LV_Ranging": 2,
            "LV_Trend": 3,
        }
        PHASE_MAP_INV = {v: k for k, v in PHASE_MAP.items()}

        # Encode the full y series once
        y_encoded = y.map(PHASE_MAP)  # NaNs preserved automatically

        n_bars = len(df)
        predictions = raw_phase.copy()  # fallback = rule-based
        model = None
        scaler = None
        last_trained = None

        print(
            f"  Walk-forward prediction: "
            f"{n_bars} bars, "
            f"warmup={self.train_window}, "
            f"retrain_freq={self.retrain_freq}"
        )

        for i in range(self.train_window, n_bars - 1):
            should_train = (
                    model is None
                    or last_trained is None
                    or (i - last_trained) >= self.retrain_freq
            )

            if should_train:
                train_start = i - self.train_window
                train_end = i

                X_train_raw = X.iloc[train_start:train_end]
                y_train = y_encoded.iloc[train_start:train_end]

                X_train_raw, y_train, _ = build_training_matrix(
                    X_train_raw,
                    y_train,
                    feature_cols=feature_cols,
                    required_feature_cols=required_feature_cols,
                    optional_feature_cols=optional_dl_feature_cols,
                    diagnostics_label=f"walkforward fold={i} pair-train-window",
                )

                if len(X_train_raw) < 50:
                    continue

                scaler = StandardScaler()
                scaler.fit(X_train_raw)

                X_train = pd.DataFrame(
                    scaler.transform(X_train_raw),
                    index=X_train_raw.index,
                    columns=X_train_raw.columns,
                )

                # y_train already masked/aligned; cast to int
                y_train_int = y_train.astype(int)
                if len(y_train_int) == 0:
                    continue

                # Ensure all 4 classes are represented
                unique_classes = set(y_train_int.unique())
                missing_classes = {0, 1, 2, 3} - unique_classes

                if missing_classes:
                    for missing_class in missing_classes:
                        X_train = pd.concat([X_train, X_train.iloc[[-1]]], ignore_index=False)
                        y_train_int = pd.concat(
                            [
                                y_train_int,
                                pd.Series([missing_class], index=[X_train.index[-1]]),
                            ]
                        )

                model = self._build_model()
                model.fit(X_train, y_train_int)

                last_trained = i

            # Predict current bar's next phase
            if model is not None and scaler is not None:
                X_pred_raw = X.iloc[i: i + 1]

                if required_feature_cols and not X_pred_raw[required_feature_cols].notna().all(axis=1).iloc[0]:
                    continue

                X_pred_raw = apply_optional_feature_imputation(
                    X_pred_raw,
                    optional_dl_feature_cols,
                    fill_value=0.0,
                    add_missing_indicators=True,
                )

                X_pred = pd.DataFrame(
                    scaler.transform(X_pred_raw),
                    index=X_pred_raw.index,
                    columns=X_pred_raw.columns,
                )

                predicted_encoded = model.predict(X_pred)
                predicted = PHASE_MAP_INV[int(predicted_encoded[0])]
                predictions.iat[i] = predicted

        print(
            f"  ✓ Predictions generated for "
            f"{n_bars - self.train_window - 1} bars "
            f"({self.train_window} warmup bars use rule-based fallback)"
        )

        return predictions

    def evaluate(self,
                 df: pd.DataFrame,
                 predictions: pd.Series) -> dict:
        """
        Evaluate prediction accuracy vs rule-based phases.

        Compares:
            - ML accuracy vs true next-bar phase
            - Rule-based accuracy vs true next-bar phase
              (baseline: assume current phase continues)

        Args:
            df:          Processed DataFrame with true phase labels
            predictions: Output of fit_predict()

        Returns:
            Dict with accuracy metrics and per-phase breakdown.
        """
        true_next = df['phase'].shift(-1).dropna()

        # Decode predictions to strings using PHASE_MAP_INV
        PHASE_MAP_INV = {
            0: 'HV_Ranging',
            1: 'HV_Trend',
            2: 'LV_Ranging',
            3: 'LV_Trend'
        }
        pred_aligned = predictions.reindex(true_next.index)

        # Only evaluate on non-warmup bars
        warmup_end  = df.index[self.train_window]
        eval_mask   = true_next.index >= warmup_end

        true_eval = true_next[eval_mask]
        pred_eval = pred_aligned[eval_mask]

        # Rule-based baseline: assume current phase = next phase
        rule_based = df['phase'].reindex(true_next.index)[eval_mask]

        ml_accuracy = accuracy_score(true_eval, pred_eval)
        rb_accuracy = accuracy_score(true_eval, rule_based)

        # Per-phase accuracy
        phase_accuracy = {}
        for phase in df['phase'].unique():
            mask = true_eval == phase
            if mask.sum() > 0:
                phase_accuracy[phase] = {
                    'ml':         accuracy_score(true_eval[mask], pred_eval[mask]),
                    'rule_based': accuracy_score(true_eval[mask], rule_based[mask]),
                    'n_samples':  int(mask.sum())
                }

        print(f'\n  Phase prediction accuracy:')
        print(f'    ML predicted:  {ml_accuracy:.4f}')
        print(f'    Rule-based:    {rb_accuracy:.4f}')
        print(f'    Improvement:   {(ml_accuracy - rb_accuracy) * 100:+.2f}pp')
        print(f'\n  Per-phase breakdown:')
        print(f'    {"Phase":<20} {"ML":>8} {"Rule":>8} {"N":>8}')
        print(f'    {"-" * 48}')
        for phase, scores in sorted(phase_accuracy.items()):
            print(
                f'    {phase:<20} '
                f'{scores["ml"]:>8.4f} '
                f'{scores["rule_based"]:>8.4f} '
                f'{scores["n_samples"]:>8}'
            )

        return {
            'ml_accuracy':       ml_accuracy,
            'rule_based_accuracy': rb_accuracy,
            'improvement_pp':    ml_accuracy - rb_accuracy,
            'phase_accuracy':    phase_accuracy,
            'n_eval_bars':       int(eval_mask.sum())
        }
class PhaseMLExperiment:
    """
    ML experiment comparing phase-aware vs phase-agnostic
    prediction approaches on a single currency pair.

    Three experiments:
        1. Baseline:       Single XGBoost model, no phase information
        2. Phase features: Single model + phase one-hot encoded
        3. Phase models:   Separate model trained per phase

    All experiments use TimeSeriesSplit cross-validation to
    avoid look-ahead bias — never random splits on time series.

    Feature columns are auto-detected from the DataFrame,
    excluding OHLCV, target variables, phase labels and
    any boolean/categorical columns added by phases.py.
    """

    # Columns to always exclude from feature matrix
    EXCLUDE_COLS = {
        # Raw OHLCV
        'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',

        # Return columns (targets and their derivatives)
        'returns', 'log_returns',
        'next_return', 'next_direction', 'next_direction_binary',

        # Phase label and boolean flags from phases.py
        'phase', 'trending', 'high_vol',

        # Position sizing columns from phases.py
        'stop_atr_mult', 'size_multiplier',

        # DL leakage guard — regime metadata and inference timestamps must
        # never be used as ML features (see _DL_LEAKAGE_GUARD_COLS).
        'dl_regime', 'mpml_regime_equiv',
        'prediction_timestamp', 'dl_prediction_timestamp',
    }

    def __init__(self,
                 n_splits: int = 5,
                 random_state: int = 42,
                 smooth_labels: bool = True,    # new
                 confirmation_bars: int = 5):   # new
        """
        Args:
            n_splits:          Number of folds for TimeSeriesSplit
            random_state:      Random seed for reproducibility
            smooth_labels:     If True, apply phase label smoothing
                               to ML training targets
            confirmation_bars: Minimum bars to confirm a new phase
                               Only used if smooth_labels=True
        """
        self.n_splits          = n_splits
        self.random_state      = random_state
        self.smooth_labels     = smooth_labels
        self.confirmation_bars = confirmation_bars
        self.tscv              = TimeSeriesSplit(n_splits=n_splits)
        self.results           = {}

    def _get_phase_target(self, df: pd.DataFrame) -> pd.Series:
        """
        Get phase target series, optionally smoothed.

        Returns next bar's phase as the prediction target,
        with optional smoothing applied to remove noise.

        Returns:
            Series of phase labels shifted by -1
            (predicting next bar's phase)
        """
        phase = df['phase'].copy()

        if self.smooth_labels:
            phase = smooth_phase_labels(
                phase,
                confirmation_bars=self.confirmation_bars
            )
            print(
                f'  Phase smoothing applied '
                f'(confirmation_bars={self.confirmation_bars})'
            )

        # Shift by -1 to predict NEXT bar's phase
        return phase.shift(-1)

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Auto-detect usable feature columns.

        Includes only numeric columns (float64, int64, float32)
        that are not in EXCLUDE_COLS and not boolean dtype.

        The 4-phase scheme produces these numeric features
        that WILL be included:
            atr, atr_pct, adx, plus_di, minus_di,
            return_lag_*, adx_lag_*,
            return_mean_*, return_std_*, return_skew_*,
            di_spread, di_ratio

        DL D1 daily features (``dl_signal_*``) are included only when
        ``DL_SIGNALS_ENABLED=true`` and the columns are present in *df*.

        Returns:
            List of column names to use as features.
        """
        feature_cols = [
            col for col in df.columns
            if col not in self.EXCLUDE_COLS
               and df[col].dtype in ['float64', 'int64', 'float32']
               and df[col].dtype != bool
        ]

        # Gate DL D1 daily feature columns behind DL_SIGNALS_ENABLED.
        # When DL signals are disabled the columns are absent from the
        # DataFrame, but guard explicitly in case they are present with NaN.
        if not DL_SIGNALS_ENABLED:
            feature_cols = [c for c in feature_cols if c not in DL_D1_FEATURE_COLS]

        return feature_cols

    def prepare_features(self,
                         df: pd.DataFrame,
                         include_phase: bool = False
                         ) -> pd.DataFrame:
        """
        Prepare feature matrix from processed DataFrame.

        Args:
            df:            DataFrame with indicators and phases
            include_phase: If True, one-hot encode phase label
                           as additional features

        Returns:
            Feature DataFrame ready for model training.
        """
        features = self.get_feature_columns(df)
        X = df[features].copy()

        if include_phase:
            phase_dummies = pd.get_dummies(
                df['phase'], prefix='phase'
            )
            X = pd.concat([X, phase_dummies], axis=1)

        return X

    def _build_model(self) -> xgb.XGBClassifier:
        """
        Build a fresh XGBoost classifier.

        Centralised here so hyperparameters are consistent
        across all three experiments and easy to tune.
        """
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=1,
            eval_metric='logloss',
            objective='binary:logistic',
            verbosity=0,
        )

    def _cross_validate(self,
                        model,
                        X: pd.DataFrame,
                        y: pd.Series) -> dict:
        """
        Time-series cross-validation with StandardScaler.

        IMPORTANT: Scaler is fit on training fold only and
        applied to test fold — no data leakage.

        Args:
            model: Sklearn-compatible classifier
            X:     Feature DataFrame
            y:     Target Series

        Returns:
            Dict with accuracy_mean, accuracy_std,
            accuracy_scores (per fold), n_samples.
        """
        accuracy_scores = []

        for train_idx, test_idx in self.tscv.split(X):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            # Fit scaler on train only
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)

            accuracy_scores.append(
                accuracy_score(y_test, y_pred)
            )

        return {
            'accuracy_mean': np.mean(accuracy_scores),
            'accuracy_std': np.std(accuracy_scores),
            'accuracy_scores': accuracy_scores,
            'n_samples': len(X)
        }

    def _print_scores(self, name: str, scores: dict) -> None:
        """Print formatted cross-validation scores."""
        fold_strs = [
            f'{s:.4f}' for s in scores['accuracy_scores']
        ]
        print(f'\n  Results for {name}:')
        print(
            f'  Accuracy: {scores["accuracy_mean"]:.4f} '
            f'(±{scores["accuracy_std"]:.4f})'
        )
        print(f'  Fold scores: {fold_strs}')

    def run_baseline(self, df: pd.DataFrame) -> dict:
        """
        Experiment 1: Single XGBoost, no phase information.

        This is the control — establishes whether market
        direction is predictable at all from technical
        indicators alone.
        """
        print('\n' + '=' * 50)
        print('EXPERIMENT 1: Baseline (No Phase Information)')
        print('=' * 50)

        X = self.prepare_features(df, include_phase=False)
        y = df['next_direction_binary']
        required_cols = [c for c in X.columns if c not in OPTIONAL_DL_FEATURE_COLS]
        optional_cols = [c for c in X.columns if c in OPTIONAL_DL_FEATURE_COLS]
        X, y, _ = build_training_matrix(
            X,
            y,
            feature_cols=list(X.columns),
            required_feature_cols=required_cols,
            optional_feature_cols=optional_cols,
            diagnostics_label="per-phase models baseline",
        )

        scores = self._cross_validate(self._build_model(), X, y)
        self.results['baseline'] = scores
        self._print_scores('Baseline', scores)

        return scores

    def run_phase_features(self, df: pd.DataFrame) -> dict:
        """
        Experiment 2: Single XGBoost with phase one-hot encoded.

        Tests whether the model can learn phase-conditional
        patterns when given phase as an explicit feature.
        If this beats baseline, phase information is useful
        but the model can integrate it itself.
        """
        print('\n' + '=' * 50)
        print('EXPERIMENT 2: Phase as Feature')
        print('=' * 50)

        X = self.prepare_features(df, include_phase=True)
        y = df['next_direction_binary']
        required_cols = [c for c in X.columns if c not in OPTIONAL_DL_FEATURE_COLS]
        optional_cols = [c for c in X.columns if c in OPTIONAL_DL_FEATURE_COLS]
        X, y, _ = build_training_matrix(
            X,
            y,
            feature_cols=list(X.columns),
            required_feature_cols=required_cols,
            optional_feature_cols=optional_cols,
            diagnostics_label="per-phase models phase-features",
        )

        scores = self._cross_validate(self._build_model(), X, y)
        self.results['phase_features'] = scores
        self._print_scores('Phase Features', scores)

        return scores

    def run_phase_models(self,
                         df: pd.DataFrame,
                         min_samples: int = 100) -> dict:
        """
        Experiment 3: Separate XGBoost per phase.

        Each model is trained and evaluated only on data
        from its own phase. Tests whether phase-specific
        patterns are strong enough to be learnable when
        the model is not distracted by other regimes.

        Phases with fewer than min_samples are skipped —
        with the 4-phase scheme all phases should have
        sufficient samples on D1 data.

        Args:
            df:          DataFrame with features and phase labels
            min_samples: Minimum samples to train a phase model
        """
        print('\n' + '=' * 50)
        print('EXPERIMENT 3: Separate Models per Phase')
        print('=' * 50)

        X = self.prepare_features(df, include_phase=False)
        y = df['next_direction_binary']
        phases = df['phase']

        phase_scores = {}
        phase_counts = phases.value_counts()

        for phase in phase_counts.index:
            count = phase_counts[phase]

            if count < min_samples:
                print(f'  Skipping {phase}: '
                      f'only {count} samples')
                continue

            phase_mask = phases == phase
            X_phase_raw = X[phase_mask]
            y_phase_raw = y[phase_mask]
            required_cols = [c for c in X_phase_raw.columns if c not in OPTIONAL_DL_FEATURE_COLS]
            optional_cols = [c for c in X_phase_raw.columns if c in OPTIONAL_DL_FEATURE_COLS]
            X_phase, y_phase, _ = build_training_matrix(
                X_phase_raw,
                y_phase_raw,
                feature_cols=list(X_phase_raw.columns),
                required_feature_cols=required_cols,
                optional_feature_cols=optional_cols,
                diagnostics_label=f"per-phase models phase={phase}",
            )

            print(f'\n  Phase: {phase} ({len(X_phase)} samples)')

            scores = self._cross_validate(
                self._build_model(), X_phase, y_phase
            )
            phase_scores[phase] = scores

            print(
                f'    Accuracy: {scores["accuracy_mean"]:.4f} '
                f'(±{scores["accuracy_std"]:.4f})'
            )

        self.results['phase_models'] = phase_scores
        return phase_scores

    def compare_results(self) -> pd.DataFrame:
        """
        Build and print a comparison table of all experiments.

        Returns:
            DataFrame with one row per experiment,
            sorted by accuracy descending.
        """
        print('\n' + '=' * 50)
        print('FINAL ML COMPARISON')
        print('=' * 50)

        rows = []

        if 'baseline' in self.results:
            rows.append({
                'Model':     'Baseline (No Phases)',
                'Accuracy':  self.results['baseline'] ['accuracy_mean'],
                'Std':       self.results['baseline'] ['accuracy_std'],
                'N Samples': self.results['baseline'] ['n_samples']
            })

        if 'phase_features' in self.results:
            rows.append({
                'Model':     'Phase as Feature',
                'Accuracy':  self.results['phase_features'] ['accuracy_mean'],
                'Std':       self.results['phase_features'] ['accuracy_std'],
                'N Samples': self.results['phase_features'] ['n_samples']
            })

        if 'phase_models' in self.results:
            phase_accs = [
                v['accuracy_mean']
                for v in self.results['phase_models'].values()
            ]
            rows.append({
                'Model':     'Separate Phase Models (avg)',
                'Accuracy':  np.mean(phase_accs),
                'Std':       np.std(phase_accs),
                'N Samples': sum(
                    v['n_samples']
                    for v in self.results['phase_models'].values()
                )
            })

        if not rows:
            print('  No results to compare yet.')
            return pd.DataFrame()

        comparison_df = (
            pd.DataFrame(rows)
            .sort_values('Accuracy', ascending=False)
        )
        print(comparison_df.to_string(index=False))
        return comparison_df

    def get_feature_importance(self,
                                df: pd.DataFrame,
                                phase: str = None
                                ) -> pd.DataFrame:
        """
        Train a single model and return feature importances.

        Useful for understanding which indicators drive
        predictions, either globally or within a specific phase.

        Args:
            df:    Processed DataFrame
            phase: If provided, train only on that phase's data.
                   If None, train on full dataset.

        Returns:
            DataFrame with feature names and importance scores,
            sorted descending.
        """
        X = self.prepare_features(df, include_phase=False)
        y = df['next_direction_binary']

        if phase is not None:
            phase_mask = df['phase'] == phase
            X_raw = X[phase_mask]
            y_raw = y[phase_mask]
            label = f'phase={phase}'
        else:
            X_raw = X
            y_raw = y
            label = 'all phases'

        required_cols = [c for c in X_raw.columns if c not in OPTIONAL_DL_FEATURE_COLS]
        optional_cols = [c for c in X_raw.columns if c in OPTIONAL_DL_FEATURE_COLS]
        X, y, _ = build_training_matrix(
            X_raw,
            y_raw,
            feature_cols=list(X_raw.columns),
            required_feature_cols=required_cols,
            optional_feature_cols=optional_cols,
            diagnostics_label=f"feature-importance {label}",
        )

        if len(X) < 50:
            print(f'  Too few samples for feature importance '
                  f'({len(X)} rows for {label})')
            return pd.DataFrame()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = self._build_model()
        model.fit(X_scaled, y)

        importance_df = pd.DataFrame({
            'feature':   X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f'\n  Top 10 features ({label}):')
        print(importance_df.head(10).to_string(index=False))

        return importance_df

def smooth_phase_labels(phase_series: pd.Series,
                        confirmation_bars: int = 5) -> pd.Series:
    """
    Smooth phase labels by requiring a new phase to persist
    for at least confirmation_bars before it is accepted.

    Until a new phase is confirmed, the previous phase label
    is retained. This removes brief phase spikes (1-5 bars)
    that are likely noise rather than genuine regime changes.

    Only applied to ML training labels — does not modify the
    rule-based phase detection in phases.py.

    Args:
        phase_series:      Raw phase labels from phases.py
        confirmation_bars: Minimum consecutive bars required
                           to confirm a new phase (default 5)

    Returns:
        Smoothed phase Series with same index as input.

    Example:
        Raw:      LV_Ranging, LV_Ranging, HV_Trend, LV_Ranging, LV_Ranging
        Smoothed: LV_Ranging, LV_Ranging, LV_Ranging, LV_Ranging, LV_Ranging
        (HV_Trend lasted only 1 bar — not confirmed, absorbed into LV_Ranging)
    """
    if confirmation_bars <= 1:
        return phase_series.copy()

    smoothed      = phase_series.copy()
    current_phase = phase_series.iloc[0]
    candidate     = phase_series.iloc[0]
    candidate_len = 1

    for i in range(1, len(phase_series)):
        new_phase = phase_series.iloc[i]

        if new_phase == current_phase:
            candidate     = current_phase
            candidate_len = 1
            smoothed.iat[i] = current_phase

        elif new_phase == candidate and new_phase != current_phase:
            candidate_len += 1

            if candidate_len >= confirmation_bars:
                current_phase = candidate
                smoothed.iat[i] = current_phase
                # Backfill the confirmation period
                backfill_idx = smoothed.index[i - candidate_len + 1: i + 1]
                smoothed.loc[backfill_idx] = current_phase
            else:
                smoothed.iat[i] = current_phase

        else:
            candidate     = new_phase
            candidate_len = 1
            smoothed.iat[i] = current_phase
    return smoothed

class StrategyPerformanceTracker:
    """
    Tracks per-strategy performance in rolling windows.
    Used to train the strategy selector ML model.

    For each bar, we collect:
    - Current market state (ADX, ATR%, phase, etc.)
    - Which strategy performed best over next N bars

    This creates training data for: "Given this phase and these indicators,
    which strategy (TF4, MR42, TF5, PhaseAware_TF4_MR42) will win?"
    """

    def __init__(self, window_days: int = 20):
        """
        Args:
            window_days: Number of bars ahead to measure performance
                        (default 20 = ~1 trading month)
        """
        self.window_days = window_days

    def compute_strategy_returns(self,
                                 df: pd.DataFrame,
                                 strategy_results: dict) -> pd.DataFrame:
        """
        For each bar, compute which strategy had the best return
        over the next window_days bars.

        Args:
            df:                 Processed DataFrame with phases
            strategy_results:   Dict of strategy backtest results
                               (from run_backtests)

        Returns:
            DataFrame with columns:
            - phase: current phase
            - adx, atr_pct, returns: features
            - best_strategy: which strategy won (target variable)
            - strategy_returns_*: returns for each strategy (for analysis)
        """
        training_data = []

        # Get equity curves for each strategy
        equity_curves = {
            name: results['equity_curve']
            for name, results in strategy_results.items()
        }
        # Align all equity curves to the bar index (df.index)
        eq_df = pd.concat(equity_curves, axis=1)
        eq_df.columns = list(equity_curves.keys())

        # Force equity curves to have exactly the same index as df (one value per bar)
        eq_df = eq_df.reindex(df.index).ffill().bfill()

        # Use the aligned length (should match len(df) after reindex, but keep it explicit)
        n_bars = len(eq_df)
        last_i = n_bars - self.window_days - 1
        if last_i <= 0:
            return pd.DataFrame()

        # For each bar, look ahead window_days
        for i in range(last_i):
            current_idx = df.index[i]
            lookahead_idx = df.index[i + self.window_days]

            # Current bar features
            row = {
                'date': current_idx,
                'phase': df['phase'].iloc[i],
                'adx': df['adx'].iloc[i],
                'atr_pct': df['atr_pct'].iloc[i],
                'plus_di': df['plus_di'].iloc[i],
                'minus_di': df['minus_di'].iloc[i],
                'rsi': df['rsi'].iloc[i],
                'returns_recent': df['returns'].iloc[max(0, i - 5):i].mean(),
                'volatility_recent': df['atr_pct'].iloc[max(0, i - 5):i].mean(),
            }

            # Propagate DL feature columns when present (DL_SIGNALS_ENABLED=True path).
            # When DL is disabled, DL_D1_FEATURE_COLS are absent from df so this loop is a no-op.
            for col in DL_D1_FEATURE_COLS:
                if col in df.columns:
                    row[col] = df[col].iloc[i]

            # Strategy returns over next window_days
            for strategy_name in equity_curves.keys():
                eq_current = eq_df[strategy_name].iloc[i]
                eq_future = eq_df[strategy_name].iloc[i + self.window_days]
                ret = (eq_future - eq_current) / eq_current
                row[f'{strategy_name}_return'] = ret

            # Which strategy had best return?
            strategy_returns = {
                name: row[f'{name}_return']
                for name in equity_curves.keys()
            }
            best_strategy = max(strategy_returns, key=strategy_returns.get)

            row['best_strategy'] = best_strategy
            row['best_return'] = strategy_returns[best_strategy]

            training_data.append(row)

        return pd.DataFrame(training_data)


class StrategySelector:
    """
    ML model to select which strategy TYPE to run given current market state.

    Predicts: TrendFollowing vs MeanReversion vs PhaseAware
    (3-class problem, much more learnable than 31-class)
    """

    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        self.model = None
        self.label_encoder = None
        self.feature_cols = None
        self.required_feature_cols = None
        self.optional_feature_cols = None
        self.scaler = None
        self.random_state = seed



    def get_feature_columns(self) -> list:
        """Features used for prediction."""
        return list(REQUIRED_FEATURE_COLS)

    @staticmethod
    def _strategy_to_category(strategy_name: str) -> str:
        """Map strategy name to category."""
        if strategy_name.startswith('TF'):
            return 'TrendFollowing'
        elif strategy_name.startswith('MR'):
            return 'MeanReversion'
        elif strategy_name.startswith('PhaseAware'):
            return 'PhaseAware'
        else:
            return None


    def train(
        self,
        training_data: pd.DataFrame,
        do_cv: bool = True,
        cv_folds: int = 3,
        diagnostics_label: str | None = None,
    ) -> dict:

        """
        Train the strategy TYPE selector model.

        Args:
            training_data: Output from StrategyPerformanceTracker.compute_strategy_returns()

        Returns:
            Dict with training metrics
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import StratifiedKFold, cross_val_score
        from sklearn.pipeline import Pipeline

        # Build feature column list: base features + any DL columns present in training_data.
        # DL columns are absent in baseline runs (DL_SIGNALS_ENABLED=False) so this is a no-op
        # in that case and baseline behavior is fully preserved.
        base_feature_cols = self.get_feature_columns()
        dl_feature_cols = [
            c for c in DL_D1_FEATURE_COLS
            if c in training_data.columns and c not in _DL_LEAKAGE_GUARD_COLS
            # _DL_LEAKAGE_GUARD_COLS excludes regime metadata and inference timestamps
            # (dl_regime, mpml_regime_equiv, prediction_timestamp, dl_prediction_timestamp)
            # that would introduce future leakage if used as model features.
        ]
        if DL_SIGNALS_ENABLED and dl_feature_cols:
            all_feature_cols = base_feature_cols + dl_feature_cols
        else:
            all_feature_cols = base_feature_cols

        X_train, y_strategy, diag = build_training_matrix(
            training_data,
            training_data["best_strategy"],
            feature_cols=all_feature_cols,
            required_feature_cols=base_feature_cols,
            optional_feature_cols=dl_feature_cols,
            diagnostics_label=diagnostics_label or "dynamic selector training",
        )

        if len(X_train) < 100:
            print(f"  ✗ Too few training samples ({len(X_train)})")
            return {}

        X = X_train.copy()

        # ✅ Map 31 strategies to 3 categories
        y_category = y_strategy.apply(self._strategy_to_category)

        # Remove rows where category is None
        valid_mask = y_category.notna()
        X = X[valid_mask]
        y_category = y_category[valid_mask]

        if len(X) < 50:
            print(f"  ✗ Too few valid samples ({len(X)})")
            return {}

        # Encode categories
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y_category)

        # Leakage-free cross-validation: scaler fitted inside each CV fold
        cv_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', xgb.XGBClassifier(
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='mlogloss',
                verbosity=0,
                random_state=self.seed,
                n_jobs=1
            ))
        ])

        # Optional CV (skip during walk-forward; outer loop is the evaluation)
        if do_cv:
            skf = StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=self.seed
            )
            cv_scores = cross_val_score(
                cv_pipeline, X, y_encoded, cv=skf, scoring='accuracy'
            )
            cv_mean = float(cv_scores.mean())
            cv_std = float(cv_scores.std())
        else:
            cv_scores = None
            cv_mean = float("nan")
            cv_std = float("nan")

        # Fit final scaler and model on full training set
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='mlogloss',
            verbosity=0,
            random_state=self.seed,
            n_jobs=1,
        )

        self.model.fit(X_scaled, y_encoded)
        self.feature_cols = list(X.columns)
        self.required_feature_cols = list(base_feature_cols)
        self.optional_feature_cols = list(dl_feature_cols)

        print(f'\n  StrategySelector Training (3-class):')
        print(f'    Samples: {len(X)}')
        # print(f'    Accuracy (CV): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')
        print(f'    Accuracy (CV): {cv_mean:.4f} (±{cv_std:.4f})' if do_cv else '    Accuracy (CV): skipped (walk-forward)')
        print(f'    Categories: {list(self.label_encoder.classes_)}')

        # Feature importance
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print(f'\n  Top 5 features:')
        for _, row in importance.head(5).iterrows():
            print(f'    {row["feature"]:<20} {row["importance"]:.4f}')

        return {
            'cv_accuracy': cv_mean,
            'cv_std': cv_std,
            'n_samples': len(X),
            'categories': list(self.label_encoder.classes_),
            'diagnostics': diag,
        }

    def predict(self, features_df: pd.DataFrame) -> str:
        """
        Predict best strategy TYPE given current market state.

        Args:
            features_df: DataFrame with one row, columns matching get_feature_columns()

        Returns:
            Strategy type: 'TrendFollowing', 'MeanReversion', or 'PhaseAware'
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        # Use reindex so that columns absent from features_df (e.g. DL columns
        # that were trained on but are missing in the current inference slice)
        # become NaN rather than raising KeyError.
        X = features_df.reindex(columns=self.feature_cols).copy()
        required_cols = [c for c in (self.required_feature_cols or []) if c in X.columns]
        if required_cols and X[required_cols].isnull().any().any():
            return 'PhaseAware'  # or: return None
        X = apply_optional_feature_imputation(
            X,
            self.optional_feature_cols or [],
            fill_value=0.0,
            add_missing_indicators=False,
        )
        if X.isnull().any().any():
            return 'PhaseAware'

        if getattr(self, "scaler", None) is None:
            raise ValueError("Scaler not fitted. Train() must set self.scaler.")

        X_scaled = self.scaler.transform(X)

        y_pred_encoded = self.model.predict(X_scaled)[0]
        y_pred = self.label_encoder.inverse_transform([y_pred_encoded])[0]

        return y_pred

    def predict_proba(self, features_df: pd.DataFrame) -> dict:
        """
        Predict probability distribution over strategy types.

        Useful for weighted portfolio allocation.

        Returns:
            Dict: {strategy_type: probability}
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call train() first.")

        # Use reindex so missing columns (e.g. absent DL columns) become NaN
        # instead of raising KeyError; callers check for NaN before predicting.
        X = features_df.reindex(columns=self.feature_cols).copy()
        required_cols = [c for c in (self.required_feature_cols or []) if c in X.columns]
        if required_cols and X[required_cols].isnull().any().any():
            raise ValueError("Required selector features contain NaN")
        X = apply_optional_feature_imputation(
            X,
            self.optional_feature_cols or [],
            fill_value=0.0,
            add_missing_indicators=False,
        )
        X_scaled = self.scaler.transform(X)

        probs = self.model.predict_proba(X_scaled)[0]
        return {
            strategy: prob
            for strategy, prob in zip(self.label_encoder.classes_, probs)
        }

    def predict_proba_df(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized predict_proba for a feature DataFrame with columns = feature_cols.
        Returns ndarray shape (n_samples, n_classes).
        """
        # Use reindex so missing columns become NaN instead of raising KeyError.
        X = X_df.reindex(columns=self.feature_cols).copy()
        required_cols = [c for c in (self.required_feature_cols or []) if c in X.columns]
        if required_cols and X[required_cols].isnull().any().any():
            raise ValueError("Required selector features contain NaN")
        X = apply_optional_feature_imputation(
            X,
            self.optional_feature_cols or [],
            fill_value=0.0,
            add_missing_indicators=False,
        )
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
