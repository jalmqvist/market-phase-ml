# src/models.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb

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
                 random_state:      int  = 42):
        self.train_window      = train_window
        self.retrain_freq      = retrain_freq
        self.confirmation_bars = confirmation_bars
        self.smooth_labels     = smooth_labels
        self.random_state      = random_state
        self._exclude_cols     = PhaseMLExperiment.EXCLUDE_COLS
        self._label_encoder    = None

    def _get_feature_cols(self, df: pd.DataFrame) -> list:
        """Auto-detect usable feature columns."""
        return [
            col for col in df.columns
            if col not in self._exclude_cols
            and df[col].dtype in ['float64', 'int64', 'float32']
            and df[col].dtype != bool
        ]

    def _build_model(self) -> xgb.XGBClassifier:
        """Build a fresh XGBoost classifier."""
        return xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',    # multi-class log loss
            verbosity=0
        )

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Run walk-forward phase prediction on full DataFrame.

        For each bar after the warmup period:
            1. Train on rolling window of train_window bars
               (only retrain every retrain_freq bars)
            2. Predict next bar's phase
            3. Store prediction

        Bars in warmup period fall back to rule-based phase.

        Args:
            df: Processed DataFrame with features and phase column

        Returns:
            Series of predicted phase labels, same index as df.
            Warmup bars contain rule-based phase (fallback).
        """
        feature_cols = self._get_feature_cols(df)
        X            = df[feature_cols].copy()
        # ------------------------------------------------------------------
        # GLOBAL FEATURE SCALING (Option 3 speed improvement)
        # ------------------------------------------------------------------
        scaler = StandardScaler()
        X_scaled_full = scaler.fit_transform(X)
        X_scaled_full = pd.DataFrame(
            X_scaled_full,
            index=X.index,
            columns=X.columns
        )

        # Get phase target — optionally smoothed
        raw_phase = df['phase'].copy()
        if self.smooth_labels:
            train_phase = smooth_phase_labels(
                raw_phase,
                confirmation_bars=self.confirmation_bars
            )
        else:
            train_phase = raw_phase.copy()

        # Next bar's phase as target
        y = train_phase.shift(-1)

        # Fixed global mapping — avoids LabelEncoder gap bug
        PHASE_MAP = {
            'HV_Ranging': 0,
            'HV_Trend': 1,
            'LV_Ranging': 2,
            'LV_Trend': 3
        }
        PHASE_MAP_INV = {v: k for k, v in PHASE_MAP.items()}

        # Encode the full y series once
        y_encoded = y.map(PHASE_MAP)  # NaNs preserved automatically

        n_bars = len(df)
        predictions = raw_phase.copy()  # fallback = rule-based
        model = None
        scaler = None
        last_trained = None

        print(f'  Walk-forward prediction: '
              f'{n_bars} bars, '
              f'warmup={self.train_window}, '
              f'retrain_freq={self.retrain_freq}')

        for i in range(self.train_window, n_bars - 1):

            should_train = (
                    model is None or
                    last_trained is None or
                    (i - last_trained) >= self.retrain_freq
            )

            if should_train:
                train_start = i - self.train_window
                train_end = i

                X_train = X_scaled_full.iloc[train_start:train_end]
                y_train = y_encoded.iloc[train_start:train_end]

                mask = X_train.notna().all(axis=1) & y_train.notna()
                X_train = X_train[mask]
                y_train = y_train[mask]

                if len(X_train) < 50:
                    continue

                y_train_int = y_train.dropna().astype(int)
                if len(y_train_int) == 0:
                    continue

                # Ensure all 4 classes are represented
                unique_classes = set(y_train_int.unique())
                missing_classes = set([0, 1, 2, 3]) - unique_classes

                if missing_classes:
                    for missing_class in missing_classes:
                        X_train = pd.concat([X_train, X_train.iloc[[-1]]], ignore_index=False)
                        y_train_int = pd.concat([
                            y_train_int,
                            pd.Series([missing_class], index=[X_train.index[-1]])
                        ])

                # Build and train model
                model = self._build_model()
                model.fit(X_train, y_train_int)

                last_trained = i

            # Predict current bar's next phase
            if model is not None:
                X_pred = X_scaled_full.iloc[i:i + 1]

                if X_pred.notna().all(axis=1).iloc[0]:
                    predicted_encoded = model.predict(X_pred)
                    predicted = PHASE_MAP_INV[int(predicted_encoded[0])]
                    predictions.iat[i] = predicted

                    # DEBUG — remove after fix
                    if i < self.train_window + 5:
                        print(f'    DEBUG bar {i}: encoded={predicted_encoded}, '
                              f'predicted={predicted}, '
                              f'stored={predictions.iloc[i]}')

        print(f'  ✓ Predictions generated for '
              f'{n_bars - self.train_window - 1} bars '
              f'({self.train_window} warmup bars use rule-based fallback)')

        # DEBUG — remove after fix
        print(f'  DEBUG predictions value_counts:')
        print(predictions.value_counts())
        print(f'  DEBUG raw_phase value_counts:')
        print(raw_phase.value_counts())

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

        Returns:
            List of column names to use as features.
        """
        feature_cols = [
            col for col in df.columns
            if col not in self.EXCLUDE_COLS
               and df[col].dtype in ['float64', 'int64', 'float32']
               and df[col].dtype != bool
        ]
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

        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

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

        mask = X.notna().all(axis=1) & y.notna()
        X, y = X[mask], y[mask]

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

            mask = (
                (phases == phase) &
                X.notna().all(axis=1) &
                y.notna()
            )
            X_phase = X[mask]
            y_phase = y[mask]

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
            mask = (
                (df['phase'] == phase) &
                X.notna().all(axis=1) &
                y.notna()
            )
            label = f'phase={phase}'
        else:
            mask = X.notna().all(axis=1) & y.notna()
            label = 'all phases'

        X, y = X[mask], y[mask]

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

        # Align all equity curves to same index
        eq_df = pd.concat(equity_curves, axis=1)
        eq_df.columns = list(equity_curves.keys())
        eq_df = eq_df.ffill()

        # For each bar, look ahead window_days
        for i in range(len(df) - self.window_days - 1):
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

    def __init__(self, random_state: int = 42):
        self.model = None
        self.label_encoder = None
        self.feature_cols = None
        self.random_state = random_state

    def get_feature_columns(self) -> list:
        """Features used for prediction."""
        return [
            'adx',
            'atr_pct',
            'plus_di',
            'minus_di',
            'rsi',
            'returns_recent',
            'volatility_recent',
        ]

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

    def train(self, training_data: pd.DataFrame) -> dict:
        """
        Train the strategy TYPE selector model.

        Args:
            training_data: Output from StrategyPerformanceTracker.compute_strategy_returns()

        Returns:
            Dict with training metrics
        """
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.model_selection import cross_val_score

        # Get base training data
        train_df = training_data.dropna(subset=self.get_feature_columns() + ['best_strategy'])

        if len(train_df) < 100:
            print(f"  ✗ Too few training samples ({len(train_df)})")
            return {}

        X = train_df[self.get_feature_columns()].copy()
        y_strategy = train_df['best_strategy'].copy()

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

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train XGBoost classifier
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss',
            verbosity=0
        )
        self.model.fit(X_scaled, y_encoded)
        self.feature_cols = self.get_feature_columns()

        # Cross-validation score
        cv_scores = cross_val_score(
            self.model, X_scaled, y_encoded, cv=3, scoring='accuracy'
        )

        print(f'\n  StrategySelector Training (3-class):')
        print(f'    Samples: {len(X)}')
        print(f'    Accuracy (CV): {cv_scores.mean():.4f} (±{cv_scores.std():.4f})')
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
            'cv_accuracy': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(X),
            'categories': list(self.label_encoder.classes_)
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

        from sklearn.preprocessing import StandardScaler

        X = features_df[self.feature_cols].copy()
        if X.isnull().any().any():
            return None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
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

        from sklearn.preprocessing import StandardScaler

        X = features_df[self.feature_cols].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        probs = self.model.predict_proba(X_scaled)[0]
        return {
            strategy: prob
            for strategy, prob in zip(self.label_encoder.classes_, probs)
        }