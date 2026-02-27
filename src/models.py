# src/models.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb


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
                 random_state: int = 42):
        """
        Args:
            n_splits:     Number of folds for TimeSeriesSplit
            random_state: Random seed for reproducibility
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
        self.results = {}

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
            verbosity=0
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