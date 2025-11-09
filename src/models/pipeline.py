"""
Comprehensive Training Pipeline

Integrates all components for end-to-end model training with best practices:
- Time-based train/val/test split (prevents look-ahead bias)
- Data validation and preprocessing
- Feature engineering
- Model training with cross-validation
- Model comparison and selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from ..data.validation import DataValidator, DataPreprocessor, FeatureSelector
from .baseline import BaselineModel, ModelComparison
from .ensemble import EnsembleModel, EnsembleComparison

logger = logging.getLogger(__name__)


class ModelPipeline:
    """
    End-to-end pipeline for training NBA prediction models.

    Implements best practices from research:
    - Time-based splitting (critical for sports prediction)
    - Data validation before training
    - Cross-validation for robust evaluation
    - Multiple model comparison
    - Proper handling of class imbalance
    """

    def __init__(
        self,
        output_dir: Path,
        random_state: int = 42
    ):
        """
        Args:
            output_dir: Directory to save models and results
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.random_state = random_state

        # Components
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor(
            handle_missing='mean',
            handle_outliers='clip'
        )
        self.feature_selector = None

        # Models
        self.models = {}
        self.best_model = None
        self.best_model_name = None

        # Results
        self.results = {}

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str,
        train_pct: float = 0.7,
        val_pct: float = 0.15,
        test_pct: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Prepare data with time-based splitting.

        CRITICAL: Uses chronological split to prevent look-ahead bias.
        Never use random split for time-series sports data!

        Args:
            df: Full dataset with features and target
            target_col: Name of target column
            date_col: Name of date column for chronological split
            train_pct: Percentage for training
            val_pct: Percentage for validation
            test_pct: Percentage for testing

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        logger.info("Preparing data with time-based split...")

        # Validate percentages
        assert abs(train_pct + val_pct + test_pct - 1.0) < 0.01, "Percentages must sum to 1.0"

        # Sort by date (CRITICAL for preventing look-ahead bias)
        df = df.sort_values(date_col).reset_index(drop=True)

        logger.info(f"Dataset: {len(df)} games from {df[date_col].min()} to {df[date_col].max()}")

        # Validate data
        self.validator.validate_dataset(df, 'full_dataset')

        # Separate features and target
        feature_cols = [col for col in df.columns if col not in [target_col, date_col]]
        X = df[feature_cols]
        y = df[target_col]

        # Time-based split
        n_train = int(len(df) * train_pct)
        n_val = int(len(df) * val_pct)

        X_train = X.iloc[:n_train]
        X_val = X.iloc[n_train:n_train + n_val]
        X_test = X.iloc[n_train + n_val:]

        y_train = y.iloc[:n_train]
        y_val = y.iloc[n_train:n_train + n_val]
        y_test = y.iloc[n_train + n_val:]

        logger.info(f"Train: {len(X_train)} games ({train_pct:.1%})")
        logger.info(f"Val:   {len(X_val)} games ({val_pct:.1%})")
        logger.info(f"Test:  {len(X_test)} games ({test_pct:.1%})")

        # Check class distribution
        logger.info(f"Train class distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Val class distribution:   {y_val.value_counts().to_dict()}")
        logger.info(f"Test class distribution:  {y_test.value_counts().to_dict()}")

        # Preprocess data
        logger.info("Preprocessing data...")
        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)
        X_test = self.preprocessor.transform(X_test)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_cv: bool = True
    ) -> Dict:
        """
        Train baseline models (Logistic Regression, Naive Bayes).

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            use_cv: Whether to use cross-validation

        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 60)
        logger.info("Training Baseline Models")
        logger.info("=" * 60)

        comparison = ModelComparison()

        # Logistic Regression
        lr_model = BaselineModel(model_type='logistic', scale_features=True)
        lr_model.fit(X_train, y_train)

        if use_cv:
            lr_model.cross_validate(X_train, y_train, cv=5)

        lr_model.evaluate(X_val, y_val, set_name='validation')
        comparison.add_model('Logistic Regression', lr_model)
        self.models['logistic_regression'] = lr_model

        # Naive Bayes
        nb_model = BaselineModel(model_type='naive_bayes', scale_features=True)
        nb_model.fit(X_train, y_train)

        if use_cv:
            nb_model.cross_validate(X_train, y_train, cv=5)

        nb_model.evaluate(X_val, y_val, set_name='validation')
        comparison.add_model('Naive Bayes', nb_model)
        self.models['naive_bayes'] = nb_model

        # Compare models
        comparison_df = comparison.evaluate_all(X_val, y_val)
        self.results['baseline_comparison'] = comparison_df

        return self.models

    def train_ensemble_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tune_hyperparameters: bool = True
    ) -> Dict:
        """
        Train ensemble models (Random Forest, XGBoost).

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary of trained models
        """
        logger.info("=" * 60)
        logger.info("Training Ensemble Models")
        logger.info("=" * 60)

        comparison = EnsembleComparison()

        # Random Forest
        logger.info("Training Random Forest...")
        rf_model = EnsembleModel(
            model_type='random_forest',
            tune_hyperparameters=tune_hyperparameters
        )
        rf_model.fit(X_train, y_train, X_val, y_val)
        rf_model.evaluate(X_val, y_val, set_name='validation')
        comparison.add_model('Random Forest', rf_model)
        self.models['random_forest'] = rf_model

        # Log feature importance
        rf_importance = rf_model.get_feature_importance(top_n=15)

        # XGBoost
        logger.info("Training XGBoost...")
        xgb_model = EnsembleModel(
            model_type='xgboost',
            tune_hyperparameters=tune_hyperparameters
        )
        xgb_model.fit(X_train, y_train, X_val, y_val)
        xgb_model.evaluate(X_val, y_val, set_name='validation')
        comparison.add_model('XGBoost', xgb_model)
        self.models['xgboost'] = xgb_model

        # Log feature importance
        xgb_importance = xgb_model.get_feature_importance(top_n=15)

        # Compare models
        comparison_df = comparison.evaluate_all(X_val, y_val)
        self.results['ensemble_comparison'] = comparison_df

        return self.models

    def select_best_model(self, metric: str = 'roc_auc') -> Tuple[str, object]:
        """
        Select the best model across all trained models.

        Args:
            metric: Metric to use for selection

        Returns:
            Tuple of (model_name, model)
        """
        logger.info("=" * 60)
        logger.info("Selecting Best Model")
        logger.info("=" * 60)

        best_score = -np.inf
        best_name = None
        best_model = None

        for name, model in self.models.items():
            if 'validation' in model.metrics:
                score = model.metrics['validation'][metric]
                logger.info(f"{name}: {metric} = {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_name = name
                    best_model = model

        self.best_model_name = best_name
        self.best_model = best_model

        logger.info(f"\nBest model: {best_name} ({metric} = {best_score:.4f})")

        return best_name, best_model

    def evaluate_best_model(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate the best model on test set.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of test metrics
        """
        if self.best_model is None:
            raise ValueError("Must select best model first")

        logger.info("=" * 60)
        logger.info(f"Final Evaluation: {self.best_model_name}")
        logger.info("=" * 60)

        test_metrics = self.best_model.evaluate(X_test, y_test, set_name='test')

        self.results['best_model'] = {
            'name': self.best_model_name,
            'test_metrics': test_metrics
        }

        return test_metrics

    def save_results(self):
        """Save all results and the best model."""
        logger.info("Saving results...")

        # Save best model
        model_path = self.output_dir / f'best_model_{self.best_model_name}.joblib'
        self.best_model.save(model_path)

        # Save results summary
        results_summary = {
            'timestamp': datetime.now().isoformat(),
            'best_model': self.best_model_name,
            'models_trained': list(self.models.keys()),
            'best_model_metrics': self.best_model.metrics,
            'all_results': {
                k: v.to_dict() if isinstance(v, pd.DataFrame) else v
                for k, v in self.results.items()
            }
        }

        results_path = self.output_dir / 'training_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")
        logger.info(f"Best model saved to {model_path}")

    def run_full_pipeline(
        self,
        df: pd.DataFrame,
        target_col: str,
        date_col: str,
        train_baseline: bool = True,
        train_ensemble: bool = True,
        tune_hyperparameters: bool = True
    ) -> Dict:
        """
        Run the complete training pipeline.

        Args:
            df: Full dataset
            target_col: Target column name
            date_col: Date column name
            train_baseline: Whether to train baseline models
            train_ensemble: Whether to train ensemble models
            tune_hyperparameters: Whether to tune hyperparameters

        Returns:
            Dictionary with all results
        """
        logger.info("=" * 60)
        logger.info("STARTING FULL TRAINING PIPELINE")
        logger.info("=" * 60)

        # 1. Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
            df, target_col, date_col
        )

        # 2. Train baseline models
        if train_baseline:
            self.train_baseline_models(X_train, y_train, X_val, y_val)

        # 3. Train ensemble models
        if train_ensemble:
            self.train_ensemble_models(
                X_train, y_train, X_val, y_val,
                tune_hyperparameters=tune_hyperparameters
            )

        # 4. Select best model
        self.select_best_model(metric='roc_auc')

        # 5. Evaluate on test set
        self.evaluate_best_model(X_test, y_test)

        # 6. Save results
        self.save_results()

        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        return self.results
