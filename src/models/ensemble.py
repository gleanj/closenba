"""
Ensemble Models for NBA Game Prediction

Implements Random Forest and XGBoost with hyperparameter tuning.
From research: Ensemble models generally outperform simple baselines in sports prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, make_scorer
)
import xgboost as xgb
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Base class for ensemble models with hyperparameter tuning.

    From research insights:
    - XGBoost: Highest accuracy in NBA prediction literature
    - Random Forest: Robust to overfitting, good for noisy sports data
    - Both benefit from hyperparameter tuning
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        tune_hyperparameters: bool = True,
        n_jobs: int = -1
    ):
        """
        Args:
            model_type: 'xgboost' or 'random_forest'
            tune_hyperparameters: Whether to perform grid search
            n_jobs: Number of parallel jobs (-1 = use all cores)
        """
        self.model_type = model_type
        self.tune_hyperparameters = tune_hyperparameters
        self.n_jobs = n_jobs

        # Initialize model
        if model_type == 'xgboost':
            self.model = self._create_xgboost()
        elif model_type == 'random_forest':
            self.model = self._create_random_forest()
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Store feature names and metrics
        self.feature_names = None
        self.metrics = {}
        self.best_params = None

    def _create_xgboost(self) -> xgb.XGBClassifier:
        """Create XGBoost classifier with good default parameters."""
        return xgb.XGBClassifier(
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=self.n_jobs,
            scale_pos_weight=1  # Will be set based on class imbalance
        )

    def _create_random_forest(self) -> RandomForestClassifier:
        """Create Random Forest classifier with good default parameters."""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=self.n_jobs
        )

    def _get_param_grid(self) -> Dict:
        """Get hyperparameter grid for tuning."""
        if self.model_type == 'xgboost':
            return {
                'max_depth': [3, 5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'n_estimators': [50, 100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'gamma': [0, 0.1, 0.2]
            }
        else:  # random_forest
            return {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> 'EnsembleModel':
        """
        Train the model with optional hyperparameter tuning.

        Args:
            X: Training features
            y: Training target
            X_val: Validation features (for early stopping in XGBoost)
            y_val: Validation target

        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_type} model on {len(X)} samples")

        # Store feature names
        self.feature_names = X.columns.tolist()

        # Handle class imbalance for XGBoost
        if self.model_type == 'xgboost':
            neg_pos_ratio = (y == 0).sum() / (y == 1).sum()
            self.model.set_params(scale_pos_weight=neg_pos_ratio)
            logger.info(f"Set scale_pos_weight to {neg_pos_ratio:.2f} to handle class imbalance")

        # Hyperparameter tuning
        if self.tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            self.model = self._tune_hyperparameters(X, y)
        else:
            # Simple fit
            if self.model_type == 'xgboost' and X_val is not None:
                # Use validation set for early stopping
                eval_set = [(X, y), (X_val, y_val)]
                self.model.fit(
                    X, y,
                    eval_set=eval_set,
                    verbose=False
                )
            else:
                self.model.fit(X, y)

        # Calculate training metrics
        train_pred = self.model.predict(X)
        train_prob = self.model.predict_proba(X)[:, 1]

        self.metrics['train'] = self._calculate_metrics(y, train_pred, train_prob)

        logger.info(f"Training complete. Accuracy: {self.metrics['train']['accuracy']:.3f}")

        return self

    def _tune_hyperparameters(self, X: pd.DataFrame, y: pd.Series):
        """
        Tune hyperparameters using RandomizedSearchCV.

        Uses Randomized Search (faster than Grid Search) with stratified CV.
        """
        param_grid = self._get_param_grid()

        # Use Stratified K-Fold to maintain class balance
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # Use RandomizedSearchCV for efficiency
        random_search = RandomizedSearchCV(
            self.model,
            param_distributions=param_grid,
            n_iter=50,  # Try 50 random combinations
            cv=cv,
            scoring='roc_auc',  # Optimize for ROC-AUC
            n_jobs=self.n_jobs,
            verbose=1,
            random_state=42
        )

        random_search.fit(X, y)

        self.best_params = random_search.best_params_
        logger.info(f"Best hyperparameters: {self.best_params}")
        logger.info(f"Best CV ROC-AUC: {random_search.best_score_:.3f}")

        return random_search.best_estimator_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make binary predictions."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: pd.DataFrame, y: pd.Series, set_name: str = 'test') -> Dict:
        """
        Evaluate model on a dataset.

        Args:
            X: Feature DataFrame
            y: True labels
            set_name: Name of dataset

        Returns:
            Dictionary of metrics
        """
        logger.info(f"Evaluating on {set_name} set ({len(X)} samples)")

        predictions = self.predict(X)
        probabilities = self.predict_proba(X)

        metrics = self._calculate_metrics(y, predictions, probabilities)
        self.metrics[set_name] = metrics

        # Log key metrics
        logger.info(f"{set_name.capitalize()} Metrics:")
        logger.info(f"  Accuracy:  {metrics['accuracy']:.3f}")
        logger.info(f"  Precision: {metrics['precision']:.3f}")
        logger.info(f"  Recall:    {metrics['recall']:.3f}")
        logger.info(f"  F1 Score:  {metrics['f1']:.3f}")
        logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.3f}")

        return metrics

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """Calculate all evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the trained model.

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with features and their importance scores
        """
        if self.feature_names is None:
            logger.warning("Model not trained yet")
            return None

        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.model_type == 'random_forest':
            importance = self.model.feature_importances_
        else:
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        })

        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False)

        logger.info(f"Top {top_n} most important features:")
        for idx, row in importance_df.head(top_n).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        return importance_df.head(top_n)

    def save(self, filepath: Path):
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'best_params': self.best_params
        }

        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'EnsembleModel':
        """Load model from disk."""
        model_data = joblib.load(filepath)

        instance = cls(model_type=model_data['model_type'], tune_hyperparameters=False)
        instance.model = model_data['model']
        instance.feature_names = model_data['feature_names']
        instance.metrics = model_data['metrics']
        instance.best_params = model_data['best_params']

        logger.info(f"Model loaded from {filepath}")
        return instance


class EnsembleComparison:
    """
    Compare multiple ensemble models.

    From research: XGBoost typically outperforms Random Forest,
    but both are worth testing on your specific dataset.
    """

    def __init__(self):
        self.models = {}
        self.comparison_results = None

    def add_model(self, name: str, model: EnsembleModel):
        """Add a model to comparison."""
        self.models[name] = model

    def train_all(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ):
        """Train all models."""
        logger.info(f"Training {len(self.models)} models...")

        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train, X_val, y_val)

    def evaluate_all(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Evaluate all models and create comparison table.

        Returns:
            DataFrame comparing all models
        """
        results = []

        for name, model in self.models.items():
            metrics = model.evaluate(X_test, y_test, set_name='test')

            results.append({
                'model': name,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'roc_auc': metrics['roc_auc']
            })

        self.comparison_results = pd.DataFrame(results)
        self.comparison_results = self.comparison_results.sort_values('roc_auc', ascending=False)

        logger.info("Model Comparison Results:")
        logger.info("\n" + str(self.comparison_results.to_string(index=False)))

        return self.comparison_results

    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, EnsembleModel]:
        """
        Get the best performing model.

        Args:
            metric: Metric to use for comparison

        Returns:
            Tuple of (model_name, model)
        """
        if self.comparison_results is None:
            raise ValueError("Must run evaluate_all() first")

        best_idx = self.comparison_results[metric].idxmax()
        best_name = self.comparison_results.loc[best_idx, 'model']

        return best_name, self.models[best_name]
