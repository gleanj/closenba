"""
Baseline Models for NBA Game Prediction

Implements simple, interpretable models as benchmarks.
From research: Simple models often outperform complex ones in sports prediction.

OPTIMIZED: Uses shared metrics module to eliminate code duplication.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
from pathlib import Path

from ..utils.metrics import calculate_classification_metrics

logger = logging.getLogger(__name__)


class BaselineModel:
    """
    Base class for baseline prediction models.
    
    From research insights:
    - Logistic Regression: Most computationally efficient, highly interpretable
    - Gaussian Naive Bayes: Strong performer, good with PCA
    """
    
    def __init__(self, model_type: str = 'logistic', scale_features: bool = True):
        """
        Args:
            model_type: 'logistic' or 'naive_bayes'
            scale_features: Whether to standardize features
        """
        self.model_type = model_type
        self.scale_features = scale_features
        
        # Initialize model
        if model_type == 'logistic':
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'  # Handle class imbalance
            )
        elif model_type == 'naive_bayes':
            self.model = GaussianNB()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Scaler for feature normalization
        self.scaler = StandardScaler() if scale_features else None
        
        # Store feature names and metrics
        self.feature_names = None
        self.metrics = {}
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaselineModel':
        """
        Train the model.
        
        Args:
            X: Feature DataFrame
            y: Target variable (binary: both teams led by 5+)
        
        Returns:
            Self for method chaining
        """
        logger.info(f"Training {self.model_type} model on {len(X)} samples")
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale features if needed
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Calculate training metrics
        train_pred = self.model.predict(X_scaled)
        train_prob = self.model.predict_proba(X_scaled)[:, 1]
        
        self.metrics['train'] = self._calculate_metrics(y, train_pred, train_prob)
        
        logger.info(f"Training complete. Accuracy: {self.metrics['train']['accuracy']:.3f}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of binary predictions
        """
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of probabilities for positive class
        """
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series, set_name: str = 'test') -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            X: Feature DataFrame
            y: True labels
            set_name: Name of dataset (e.g., 'test', 'validation')
        
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
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """
        Perform cross-validation on the model.

        Args:
            X: Feature DataFrame
            y: Target variable
            cv: Number of cross-validation folds

        Returns:
            Dictionary with cross-validation metrics
        """
        logger.info(f"Running {cv}-fold cross-validation...")

        # Scale features if needed
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values

        # Stratified K-Fold to maintain class balance
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        # Calculate multiple metrics
        cv_metrics = {
            'accuracy': cross_val_score(self.model, X_scaled, y, cv=skf, scoring='accuracy'),
            'precision': cross_val_score(self.model, X_scaled, y, cv=skf, scoring='precision', error_score=0),
            'recall': cross_val_score(self.model, X_scaled, y, cv=skf, scoring='recall', error_score=0),
            'f1': cross_val_score(self.model, X_scaled, y, cv=skf, scoring='f1', error_score=0),
            'roc_auc': cross_val_score(self.model, X_scaled, y, cv=skf, scoring='roc_auc')
        }

        # Calculate mean and std for each metric
        cv_results = {}
        for metric, scores in cv_metrics.items():
            cv_results[f'{metric}_mean'] = scores.mean()
            cv_results[f'{metric}_std'] = scores.std()

        logger.info(f"Cross-validation results:")
        logger.info(f"  Accuracy:  {cv_results['accuracy_mean']:.3f} (+/- {cv_results['accuracy_std']:.3f})")
        logger.info(f"  Precision: {cv_results['precision_mean']:.3f} (+/- {cv_results['precision_std']:.3f})")
        logger.info(f"  Recall:    {cv_results['recall_mean']:.3f} (+/- {cv_results['recall_std']:.3f})")
        logger.info(f"  F1:        {cv_results['f1_mean']:.3f} (+/- {cv_results['f1_std']:.3f})")
        logger.info(f"  ROC-AUC:   {cv_results['roc_auc_mean']:.3f} (+/- {cv_results['roc_auc_std']:.3f})")

        return cv_results

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """
        Calculate all evaluation metrics.

        OPTIMIZED: Uses shared metrics utility to eliminate code duplication.
        """
        return calculate_classification_metrics(
            y_true=y_true.values if isinstance(y_true, pd.Series) else y_true,
            y_pred=y_pred,
            y_prob=y_prob
        )
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance (for Logistic Regression only).
        
        Returns:
            DataFrame with features and their coefficients
        """
        if self.model_type != 'logistic' or self.feature_names is None:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.model.coef_[0]
        })
        
        # Sort by absolute coefficient value
        importance_df['abs_coef'] = importance_df['coefficient'].abs()
        importance_df = importance_df.sort_values('abs_coef', ascending=False)
        
        return importance_df.drop('abs_coef', axis=1)
    
    def save(self, filepath: Path):
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path) -> 'BaselineModel':
        """Load model from disk."""
        model_data = joblib.load(filepath)
        
        instance = cls(model_type=model_data['model_type'], scale_features=False)
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.feature_names = model_data['feature_names']
        instance.metrics = model_data['metrics']
        
        logger.info(f"Model loaded from {filepath}")
        return instance


class ModelComparison:
    """
    Compare multiple baseline models.
    
    From research: Start simple and only add complexity if justified.
    """
    
    def __init__(self):
        self.models = {}
        self.comparison_results = None
    
    def add_model(self, name: str, model: BaselineModel):
        """Add a model to comparison."""
        self.models[name] = model
    
    def train_all(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Train all models."""
        logger.info(f"Training {len(self.models)} models...")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
    
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
        self.comparison_results = self.comparison_results.sort_values('accuracy', ascending=False)
        
        logger.info("Model Comparison Results:")
        logger.info("\n" + str(self.comparison_results.to_string(index=False)))
        
        return self.comparison_results
    
    def get_best_model(self, metric: str = 'accuracy') -> Tuple[str, BaselineModel]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for comparison
        
        Returns:
            Tuple of (model_name, model)
        """
        if self.comparison_results is None:
            raise ValueError("Must run evaluate_all() first")
        
        best_name = self.comparison_results.iloc[0]['model']
        return best_name, self.models[best_name]
