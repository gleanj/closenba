"""
Data Validation and Quality Checks

Ensures data quality before model training to improve accuracy.
Handles missing values, outliers, and data inconsistencies.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates data quality and consistency.

    Poor data quality is a major cause of poor model accuracy.
    This class catches issues early.
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.validation_results = {}

    def validate_dataset(self, df: pd.DataFrame, dataset_name: str = 'dataset') -> Dict:
        """
        Run all validation checks on a dataset.

        Args:
            df: DataFrame to validate
            dataset_name: Name for logging

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating {dataset_name} ({len(df)} rows, {len(df.columns)} columns)")

        results = {
            'dataset_name': dataset_name,
            'n_rows': len(df),
            'n_cols': len(df.columns),
            'missing_values': self._check_missing_values(df),
            'duplicates': self._check_duplicates(df),
            'outliers': self._check_outliers(df),
            'data_types': self._check_data_types(df),
            'value_ranges': self._check_value_ranges(df)
        }

        self.validation_results[dataset_name] = results

        # Log summary
        self._log_validation_summary(results)

        return results

    def _check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values in each column."""
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100

        missing_info = {
            'total_missing': int(missing.sum()),
            'columns_with_missing': list(missing[missing > 0].index),
            'missing_percentages': {
                col: f"{pct:.2f}%"
                for col, pct in missing_pct[missing_pct > 0].items()
            }
        }

        if missing_info['total_missing'] > 0:
            logger.warning(f"Found {missing_info['total_missing']} missing values across "
                         f"{len(missing_info['columns_with_missing'])} columns")

        return missing_info

    def _check_duplicates(self, df: pd.DataFrame) -> Dict:
        """Check for duplicate rows."""
        n_duplicates = df.duplicated().sum()

        duplicate_info = {
            'n_duplicates': int(n_duplicates),
            'duplicate_pct': f"{(n_duplicates / len(df)) * 100:.2f}%"
        }

        if n_duplicates > 0:
            logger.warning(f"Found {n_duplicates} duplicate rows ({duplicate_info['duplicate_pct']})")

        return duplicate_info

    def _check_outliers(self, df: pd.DataFrame) -> Dict:
        """
        Check for outliers using IQR method.

        Outliers can significantly impact model training.
        """
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outlier_info = {}

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]

            if len(outliers) > 0:
                outlier_info[col] = {
                    'n_outliers': len(outliers),
                    'outlier_pct': f"{(len(outliers) / len(df)) * 100:.2f}%",
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound)
                }

        if outlier_info:
            logger.info(f"Found outliers in {len(outlier_info)} columns")

        return outlier_info

    def _check_data_types(self, df: pd.DataFrame) -> Dict:
        """Check data types of columns."""
        dtype_counts = df.dtypes.value_counts().to_dict()
        return {str(k): int(v) for k, v in dtype_counts.items()}

    def _check_value_ranges(self, df: pd.DataFrame) -> Dict:
        """Check value ranges for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        ranges = {}

        for col in numeric_cols:
            ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }

        return ranges

    def _log_validation_summary(self, results: Dict):
        """Log a summary of validation results."""
        logger.info(f"Validation Summary for {results['dataset_name']}:")
        logger.info(f"  Rows: {results['n_rows']}, Columns: {results['n_cols']}")
        logger.info(f"  Missing values: {results['missing_values']['total_missing']}")
        logger.info(f"  Duplicates: {results['duplicates']['n_duplicates']}")
        logger.info(f"  Columns with outliers: {len(results['outliers'])}")


class DataPreprocessor:
    """
    Preprocesses data to improve quality and model accuracy.

    Handles:
    - Missing value imputation
    - Outlier treatment
    - Feature scaling
    - Encoding categorical variables
    """

    def __init__(self, handle_missing: str = 'mean', handle_outliers: str = 'clip'):
        """
        Args:
            handle_missing: Method for missing values ('mean', 'median', 'drop', 'ffill')
            handle_outliers: Method for outliers ('clip', 'remove', 'none')
        """
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers

        # Store imputation values
        self.imputation_values = {}
        self.outlier_bounds = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessing parameters and transform data.

        Args:
            df: DataFrame to preprocess

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df, fit=True)

        # Handle outliers
        df = self._handle_outliers(df, fit=True)

        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted parameters.

        Args:
            df: DataFrame to preprocess

        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()

        # Handle missing values
        df = self._handle_missing_values(df, fit=False)

        # Handle outliers
        df = self._handle_outliers(df, fit=False)

        return df

    def _handle_missing_values(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle missing values using specified strategy."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                if fit:
                    if self.handle_missing == 'mean':
                        self.imputation_values[col] = df[col].mean()
                    elif self.handle_missing == 'median':
                        self.imputation_values[col] = df[col].median()
                    elif self.handle_missing == 'drop':
                        continue  # Will drop rows later
                    elif self.handle_missing == 'ffill':
                        continue  # Forward fill doesn't need fitting

                # Apply imputation
                if self.handle_missing in ['mean', 'median']:
                    df[col].fillna(self.imputation_values[col], inplace=True)
                elif self.handle_missing == 'ffill':
                    df[col].ffill(inplace=True)
                    df[col].fillna(0, inplace=True)  # Fill remaining with 0

        # Drop rows with missing values if specified
        if self.handle_missing == 'drop':
            df.dropna(inplace=True)

        return df

    def _handle_outliers(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle outliers using specified strategy."""
        if self.handle_outliers == 'none':
            return df

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if fit:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                self.outlier_bounds[col] = {
                    'lower': lower_bound,
                    'upper': upper_bound
                }

            # Apply outlier handling
            if self.handle_outliers == 'clip':
                df[col] = df[col].clip(
                    lower=self.outlier_bounds[col]['lower'],
                    upper=self.outlier_bounds[col]['upper']
                )
            elif self.handle_outliers == 'remove':
                # Mark outliers for removal
                mask = (
                    (df[col] >= self.outlier_bounds[col]['lower']) &
                    (df[col] <= self.outlier_bounds[col]['upper'])
                )
                df = df[mask]

        return df


class FeatureSelector:
    """
    Selects most important features to improve model accuracy and reduce overfitting.

    From research: Feature selection can improve accuracy by removing noise.
    """

    def __init__(self, method: str = 'correlation', threshold: float = 0.05):
        """
        Args:
            method: Selection method ('correlation', 'variance', 'importance')
            threshold: Threshold for selection
        """
        self.method = method
        self.threshold = threshold
        self.selected_features = None

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_importances: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Select features and transform data.

        Args:
            X: Feature DataFrame
            y: Target variable
            feature_importances: Optional feature importance scores from model

        Returns:
            DataFrame with selected features
        """
        if self.method == 'correlation':
            self.selected_features = self._select_by_correlation(X, y)
        elif self.method == 'variance':
            self.selected_features = self._select_by_variance(X)
        elif self.method == 'importance' and feature_importances is not None:
            self.selected_features = self._select_by_importance(feature_importances)
        else:
            logger.warning(f"Unknown selection method: {self.method}. Using all features.")
            self.selected_features = X.columns.tolist()

        logger.info(f"Selected {len(self.selected_features)} features out of {len(X.columns)}")

        return X[self.selected_features]

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using selected features."""
        if self.selected_features is None:
            raise ValueError("Must call fit_transform first")

        return X[self.selected_features]

    def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features based on correlation with target."""
        correlations = X.corrwith(y).abs()
        selected = correlations[correlations > self.threshold].index.tolist()

        logger.info(f"Correlation-based selection: {len(selected)} features with |r| > {self.threshold}")

        return selected

    def _select_by_variance(self, X: pd.DataFrame) -> List[str]:
        """Select features with variance above threshold."""
        variances = X.var()
        selected = variances[variances > self.threshold].index.tolist()

        logger.info(f"Variance-based selection: {len(selected)} features with var > {self.threshold}")

        return selected

    def _select_by_importance(self, feature_importances: pd.DataFrame) -> List[str]:
        """Select features based on model importance scores."""
        selected = feature_importances[
            feature_importances['importance'] > self.threshold
        ]['feature'].tolist()

        logger.info(f"Importance-based selection: {len(selected)} features with importance > {self.threshold}")

        return selected
