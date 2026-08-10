"""
Provide reusable preprocessing transformers for the credit-risk pipeline.

This module contains generic transformations for numerical missingness,
event-history missingness, structural missingness, categorical missingness,
and datetime normalization.

Dataset-specific feature decisions are maintained separately in registry.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MissingIndicatorImputer(BaseEstimator, TransformerMixin):
    """Impute numerical values while preserving missingness indicators."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "MissingIndicatorImputer":
        self.feature_names_in_ = list(X.columns)
        self.medians_ = X.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        result = {}

        for feature in self.feature_names_in_:
            result[feature] = X[feature].fillna(self.medians_[feature])
            result[f"{feature}__missing"] = X[feature].isna().astype(int)

        return pd.DataFrame(result, index=X.index)

    def get_feature_names_out(
        self,
        input_features: np.ndarray | None = None,
    ) -> np.ndarray:
        features = (
            self.feature_names_in_ if input_features is None else list(input_features)
        )

        return np.asarray(
            [name for feature in features for name in (feature, f"{feature}__missing")],
            dtype=object,
        )


class EventHistoryImputer(BaseEstimator, TransformerMixin):
    """Handle missing event-history values using learned sentinel values."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "EventHistoryImputer":
        self.feature_names_in_ = list(X.columns)

        self.sentinels_ = {
            feature: X[feature].max(skipna=True) + 1
            for feature in self.feature_names_in_
        }

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        result = {}

        for feature in self.feature_names_in_:
            result[feature] = X[feature].fillna(self.sentinels_[feature])
            result[f"{feature}__missing"] = X[feature].isna().astype(int)

        return pd.DataFrame(result, index=X.index)

    def get_feature_names_out(
        self,
        input_features: np.ndarray | None = None,
    ) -> np.ndarray:
        features = (
            self.feature_names_in_ if input_features is None else list(input_features)
        )

        return np.asarray(
            [name for feature in features for name in (feature, f"{feature}__missing")],
            dtype=object,
        )


class StructuralMissingnessImputer(BaseEstimator, TransformerMixin):
    """Handle feature groups exhibiting structural joint missingness."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "StructuralMissingnessImputer":
        self.feature_names_in_ = list(X.columns)
        self.medians_ = X.median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        result = {}

        block_missing = X[self.feature_names_in_].isna().all(axis=1)

        for feature in self.feature_names_in_:
            result[feature] = X[feature].fillna(self.medians_[feature])
            result[f"{feature}__missing"] = X[feature].isna().astype(int)

        result["structural_block__missing"] = block_missing.astype(int)

        return pd.DataFrame(result, index=X.index)

    def get_feature_names_out(
        self,
        input_features: np.ndarray | None = None,
    ) -> np.ndarray:
        features = (
            self.feature_names_in_ if input_features is None else list(input_features)
        )

        return np.asarray(
            [name for feature in features for name in (feature, f"{feature}__missing")]
            + ["structural_block__missing"],
            dtype=object,
        )


class CategoricalMissingHandler(BaseEstimator, TransformerMixin):
    """Replace missing categorical values with an explicit category."""

    MISSING_VALUE = "__MISSING__"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "CategoricalMissingHandler":
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.feature_names_in_:
            X[feature] = X[feature].astype("object").fillna(self.MISSING_VALUE)

        return X

    def get_feature_names_out(
        self,
        input_features: np.ndarray | None = None,
    ) -> np.ndarray:
        features = (
            self.feature_names_in_ if input_features is None else list(input_features)
        )

        return np.asarray(features, dtype=object)


class DatetimeTransformer(BaseEstimator, TransformerMixin):
    """Normalize configured date columns into pandas datetime values."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "DatetimeTransformer":
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for feature in self.feature_names_in_:
            X[feature] = pd.to_datetime(
                X[feature],
                format="%b-%Y",
                errors="coerce",
            )

        return X

    def get_feature_names_out(
        self,
        input_features: np.ndarray | None = None,
    ) -> np.ndarray:
        features = (
            self.feature_names_in_ if input_features is None else list(input_features)
        )

        return np.asarray(features, dtype=object)
