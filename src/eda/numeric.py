from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.models import DataProfile
from src.eda.config import EDAConfig
from src.eda.schema import NumericSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)


class NumericAnalyzer:
    """Analyze numerical features."""

    def __init__(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        config: EDAConfig,
    ) -> None:
        self.df = df
        self.profile = profile
        self.config = config

    def run(self) -> NumericSummary:
        logger.info("Analyzing numerical features.")

        numeric_df = self.df[self.profile.numerical_columns]

        constant_columns = [
            col
            for col in numeric_df.columns
            if numeric_df[col].nunique(dropna=False) <= 1
        ]

        near_constant_columns = [
            col
            for col in numeric_df.columns
            if numeric_df[col].value_counts(normalize=True, dropna=False).iloc[0]
            >= EDAConfig.NEAR_CONSTANT_THRESHOLD
            and col not in constant_columns
        ]

        infinite_value_columns = [
            col for col in numeric_df.columns if np.isinf(numeric_df[col]).any()
        ]

        skewness = numeric_df.skew(numeric_only=True).round(4).to_dict()

        kurtosis = numeric_df.kurt(numeric_only=True).round(4).to_dict()

        summary = NumericSummary(
            constant_columns=constant_columns,
            near_constant_columns=near_constant_columns,
            infinite_value_columns=infinite_value_columns,
            skewness=skewness,
            kurtosis=kurtosis,
        )

        logger.info("Numeric feature analysis completed.")

        return summary
