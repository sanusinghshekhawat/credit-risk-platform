from __future__ import annotations

import pandas as pd

from src.data.models import DataProfile
from src.eda.config import EDAConfig
from src.eda.schema import OutlierSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OutlierAnalyzer:
    """Detect outliers using the IQR method."""

    def __init__(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        config: EDAConfig,
    ) -> None:
        self.df = df
        self.profile = profile
        self.config = config

    def run(self) -> OutlierSummary:
        logger.info("Analyzing outliers.")

        numeric_df = self.df[self.profile.numerical_columns]

        outlier_counts = {}
        outlier_percentages = {}

        for column in numeric_df.columns:
            q1 = numeric_df[column].quantile(0.25)
            q3 = numeric_df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - EDAConfig.IQR_MULTIPLIER * iqr
            upper = q3 + EDAConfig.IQR_MULTIPLIER * iqr

            mask = (numeric_df[column] < lower) | (numeric_df[column] > upper)

            count = int(mask.sum())

            outlier_counts[column] = count
            outlier_percentages[column] = round(
                count / len(numeric_df) * 100,
                2,
            )

        summary = OutlierSummary(
            iqr_outlier_counts=outlier_counts,
            outlier_percentages=outlier_percentages,
        )

        logger.info("Outlier analysis completed.")

        return summary
