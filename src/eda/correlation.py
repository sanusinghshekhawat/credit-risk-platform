from __future__ import annotations

import pandas as pd

from src.data.models import DataProfile
from src.eda.config import EDAConfig
from src.eda.schema import CorrelationSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CorrelationAnalyzer:
    """Analyze correlations between numerical features."""

    def __init__(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        config: EDAConfig,
    ) -> None:
        self.df = df
        self.profile = profile
        self.config = config

    def run(self) -> CorrelationSummary:
        logger.info("Analyzing feature correlations.")

        numeric_df = self.df[self.profile.numerical_columns]

        corr = numeric_df.corr()

        high_correlation_pairs = []
        perfectly_correlated_pairs = []

        columns = corr.columns

        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                value = corr.iloc[i, j]

                if abs(value) >= EDAConfig.RARE_CATEGORY_THRESHOLD:
                    high_correlation_pairs.append(
                        (
                            columns[i],
                            columns[j],
                            round(float(value), 4),
                        )
                    )

                if abs(value) == 1.0:
                    perfectly_correlated_pairs.append(
                        (
                            columns[i],
                            columns[j],
                        )
                    )

        summary = CorrelationSummary(
            high_correlation_pairs=high_correlation_pairs,
            perfectly_correlated_pairs=perfectly_correlated_pairs,
        )

        logger.info(
            "Correlation analysis completed. Found %d highly correlated pairs.",
            len(summary.high_correlation_pairs),
        )

        return summary
