from __future__ import annotations

import pandas as pd

from src.data.models import DataProfile
from src.eda.config import EDAConfig
from src.eda.schema import CategoricalSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CategoricalAnalyzer:
    """Analyze categorical features."""

    def __init__(
        self,
        df: pd.DataFrame,
        profile: DataProfile,
        config: EDAConfig = EDAConfig(),
    ):
        self.config = config
        self.df = df
        self.profile = profile

    def run(self) -> CategoricalSummary:
        logger.info("Analyzing categorical features.")

        categorical_df = self.df[self.profile.categorical_columns]

        cardinality = {}
        rare_categories = {}
        dominant_categories = {}

        for column in categorical_df.columns:
            value_counts = categorical_df[column].value_counts(
                normalize=True,
                dropna=False,
            )

            cardinality[column] = categorical_df[column].nunique(dropna=True)

            rare_categories[column] = (
                value_counts[value_counts < EDAConfig.RARE_CATEGORY_THRESHOLD]
                .index.astype(str)
                .tolist()
            )

            dominant_categories[column] = str(value_counts.idxmax())

        summary = CategoricalSummary(
            cardinality=cardinality,
            rare_categories=rare_categories,
            dominant_categories=dominant_categories,
        )

        logger.info("Categorical feature analysis completed.")

        return summary
