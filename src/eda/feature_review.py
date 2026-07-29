from __future__ import annotations

from src.data.models import LeakageReport
from src.eda.schema import (
    CorrelationSummary,
    FeatureReviewSummary,
    MissingSummary,
    NumericSummary,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureReviewAnalyzer:
    """Combine EDA results into a feature review."""

    def __init__(
        self,
        missing: MissingSummary,
        numeric: NumericSummary,
        correlation: CorrelationSummary,
        leakage: LeakageReport,
    ) -> None:
        self.missing = missing
        self.numeric = numeric
        self.correlation = correlation
        self.leakage = leakage

    def run(self) -> FeatureReviewSummary:
        logger.info("Reviewing features.")

        summary = FeatureReviewSummary(
            constant_columns=self.numeric.constant_columns,
            high_missing_columns=[],
            highly_correlated_columns=[],
            leakage_columns=self.leakage.leakage,
            review_columns=self.leakage.review,
            recommended_drop_columns=[],
            recommended_keep_columns=self.leakage.keep,
        )

        logger.info("Feature review completed.")

        return summary
