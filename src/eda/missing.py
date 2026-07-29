from __future__ import annotations

from src.data.models import DataProfile
from src.eda.schema import MissingSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)


class MissingAnalyzer:
    """
    Analyze missing values using a DataProfile.

    This analyzer interprets missing value statistics computed by the
    DataProfiler without recomputing them.
    """

    def __init__(self, profile: DataProfile) -> None:
        self.profile = profile

    def run(self) -> MissingSummary:
        """
        Analyze missing values and return a summary.
        """
        logger.info("Analyzing missing values.")

        # Sort by missing count (highest first)
        missing_counts = dict(
            sorted(
                self.profile.missing_counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )

        # Compute missing percentages
        missing_percentages = {
            column: round(count / self.profile.rows * 100, 2)
            for column, count in missing_counts.items()
        }

        # Count columns that actually have missing values
        columns_with_missing = sum(count > 0 for count in missing_counts.values())

        summary = MissingSummary(
            total_columns=self.profile.columns,
            columns_with_missing=columns_with_missing,
            missing_counts=missing_counts,
            missing_percentages=missing_percentages,
        )

        logger.info(
            "Missing value analysis completed. %d columns contain missing values.",
            summary.columns_with_missing,
        )

        return summary
