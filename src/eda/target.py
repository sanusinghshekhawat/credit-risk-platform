from __future__ import annotations

import pandas as pd

from src.eda.schema import TargetSummary
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TargetAnalyzer:
    """
    Analyze the target variable of the dataset.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str = "loan_status",
    ) -> None:
        self.df = df
        self.target_column = target_column

    def _validate_target_column(self) -> None:
        if self.target_column not in self.df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")

    def run(self) -> TargetSummary:
        """
        Analyze the target column.
        """

        logger.info(
            "Analyzing target column '%s'.",
            self.target_column,
        )

        self._validate_target_column()

        target = self.df[self.target_column]

        class_counts = target.value_counts(dropna=False)

        class_percentages = (
            target.value_counts(normalize=True, dropna=False).mul(100).round(2)
        )

        summary = TargetSummary(
            target_column=self.target_column,
            total_samples=len(target),
            missing_values=int(target.isna().sum()),
            n_classes=int(target.nunique(dropna=True)),
            unique_labels=target.dropna().unique().tolist(),
            class_counts=class_counts,
            class_percentages=class_percentages,
        )

        logger.info(
            "Target analysis completed. Found %d classes.",
            summary.n_classes,
        )

        return summary
