from __future__ import annotations

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TargetBuilder:
    """
    Build the observed-outcome population and binary default target.

    Only loans with finalized outcomes are retained.

    Default:
        - Charged Off
        - Default
        - Does not meet the credit policy. Status:Charged Off

    Non-default:
        - Fully Paid
        - Does not meet the credit policy. Status:Fully Paid

    Ongoing outcomes are excluded because their eventual outcome
    has not yet been observed.
    """

    DEFAULT_STATUSES = (
        "Charged Off",
        "Default",
        "Does not meet the credit policy. Status:Charged Off",
    )

    NON_DEFAULT_STATUSES = (
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
    )

    OBSERVED_STATUSES = DEFAULT_STATUSES + NON_DEFAULT_STATUSES

    def __init__(
        self,
        target_column: str = "loan_status",
        output_column: str = "default",
    ) -> None:
        self.target_column = target_column
        self.output_column = output_column

    def _validate_target_column(self, df: pd.DataFrame) -> None:
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found.")

    def build(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the observed-outcome dataset with a binary default target.
        """

        logger.info("Building observed-outcome target.")

        self._validate_target_column(df)

        observed_df = df[df[self.target_column].isin(self.OBSERVED_STATUSES)].copy()

        observed_df[self.output_column] = (
            observed_df[self.target_column].isin(self.DEFAULT_STATUSES).astype(int)
        )

        logger.info(
            "Target construction completed. Retained %d of %d rows.",
            len(observed_df),
            len(df),
        )

        return observed_df
