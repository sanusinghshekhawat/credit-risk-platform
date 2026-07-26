from __future__ import annotations

import pandas as pd

from src.data.models import DataProfile
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataProfiler:
    """
    Profile a pandas DataFrame.
    """

    def profile(self, df: pd.DataFrame) -> DataProfile:
        """
        Generate a complete profile of the dataset.
        """
        logger.info("Generating dataset profile.")

        if df.empty:
            raise ValueError("Cannot profile an empty DataFrame.")

        profile = DataProfile(
            rows=self.rows(df),
            columns=self.columns(df),
            memory_mb=self.memory_usage(df),
            duplicate_rows=self.duplicate_rows(df),
            missing_values=self.missing_values(df),
            numerical_columns=self.numerical_columns(df),
            categorical_columns=self.categorical_columns(df),
            datetime_columns=self.datetime_columns(df),
            boolean_columns=self.boolean_columns(df),
            descriptive_statistics=self.descriptive_statistics(df),
        )

        logger.info("Dataset profiling completed.")

        return profile

    # --------------------------------------------------------
    # Shape
    # --------------------------------------------------------

    def rows(self, df: pd.DataFrame) -> int:
        return df.shape[0]

    def columns(self, df: pd.DataFrame) -> int:
        return df.shape[1]

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    def memory_usage(self, df: pd.DataFrame) -> float:
        memory = df.memory_usage(deep=True).sum()
        return round(memory / (1024**2), 2)

    # --------------------------------------------------------
    # Missing Values
    # --------------------------------------------------------

    def missing_values(self, df: pd.DataFrame) -> dict[str, int]:
        missing = df.isna().sum()

        return {column: int(count) for column, count in missing.items() if count > 0}

    # --------------------------------------------------------
    # Duplicate Rows
    # --------------------------------------------------------

    def duplicate_rows(self, df: pd.DataFrame) -> int:
        return int(df.duplicated().sum())

    # --------------------------------------------------------
    # Column Types
    # --------------------------------------------------------

    def numerical_columns(self, df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include="number").columns.tolist()

    def categorical_columns(self, df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include="object").columns.tolist()

    def datetime_columns(self, df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include="datetime").columns.tolist()

    def boolean_columns(self, df: pd.DataFrame) -> list[str]:
        return df.select_dtypes(include="bool").columns.tolist()

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def descriptive_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.describe(include="all").T
