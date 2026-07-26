from __future__ import annotations

import pandas as pd

from src.data.models import DataDictionaryEntry
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataDictionary:
    """
    Generate structural metadata for dataset columns.
    """

    def generate(self, df: pd.DataFrame) -> list[DataDictionaryEntry]:
        """
        Generate metadata for every column.
        """
        logger.info("Generating data dictionary.")

        entries: list[DataDictionaryEntry] = []

        total_rows = len(df)

        for column in df.columns:
            missing = int(df[column].isna().sum())

            entries.append(
                DataDictionaryEntry(
                    column=column,
                    dtype=str(df[column].dtype),
                    missing_count=missing,
                    missing_percentage=round(
                        (missing / total_rows) * 100,
                        2,
                    ),
                    unique_count=int(df[column].nunique(dropna=True)),
                )
            )

        logger.info("Generated %d dictionary entries.", len(entries))

        return entries
