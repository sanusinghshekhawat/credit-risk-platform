from pathlib import Path
from typing import Iterator

import pandas as pd

from src.config.settings import DataConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetLoader:
    """
    Load datasets efficiently.
    """

    def __init__(self, config: DataConfig):
        self.config = config

    def load(
        self,
        file_path: Path,
        columns: list[str] | None = None,
        n_rows: int | None = None,
    ) -> pd.DataFrame:
        logger.info(f"Loading dataset: {file_path.name}")

        return pd.read_csv(
            file_path,
            usecols=columns,
            nrows=n_rows,
            low_memory=False,
            engine=self.config.csv_engine,
        )

    def load_chunks(
        self,
        file_path: Path,
        columns: list[str] | None = None,
    ) -> Iterator[pd.DataFrame]:
        logger.info(f"Loading dataset in chunks: {file_path.name}")

        return pd.read_csv(
            file_path,
            usecols=columns,
            chunksize=self.config.chunk_size,
            low_memory=False,
            engine=self.config.csv_engine,
        )
