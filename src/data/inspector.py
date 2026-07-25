from pathlib import Path

import pandas as pd

from src.config.settings import DataConfig


class DatasetInspector:
    """
    Inspect a dataset without loading the entire file into memory.

    Parameters
    ----------
    file_path : Path
        Path to the dataset.
    config : DataConfig
        Project data configuration.
    """

    def __init__(self, file_path: Path, config: DataConfig):
        self.file_path = Path(file_path)
        self.config = config

    def exists(self) -> bool:
        """
        Check whether the dataset exists.
        """
        return self.file_path.exists()

    def file_size(self) -> float:
        """
        Return file size in GB.
        """
        self._validate_file()

        size_bytes = self.file_path.stat().st_size
        return round(size_bytes / (1024**3), 2)

    def preview(self, n_rows: int = 5) -> pd.DataFrame:
        """
        Return the first n rows.
        """
        return self._read_sample(n_rows)

    def columns(self) -> list[str]:
        """
        Return all column names.
        """
        return self._read_sample(0).columns.tolist()

    def dtypes(self) -> pd.Series:
        """
        Infer column dtypes from a sample.
        """
        return self._read_sample(self.config.sample_size).dtypes

    def summary(self) -> dict:
        """
        Return a dataset summary.
        """
        cols = self.columns()
        dtypes = self.dtypes()

        return {
            "file_name": self.file_path.name,
            "exists": self.exists(),
            "file_size_gb": self.file_size(),
            "num_columns": len(cols),
            "columns": cols,
            "dtypes": dtypes.astype(str).to_dict(),
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _validate_file(self) -> None:
        """
        Ensure the dataset exists.
        """
        if not self.exists():
            raise FileNotFoundError(f"Dataset not found:\n{self.file_path}")

    def _read_sample(self, n_rows: int) -> pd.DataFrame:
        """
        Read only a small portion of the dataset.

        Uses pandas' C engine because it supports
        nrows, chunksize, and skiprows.
        """
        self._validate_file()

        return pd.read_csv(
            self.file_path,
            nrows=n_rows,
            engine="c",
            low_memory=False,
        )
