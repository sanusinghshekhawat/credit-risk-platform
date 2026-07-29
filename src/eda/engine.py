from __future__ import annotations

import pandas as pd

from src.data.profiler import DataProfiler
from src.eda.categorical import CategoricalAnalyzer
from src.eda.config import EDAConfig
from src.eda.correlation import CorrelationAnalyzer
from src.eda.missing import MissingAnalyzer
from src.eda.numeric import NumericAnalyzer
from src.eda.outliers import OutlierAnalyzer
from src.eda.schema import EDAReport
from src.eda.target import TargetAnalyzer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EDAEngine:
    """Run the complete Exploratory Data Analysis pipeline."""

    def __init__(
        self,
        df: pd.DataFrame,
        target_column: str,
        config: EDAConfig | None = None,
    ) -> None:
        self.df = df
        self.target_column = target_column
        self.config = config or EDAConfig()

    def run(self) -> EDAReport:
        """Execute the complete EDA pipeline."""

        logger.info("Starting EDA pipeline.")

        profile = DataProfiler().profile(self.df)

        target = TargetAnalyzer(
            self.df,
            self.target_column,
        ).run()

        missing = MissingAnalyzer(
            profile,
        ).run()

        numeric = NumericAnalyzer(
            self.df,
            profile,
            self.config,
        ).run()

        categorical = CategoricalAnalyzer(
            self.df,
            profile,
            self.config,
        ).run()

        correlation = CorrelationAnalyzer(
            self.df,
            profile,
            self.config,
        ).run()

        outliers = OutlierAnalyzer(
            self.df,
            profile,
            self.config,
        ).run()

        report = EDAReport(
            target=target,
            missing=missing,
            numeric=numeric,
            categorical=categorical,
            correlation=correlation,
            outliers=outliers,
        )

        logger.info("EDA pipeline completed successfully.")

        return report
