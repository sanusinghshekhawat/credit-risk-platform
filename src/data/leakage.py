from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.config.paths import ProjectPaths
from src.data.models import (
    LeakageDecision,
    LeakageReport,
    LeakageRule,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LeakageAnalyzer:
    """
    Applies leakage rules to dataset columns.
    """

    def __init__(
        self,
        config_path: Path = ProjectPaths.LEAKAGE_RULES,
    ) -> None:
        self.config_path = config_path
        self._rules = self._load_rules()
        self._validate_rules()

    def _load_rules(self) -> dict:
        """
        Load leakage rules from YAML.
        """
        logger.info("Loading leakage rules from %s", self.config_path)

        with open(self.config_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    def _validate_rules(self) -> None:
        """
        Validate configuration file.
        """

        valid_decisions = {
            decision.value
            for decision in LeakageDecision
            if decision != LeakageDecision.UNKNOWN
        }

        for column, rule in self._rules.items():
            if "decision" not in rule:
                raise ValueError(f"Missing 'decision' for column '{column}'.")

            if "reason" not in rule:
                raise ValueError(f"Missing 'reason' for column '{column}'.")

            if rule["decision"] not in valid_decisions:
                raise ValueError(
                    f"Invalid decision '{rule['decision']}' for column '{column}'."
                )

    def rule_for(
        self,
        column: str,
    ) -> LeakageRule:
        """
        Return leakage rule for a single column.
        """

        if column not in self._rules:
            logger.warning(
                "No leakage rule configured for '%s'.",
                column,
            )

            return LeakageRule(
                column=column,
                decision=LeakageDecision.UNKNOWN,
                reason="No rule configured.",
                configured=False,
            )

        rule = self._rules[column]

        return LeakageRule(
            column=column,
            decision=LeakageDecision(rule["decision"]),
            reason=rule["reason"],
            configured=True,
        )

    def analyze(
        self,
        df: pd.DataFrame,
    ) -> LeakageReport:
        """
        Analyze dataframe columns.
        """

        rules = [self.rule_for(column) for column in df.columns]

        keep = []
        review = []
        drop = []
        leakage = []
        target = []
        unknown = []

        for rule in rules:
            if rule.decision == LeakageDecision.KEEP:
                keep.append(rule.column)

            elif rule.decision == LeakageDecision.REVIEW:
                review.append(rule.column)

            elif rule.decision == LeakageDecision.DROP:
                drop.append(rule.column)

            elif rule.decision == LeakageDecision.LEAKAGE:
                leakage.append(rule.column)

            elif rule.decision == LeakageDecision.TARGET:
                target.append(rule.column)

            else:
                unknown.append(rule.column)

        logger.info(
            "Leakage analysis completed. "
            "KEEP=%d REVIEW=%d DROP=%d "
            "LEAKAGE=%d TARGET=%d UNKNOWN=%d",
            len(keep),
            len(review),
            len(drop),
            len(leakage),
            len(target),
            len(unknown),
        )

        return LeakageReport(
            rules=rules,
            keep=keep,
            review=review,
            drop=drop,
            leakage=leakage,
            target=target,
            unknown=unknown,
        )

    @property
    def unknown_rules(self):
        return [rule for rule in self.rules if rule.decision == LeakageDecision.UNKNOWN]
