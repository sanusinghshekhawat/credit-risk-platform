from dataclasses import dataclass


@dataclass(frozen=True)
class EDAConfig:
    # Missing Values
    HIGH_MISSING_THRESHOLD: float = 0.50

    # Numerical Features
    NEAR_CONSTANT_THRESHOLD: float = 0.99

    # Correlation
    HIGH_CORRELATION_THRESHOLD: float = 0.90

    # Categorical Features
    RARE_CATEGORY_THRESHOLD: float = 0.01

    # Outliers
    IQR_MULTIPLIER: float = 1.5
