from pathlib import Path


class ProjectPaths:
    """Centralized project paths."""

    ROOT = Path(__file__).resolve().parents[2]

    CONFIGS = ROOT / "configs"

    DATA = ROOT / "data"
    RAW = DATA / "raw"
    RAW_ARCHIVES = RAW / "archives"
    RAW_EXTRACTED = RAW / "extracted"

    INTERIM = DATA / "interim"
    EDA_DATASET = INTERIM / "eda_dataset.parquet"

    PROCESSED = DATA / "processed"
    TRAIN_PROCESSED = PROCESSED / "train.parquet"
    VALIDATION_PROCESSED = PROCESSED / "validation.parquet"
    TEST_PROCESSED = PROCESSED / "test.parquet"

    EXTERNAL = DATA / "external"

    MODELS = ROOT / "models"
    PREPROCESSING_MODELS = MODELS / "preprocessing"
    PREPROCESSOR = PREPROCESSING_MODELS / "preprocessor.joblib"

    NOTEBOOKS = ROOT / "notebooks"
    REPORTS = ROOT / "reports"
    DOCS = ROOT / "docs"

    ACCEPTED_LOANS = RAW_EXTRACTED / "accepted_2007_to_2018Q4.csv"
    REJECTED_LOANS = RAW_EXTRACTED / "rejected_2007_to_2018Q4.csv"

    LEAKAGE_RULES = CONFIGS / "leakage_rules.yaml"
