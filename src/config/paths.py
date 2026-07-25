from pathlib import Path


class ProjectPaths:
    """Centralized project paths."""

    ROOT = Path(__file__).resolve().parents[2]

    DATA = ROOT / "data"

    RAW = DATA / "raw"
    RAW_ARCHIVES = RAW / "archives"
    RAW_EXTRACTED = RAW / "extracted"

    INTERIM = DATA / "interim"
    PROCESSED = DATA / "processed"
    EXTERNAL = DATA / "external"

    MODELS = ROOT / "models"
    NOTEBOOKS = ROOT / "notebooks"
    REPORTS = ROOT / "reports"
    DOCS = ROOT / "docs"

    ACCEPTED_LOANS = RAW_EXTRACTED / "accepted_2007_to_2018Q4.csv"
    REJECTED_LOANS = RAW_EXTRACTED / "rejected_2007_to_2018Q4.csv"
