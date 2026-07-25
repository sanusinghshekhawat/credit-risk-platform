from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    sample_size: int = 10_000
    random_state: int = 42
    chunk_size: int = 100_000
    csv_engine: str = "c"
