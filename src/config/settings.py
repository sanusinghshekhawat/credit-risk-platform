from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    sample_size: int = 100
    random_state: int = 42
    chunk_size: int = 100
    csv_engine: str = "c"
