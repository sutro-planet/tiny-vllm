from dataclasses import dataclass


@dataclass(frozen=True)
class EngineConfig:
    model_name: str = "mock-gemma"
    max_batch_size: int = 4
    max_num_blocks: int = 16
    block_size: int = 16
    max_new_tokens: int = 8

    def __post_init__(self) -> None:
        for field_name in ("max_batch_size", "max_num_blocks", "block_size", "max_new_tokens"):
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
