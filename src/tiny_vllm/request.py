from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if self.max_new_tokens is not None and self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")


@dataclass(frozen=True)
class GenerationOutput:
    request_id: str
    text: str
    generated_tokens: int
    error: str | None = None
