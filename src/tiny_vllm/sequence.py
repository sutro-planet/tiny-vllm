from dataclasses import dataclass, field


@dataclass
class SequenceState:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    generated_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    @property
    def token_budget(self) -> int:
        return max(1, len(self.prompt_token_ids) + self.max_new_tokens)

    @property
    def all_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.generated_token_ids]

    @property
    def num_tokens(self) -> int:
        return len(self.all_token_ids)

    @property
    def num_new_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def latest_token_id(self) -> int:
        if self.generated_token_ids:
            return self.generated_token_ids[-1]
        if not self.prompt_token_ids:
            raise ValueError("sequence has no tokens")
        return self.prompt_token_ids[-1]

    @property
    def is_complete(self) -> bool:
        return len(self.generated_token_ids) >= self.max_new_tokens

    def should_sample_after(self, *, num_scheduled_tokens: int) -> bool:
        if num_scheduled_tokens <= 0:
            raise ValueError("num_scheduled_tokens must be positive")
        return self.num_computed_tokens + num_scheduled_tokens >= self.num_tokens

    def scheduled_context_token_ids(self, *, num_scheduled_tokens: int) -> list[int]:
        if num_scheduled_tokens <= 0:
            raise ValueError("num_scheduled_tokens must be positive")

        end = self.num_computed_tokens + num_scheduled_tokens
        if end > self.num_tokens:
            raise ValueError("cannot schedule beyond known tokens")
        return self.all_token_ids[:end]

    def advance_computed_tokens(self, num_tokens: int) -> None:
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if self.num_computed_tokens + num_tokens > self.num_tokens:
            raise ValueError("cannot compute beyond known tokens")
        self.num_computed_tokens += num_tokens

    def append_sampled_token(self, token_id: int) -> None:
        if self.num_computed_tokens < self.num_tokens:
            raise RuntimeError(
                "cannot sample before computed tokens reach current tokens: "
                f"{self.request_id}"
            )
        if self.is_complete:
            raise RuntimeError(f"sequence is already complete: {self.request_id}")
        self.generated_token_ids.append(token_id)

    def reset_computed_tokens(self) -> None:
        self.num_computed_tokens = 0
