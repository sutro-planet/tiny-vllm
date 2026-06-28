from dataclasses import dataclass, field
from enum import Enum


class SequencePhase(Enum):
    WAITING_PREFILL = "waiting_prefill"
    DECODING = "decoding"
    FINISHED = "finished"


@dataclass
class SequenceState:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    generated_token_ids: list[int] = field(default_factory=list)
    phase: SequencePhase = SequencePhase.WAITING_PREFILL

    @property
    def token_budget(self) -> int:
        return max(1, len(self.prompt_token_ids) + self.max_new_tokens)

    @property
    def all_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.generated_token_ids]

    @property
    def latest_token_id(self) -> int:
        if self.generated_token_ids:
            return self.generated_token_ids[-1]
        if not self.prompt_token_ids:
            raise ValueError("sequence has no tokens")
        return self.prompt_token_ids[-1]

    @property
    def is_complete(self) -> bool:
        return self.phase is SequencePhase.FINISHED

    def append_prefill_token(self, token_id: int) -> None:
        if self.phase is not SequencePhase.WAITING_PREFILL:
            raise RuntimeError(f"sequence is not waiting for prefill: {self.request_id}")
        self.generated_token_ids.append(token_id)
        self._advance_after_append()

    def append_decode_token(self, token_id: int) -> None:
        if self.phase is not SequencePhase.DECODING:
            raise RuntimeError(f"sequence is not decoding: {self.request_id}")
        self.generated_token_ids.append(token_id)
        self._advance_after_append()

    def _advance_after_append(self) -> None:
        if len(self.generated_token_ids) >= self.max_new_tokens:
            self.phase = SequencePhase.FINISHED
        else:
            self.phase = SequencePhase.DECODING
