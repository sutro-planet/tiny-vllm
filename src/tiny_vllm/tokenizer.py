from typing import Protocol


class TokenizerBackend(Protocol):
    @property
    def eos_token_id(self) -> int | None: ...

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


class Tokenizer:
    def __init__(self, backend: TokenizerBackend) -> None:
        self.backend = backend

    @property
    def eos_token_id(self) -> int | None:
        return self.backend.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self.backend.encode(text, add_special_tokens=False)

    def decode(self, token_ids: list[int]) -> str:
        return self.backend.decode(token_ids, skip_special_tokens=True)
