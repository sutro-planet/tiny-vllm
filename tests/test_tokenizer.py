from tiny_vllm.tokenizer import Tokenizer


class FakeHFTokenizer:
    eos_token_id: int | None = 7

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [len(part) for part in text.split()]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return "|".join(str(token_id) for token_id in token_ids)


def test_tokenizer_wraps_encode_decode_and_eos_id() -> None:
    tokenizer = Tokenizer(FakeHFTokenizer())

    assert tokenizer.encode("aa bbb") == [2, 3]
    assert tokenizer.decode([2, 3, 7]) == "2|3|7"
    assert tokenizer.eos_token_id == 7
