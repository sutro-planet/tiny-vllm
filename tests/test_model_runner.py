from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import tiny_vllm.model_runner as model_runner_module
from tiny_vllm.model_runner import TransformersModelRunner
from tiny_vllm.request import GenerationRequest
from tiny_vllm.tokenizer import Tokenizer


class FakeTensor:
    def __init__(self, data: list[list[int]]) -> None:
        self.data = data

    def to(self, device: str) -> FakeTensor:
        assert device == "cpu"
        return self

    def tolist(self) -> list[list[int]]:
        return self.data


class FakeHFTokenizer:
    eos_token_id: int | None = 99

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, FakeTensor]:
        assert return_tensors == "pt"
        return {"input_ids": FakeTensor([[len(part) for part in prompt.split()]])}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [len(part) for part in text.split()]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return " ".join(f"tok-{token_id}" for token_id in token_ids)


class FakeModel:
    def __init__(self, output_ids: list[int]) -> None:
        self.device = "cpu"
        self.output_ids = output_ids
        self.eval_called = False
        self.generate_kwargs: dict[str, Any] | None = None

    def eval(self) -> None:
        self.eval_called = True

    def generate(self, **kwargs: Any) -> FakeTensor:
        self.generate_kwargs = kwargs
        return FakeTensor([self.output_ids])


def test_transformers_runner_generates_and_decodes_only_new_tokens() -> None:
    hf_tokenizer = FakeHFTokenizer()
    model = FakeModel(output_ids=[5, 4, 11, 12])
    runner = TransformersModelRunner(model=model, tokenizer=Tokenizer(hf_tokenizer))

    output = runner.generate(
        GenerationRequest(request_id="req-a", prompt="hello tiny", max_new_tokens=2),
        max_new_tokens=2,
    )

    assert model.eval_called is True
    assert model.generate_kwargs == {
        "input_ids": [[5, 4]],
        "attention_mask": [[1, 1]],
        "max_new_tokens": 2,
        "do_sample": False,
        "pad_token_id": 99,
    }
    assert output.request_id == "req-a"
    assert output.generated_tokens == 2
    assert output.text == "tok-11 tok-12"


def test_transformers_runner_does_not_decode_prompt_tokens_when_generation_stops_early() -> None:
    hf_tokenizer = FakeHFTokenizer()
    model = FakeModel(output_ids=[5, 4, 11])
    runner = TransformersModelRunner(model=model, tokenizer=Tokenizer(hf_tokenizer))

    output = runner.generate(
        GenerationRequest(request_id="req-a", prompt="hello tiny", max_new_tokens=2),
        max_new_tokens=2,
    )

    assert output.generated_tokens == 1
    assert output.text == "tok-11"


def test_transformers_runner_from_pretrained_forwards_safetensors_option(
    monkeypatch: Any,
) -> None:
    model_calls: list[dict[str, Any]] = []

    class FakeLoadedModel:
        def to(self, device: str) -> FakeLoadedModel:
            assert device == "cpu"
            return self

        def eval(self) -> None:
            pass

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str) -> FakeHFTokenizer:
            assert model_name == "gpt2"
            return FakeHFTokenizer()

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: Any) -> FakeLoadedModel:
            assert model_name == "gpt2"
            model_calls.append(kwargs)
            return FakeLoadedModel()

    def fake_import_module(name: str) -> Any:
        if name == "transformers":
            return SimpleNamespace(
                AutoTokenizer=FakeAutoTokenizer,
                AutoModelForCausalLM=FakeAutoModel,
            )
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(model_runner_module, "import_module", fake_import_module)

    TransformersModelRunner.from_pretrained("gpt2", use_safetensors=False)

    assert model_calls == [{"use_safetensors": False}]
