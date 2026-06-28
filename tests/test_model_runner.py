from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import tiny_vllm.model_runner as model_runner_module
from tiny_vllm.model_runner import ExecutionBatch, MockModelRunner, TransformersModelRunner
from tiny_vllm.sequence import SequenceState
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


def test_mock_runner_execute_returns_one_token_per_scheduled_sequence() -> None:
    runner = MockModelRunner()
    first = SequenceState(
        request_id="req-a",
        prompt="alpha",
        prompt_token_ids=runner.encode_prompt("alpha"),
        max_new_tokens=2,
    )
    second = SequenceState(
        request_id="req-b",
        prompt="beta gamma",
        prompt_token_ids=runner.encode_prompt("beta gamma"),
        max_new_tokens=1,
    )

    prefill_output = runner.execute(
        ExecutionBatch(
            sequences=[first, second],
            num_scheduled_tokens=[len(first.prompt_token_ids), len(second.prompt_token_ids)],
        )
    )

    assert prefill_output.sampled_token_ids == [0, 0]

    first.append_prefill_token(0)
    second.append_prefill_token(0)

    decode_output = runner.execute(
        ExecutionBatch(sequences=[first], num_scheduled_tokens=[1])
    )

    assert decode_output.sampled_token_ids == [1]
    assert runner.detokenize([0, 1]) == "<mock-0> <mock-1>"


def test_transformers_runner_execute_prefill_returns_first_generated_token() -> None:
    hf_tokenizer = FakeHFTokenizer()
    model = FakeModel(output_ids=[5, 4, 11])
    runner = TransformersModelRunner(model=model, tokenizer=Tokenizer(hf_tokenizer))
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello tiny",
        prompt_token_ids=[5, 4],
        max_new_tokens=2,
    )

    output = runner.execute(
        ExecutionBatch(
            sequences=[sequence],
            num_scheduled_tokens=[len(sequence.prompt_token_ids)],
        )
    )

    assert output.sampled_token_ids == [11]
    assert model.generate_kwargs == {
        "input_ids": [[5, 4]],
        "attention_mask": [[1, 1]],
        "max_new_tokens": 1,
        "do_sample": False,
        "pad_token_id": 99,
    }


def test_transformers_runner_execute_decode_returns_next_generated_token() -> None:
    hf_tokenizer = FakeHFTokenizer()
    model = FakeModel(output_ids=[5, 4, 11, 12])
    runner = TransformersModelRunner(model=model, tokenizer=Tokenizer(hf_tokenizer))
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello tiny",
        prompt_token_ids=[5, 4],
        max_new_tokens=2,
    )
    sequence.append_prefill_token(11)

    output = runner.execute(ExecutionBatch(sequences=[sequence], num_scheduled_tokens=[1]))

    assert output.sampled_token_ids == [12]
    assert runner.detokenize([11, 12]) == "tok-11 tok-12"


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
