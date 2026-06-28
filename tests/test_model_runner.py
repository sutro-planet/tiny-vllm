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


class FakeForwardOutput:
    def __init__(self, logits: list[list[list[float]]]) -> None:
        self.logits = logits


class FakeForwardModel:
    def __init__(self, sampled_token_ids: list[int]) -> None:
        self.device = "cpu"
        self.sampled_token_ids = sampled_token_ids
        self.eval_called = False
        self.forward_kwargs: dict[str, Any] | None = None
        self.forward_calls = 0

    def eval(self) -> None:
        self.eval_called = True

    def __call__(self, **kwargs: Any) -> FakeForwardOutput:
        self.forward_calls += 1
        self.forward_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        vocab_size = max(self.sampled_token_ids) + 1
        logits: list[list[list[float]]] = []
        for row_index, row in enumerate(input_ids):
            row_logits: list[list[float]] = []
            for _ in row:
                scores = [0.0] * vocab_size
                scores[self.sampled_token_ids[row_index]] = 1.0
                row_logits.append(scores)
            logits.append(row_logits)
        return FakeForwardOutput(logits=logits)

    def generate(self, **kwargs: Any) -> FakeTensor:
        raise AssertionError("execute should call forward, not generate")


class FakeNoGrad:
    def __enter__(self) -> None:
        return None

    def __exit__(self, *args: Any) -> None:
        return None


class FakeTorch:
    def __init__(self) -> None:
        self.tensor_calls = 0
        self.no_grad_calls = 0

    def tensor(self, rows: list[list[int]], device: str) -> list[list[int]]:
        assert device == "cpu"
        self.tensor_calls += 1
        return rows

    def no_grad(self) -> FakeNoGrad:
        self.no_grad_calls += 1
        return FakeNoGrad()


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

    first.advance_computed_tokens(len(first.prompt_token_ids))
    first.append_sampled_token(0)
    second.advance_computed_tokens(len(second.prompt_token_ids))
    second.append_sampled_token(0)

    decode_output = runner.execute(
        ExecutionBatch(sequences=[first], num_scheduled_tokens=[1])
    )

    assert decode_output.sampled_token_ids == [1]
    assert runner.detokenize([0, 1]) == "<mock-0> <mock-1>"


def test_transformers_runner_execute_prefill_returns_first_generated_token() -> None:
    hf_tokenizer = FakeHFTokenizer()
    model = FakeForwardModel(sampled_token_ids=[11])
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
    assert model.forward_calls == 1
    assert model.forward_kwargs == {
        "input_ids": [[5, 4]],
        "attention_mask": [[1, 1]],
    }


def test_transformers_runner_execute_decode_returns_next_generated_token() -> None:
    hf_tokenizer = FakeHFTokenizer()
    model = FakeForwardModel(sampled_token_ids=[12])
    runner = TransformersModelRunner(model=model, tokenizer=Tokenizer(hf_tokenizer))
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello tiny",
        prompt_token_ids=[5, 4],
        max_new_tokens=2,
        generated_token_ids=[11],
        num_computed_tokens=2,
    )

    output = runner.execute(ExecutionBatch(sequences=[sequence], num_scheduled_tokens=[1]))

    assert output.sampled_token_ids == [12]
    assert runner.detokenize([11, 12]) == "tok-11 tok-12"


def test_transformers_runner_execute_batches_mixed_decode_and_prefill_rows() -> None:
    hf_tokenizer = FakeHFTokenizer()
    model = FakeForwardModel(sampled_token_ids=[12, 13])
    runner = TransformersModelRunner(model=model, tokenizer=Tokenizer(hf_tokenizer))
    decode_sequence = SequenceState(
        request_id="decode",
        prompt="hello tiny",
        prompt_token_ids=[5, 4],
        max_new_tokens=2,
        generated_token_ids=[11],
        num_computed_tokens=2,
    )
    prefill_sequence = SequenceState(
        request_id="prefill",
        prompt="new request",
        prompt_token_ids=[7, 8],
        max_new_tokens=1,
    )

    output = runner.execute(
        ExecutionBatch(
            sequences=[decode_sequence, prefill_sequence],
            num_scheduled_tokens=[1, len(prefill_sequence.prompt_token_ids)],
        )
    )

    assert output.sampled_token_ids == [12, 13]
    assert model.forward_calls == 1
    assert model.forward_kwargs == {
        "input_ids": [[5, 4, 11], [7, 8, 99]],
        "attention_mask": [[1, 1, 1], [1, 1, 0]],
    }


def test_transformers_runner_execute_uses_scheduled_token_range_prefix() -> None:
    hf_tokenizer = FakeHFTokenizer()
    model = FakeForwardModel(sampled_token_ids=[12])
    runner = TransformersModelRunner(model=model, tokenizer=Tokenizer(hf_tokenizer))
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello tiny runner",
        prompt_token_ids=[5, 4, 6],
        max_new_tokens=1,
        num_computed_tokens=1,
    )

    output = runner.execute(ExecutionBatch(sequences=[sequence], num_scheduled_tokens=[1]))

    assert output.sampled_token_ids == [12]
    assert model.forward_kwargs == {
        "input_ids": [[5, 4]],
        "attention_mask": [[1, 1]],
    }


def test_transformers_runner_caches_torch_import(monkeypatch: Any) -> None:
    hf_tokenizer = FakeHFTokenizer()
    model = FakeForwardModel(sampled_token_ids=[11])
    fake_torch = FakeTorch()
    import_calls: list[str] = []

    def fake_import_module(name: str) -> Any:
        import_calls.append(name)
        if name == "torch":
            return fake_torch
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(model_runner_module, "import_module", fake_import_module)
    runner = TransformersModelRunner(
        model=model,
        tokenizer=Tokenizer(hf_tokenizer),
        use_torch_inputs=True,
    )
    sequence = SequenceState(
        request_id="req-a",
        prompt="hello tiny",
        prompt_token_ids=[5, 4],
        max_new_tokens=2,
    )
    batch = ExecutionBatch(
        sequences=[sequence],
        num_scheduled_tokens=[len(sequence.prompt_token_ids)],
    )

    first_output = runner.execute(batch)
    second_output = runner.execute(batch)

    assert first_output.sampled_token_ids == [11]
    assert second_output.sampled_token_ids == [11]
    assert import_calls == ["torch"]
    assert fake_torch.tensor_calls == 4
    assert fake_torch.no_grad_calls == 2


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
