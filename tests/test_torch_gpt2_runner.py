from importlib.util import find_spec
from typing import Any, cast
from unittest import SkipTest

if find_spec("torch") is None:
    raise SkipTest("torch is required for TorchGPT2ModelRunner tests")

import torch

import tiny_vllm.torch_gpt2 as torch_gpt2_module
from tiny_vllm.model_runner import ExecutionBatch
from tiny_vllm.sequence import SequenceState
from tiny_vllm.tokenizer import Tokenizer
from tiny_vllm.torch_gpt2 import GPT2Config, TinyGPT2LMHeadModel, TorchGPT2ModelRunner


class FakeHFTokenizer:
    eos_token_id: int | None = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [int(part) for part in text.split()]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        assert skip_special_tokens is True
        return " ".join(str(token_id) for token_id in token_ids)


class FakeHFConfig:
    model_type = "gpt2"
    vocab_size = 11
    n_positions = 8
    n_embd = 8
    n_layer = 1
    n_head = 2
    n_inner = None
    layer_norm_epsilon = 1e-5
    activation_function = "gelu_new"


class CountingTinyGPT2LMHeadModel(TinyGPT2LMHeadModel):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__(config)
        self.forward_calls = 0
        self.forward_flat_calls = 0
        self.last_flat_input_ids: torch.Tensor | None = None
        self.last_flat_positions: torch.Tensor | None = None
        self.last_flat_req_indices: torch.Tensor | None = None
        self.last_flat_query_start_loc: torch.Tensor | None = None

    def forward(self, **kwargs: Any) -> Any:
        self.forward_calls += 1
        return super().forward(**kwargs)

    def forward_flat(self, **kwargs: Any) -> Any:
        self.forward_flat_calls += 1
        self.last_flat_input_ids = kwargs["input_ids"].detach().cpu()
        self.last_flat_positions = kwargs["positions"].detach().cpu()
        self.last_flat_req_indices = kwargs["req_indices"].detach().cpu()
        self.last_flat_query_start_loc = kwargs["query_start_loc"].detach().cpu()
        return super().forward_flat(**kwargs)


def test_torch_gpt2_runner_extends_request_cache_by_scheduled_tokens() -> None:
    torch.manual_seed(0)
    runner = TorchGPT2ModelRunner(
        TinyGPT2LMHeadModel(
            GPT2Config(
                vocab_size=17,
                max_position_embeddings=8,
                hidden_size=12,
                num_layers=2,
                num_heads=3,
            )
        ),
        Tokenizer(FakeHFTokenizer()),
    )
    sequence = SequenceState(
        request_id="req-a",
        prompt="1 2",
        prompt_token_ids=[1, 2],
        max_new_tokens=2,
    )

    prefill_output = runner.execute(
        ExecutionBatch(sequences=[sequence], num_scheduled_tokens=[2])
    )
    sequence.advance_computed_tokens(2)
    sequence.append_sampled_token(prefill_output.sampled_token_ids[0])

    assert runner.cache_token_count("req-a") == 2

    decode_output = runner.execute(
        ExecutionBatch(sequences=[sequence], num_scheduled_tokens=[1])
    )

    assert len(decode_output.sampled_token_ids) == 1
    assert runner.cache_token_count("req-a") == 3


def test_torch_gpt2_runner_executes_mixed_batch_with_one_flat_forward() -> None:
    torch.manual_seed(4)
    model = CountingTinyGPT2LMHeadModel(
        GPT2Config(
            vocab_size=17,
            max_position_embeddings=8,
            hidden_size=12,
            num_layers=1,
            num_heads=3,
        )
    )
    runner = TorchGPT2ModelRunner(
        model,
        Tokenizer(FakeHFTokenizer()),
        block_size=2,
    )
    decode_sequence = SequenceState(
        request_id="decode",
        prompt="1 2",
        prompt_token_ids=[1, 2],
        max_new_tokens=2,
    )
    prefill_output = runner.execute(
        ExecutionBatch(sequences=[decode_sequence], num_scheduled_tokens=[2])
    )
    decode_sequence.advance_computed_tokens(2)
    decode_sequence.append_sampled_token(prefill_output.sampled_token_ids[0])
    prefill_sequence = SequenceState(
        request_id="prefill",
        prompt="3 4 5",
        prompt_token_ids=[3, 4, 5],
        max_new_tokens=1,
    )

    model.forward_calls = 0
    model.forward_flat_calls = 0
    output = runner.execute(
        ExecutionBatch(
            sequences=[decode_sequence, prefill_sequence],
            num_scheduled_tokens=[1, 3],
        )
    )

    assert len(output.sampled_token_ids) == 2
    assert model.forward_calls == 0
    assert model.forward_flat_calls == 1
    assert model.last_flat_input_ids is not None
    assert model.last_flat_positions is not None
    assert model.last_flat_req_indices is not None
    assert model.last_flat_query_start_loc is not None
    assert model.last_flat_input_ids.tolist() == [
        decode_sequence.generated_token_ids[0],
        3,
        4,
        5,
    ]
    assert model.last_flat_positions.tolist() == [2, 0, 1, 2]
    assert model.last_flat_req_indices.tolist() == [0, 1, 1, 1]
    assert model.last_flat_query_start_loc.tolist() == [0, 1, 4]
    assert runner.cache_token_count("decode") == 3
    assert runner.cache_token_count("prefill") == 3
    assert set(runner._req_to_blocks) == {"decode", "prefill"}


def test_torch_gpt2_runner_release_clears_request_cache() -> None:
    torch.manual_seed(1)
    runner = TorchGPT2ModelRunner(
        TinyGPT2LMHeadModel(
            GPT2Config(
                vocab_size=17,
                max_position_embeddings=8,
                hidden_size=12,
                num_layers=1,
                num_heads=3,
            )
        ),
        Tokenizer(FakeHFTokenizer()),
    )
    sequence = SequenceState(
        request_id="req-a",
        prompt="1 2",
        prompt_token_ids=[1, 2],
        max_new_tokens=1,
    )

    runner.execute(ExecutionBatch(sequences=[sequence], num_scheduled_tokens=[2]))
    runner.release("req-a")

    assert runner.cache_token_count("req-a") is None


def test_torch_gpt2_runner_rejects_stale_cache_after_recompute_reset() -> None:
    torch.manual_seed(2)
    runner = TorchGPT2ModelRunner(
        TinyGPT2LMHeadModel(
            GPT2Config(
                vocab_size=17,
                max_position_embeddings=8,
                hidden_size=12,
                num_layers=1,
                num_heads=3,
            )
        ),
        Tokenizer(FakeHFTokenizer()),
    )
    sequence = SequenceState(
        request_id="req-a",
        prompt="1",
        prompt_token_ids=[1],
        max_new_tokens=2,
    )

    output = runner.execute(ExecutionBatch(sequences=[sequence], num_scheduled_tokens=[1]))
    sequence.advance_computed_tokens(1)
    sequence.append_sampled_token(output.sampled_token_ids[0])
    sequence.reset_computed_tokens()

    try:
        runner.execute(ExecutionBatch(sequences=[sequence], num_scheduled_tokens=[2]))
    except RuntimeError as exc:
        assert "expected 0" in str(exc)
    else:
        raise AssertionError("expected stale KV cache to be rejected")


def test_torch_gpt2_runner_rejects_missing_cache_for_nonzero_start() -> None:
    torch.manual_seed(3)
    runner = TorchGPT2ModelRunner(
        TinyGPT2LMHeadModel(
            GPT2Config(
                vocab_size=17,
                max_position_embeddings=8,
                hidden_size=12,
                num_layers=1,
                num_heads=3,
            )
        ),
        Tokenizer(FakeHFTokenizer()),
    )
    sequence = SequenceState(
        request_id="req-a",
        prompt="1 2",
        prompt_token_ids=[1, 2],
        max_new_tokens=1,
        num_computed_tokens=1,
    )

    try:
        runner.execute(ExecutionBatch(sequences=[sequence], num_scheduled_tokens=[1]))
    except RuntimeError as exc:
        assert "missing KV cache" in str(exc)
    else:
        raise AssertionError("expected missing KV cache to be rejected")


def test_torch_gpt2_runner_from_pretrained_accepts_torch_dtype_object(monkeypatch: Any) -> None:
    model_calls: list[dict[str, object]] = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name: str) -> FakeHFTokenizer:
            assert model_name == "gpt2"
            return FakeHFTokenizer()

    class FakeAutoConfig:
        @staticmethod
        def from_pretrained(model_name: str) -> FakeHFConfig:
            assert model_name == "gpt2"
            return FakeHFConfig()

    class FakeLoadedModel:
        def __init__(self) -> None:
            self.source = TinyGPT2LMHeadModel(GPT2Config.from_hf_config(FakeHFConfig()))

        def state_dict(self) -> dict[str, torch.Tensor]:
            return cast(dict[str, torch.Tensor], self.source.state_dict())

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name: str, **kwargs: object) -> FakeLoadedModel:
            assert model_name == "gpt2"
            model_calls.append(kwargs)
            return FakeLoadedModel()

    class FakeTransformers:
        AutoTokenizer = FakeAutoTokenizer
        AutoConfig = FakeAutoConfig
        AutoModelForCausalLM = FakeAutoModel

    def fake_import_module(name: str) -> object:
        if name == "transformers":
            return FakeTransformers
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(torch_gpt2_module, "import_module", fake_import_module)

    runner = TorchGPT2ModelRunner.from_pretrained("gpt2", torch_dtype=torch.float16)

    assert model_calls == [{"torch_dtype": torch.float16}]
    assert next(runner.model.parameters()).dtype == torch.float16
