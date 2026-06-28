from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

from tiny_vllm.sequence import SequencePhase, SequenceState
from tiny_vllm.tokenizer import Tokenizer, TokenizerBackend


@dataclass
class ExecutionBatch:
    sequences: list[SequenceState]
    num_scheduled_tokens: list[int]

    def __post_init__(self) -> None:
        if len(self.sequences) != len(self.num_scheduled_tokens):
            raise ValueError(
                "execution batch must provide one scheduled-token count per sequence"
            )

    @property
    def request_ids(self) -> list[str]:
        return [sequence.request_id for sequence in self.sequences]


@dataclass
class ModelRunnerOutput:
    sampled_token_ids: list[int]


class ModelRunner(Protocol):
    def encode_prompt(self, prompt: str) -> list[int]: ...

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput: ...

    def detokenize(self, token_ids: list[int]) -> str: ...


class MockModelRunner:
    def encode_prompt(self, prompt: str) -> list[int]:
        return list(range(max(1, len(prompt.split()))))

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
        return ModelRunnerOutput(
            sampled_token_ids=[
                0
                if sequence.phase is SequencePhase.WAITING_PREFILL
                else len(sequence.generated_token_ids)
                for sequence in batch.sequences
            ]
        )

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(f"<mock-{token_id}>" for token_id in token_ids)


class TransformersModelRunner:
    def __init__(
        self,
        model: Any,
        tokenizer: Tokenizer,
        *,
        device: str = "cpu",
        use_torch_inputs: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.use_torch_inputs = use_torch_inputs
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def encode_prompt(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
        return ModelRunnerOutput(
            sampled_token_ids=[
                self._generate_one(self._context_token_ids(sequence))
                for sequence in batch.sequences
            ]
        )

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device: str = "cpu",
        torch_dtype: str | None = None,
        use_safetensors: bool | None = None,
    ) -> "TransformersModelRunner":
        try:
            transformers = cast(Any, import_module("transformers"))
        except ImportError as exc:
            raise RuntimeError(
                "Install tiny-vllm[transformers] to use TransformersModelRunner"
            ) from exc

        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model_kwargs: dict[str, Any] = {}
        if torch_dtype is not None:
            try:
                torch = import_module("torch")
            except ImportError as exc:
                raise RuntimeError("Install torch to use torch_dtype") from exc
            model_kwargs["torch_dtype"] = getattr(torch, torch_dtype)
        if use_safetensors is not None:
            model_kwargs["use_safetensors"] = use_safetensors

        model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        to_fn = getattr(model, "to", None)
        if callable(to_fn):
            model = to_fn(device)

        return cls(
            model=model,
            tokenizer=Tokenizer(cast(TokenizerBackend, tokenizer)),
            device=device,
            use_torch_inputs=True,
        )

    def _generate_one(self, token_ids: list[int]) -> int:
        input_ids = self._model_input_ids(token_ids)
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": self._attention_mask(token_ids, input_ids),
            "max_new_tokens": 1,
            "do_sample": False,
        }
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        output_ids = self.model.generate(**generation_kwargs)
        output_token_ids = self._to_token_ids(output_ids)
        return output_token_ids[-1]

    @staticmethod
    def _context_token_ids(sequence: SequenceState) -> list[int]:
        if sequence.phase is SequencePhase.WAITING_PREFILL:
            return sequence.prompt_token_ids
        return sequence.all_token_ids

    def _model_input_ids(self, input_token_ids: list[int]) -> Any:
        if not self.use_torch_inputs:
            return [input_token_ids]

        try:
            torch = import_module("torch")
        except ImportError as exc:
            raise RuntimeError("Install torch to run a transformers model") from exc

        return torch.tensor([input_token_ids], device=self.device)

    def _attention_mask(self, input_token_ids: list[int], input_ids: Any) -> Any:
        if not self.use_torch_inputs:
            return [[1] * len(input_token_ids)]

        try:
            torch = import_module("torch")
        except ImportError as exc:
            raise RuntimeError("Install torch to run a transformers model") from exc

        return torch.ones_like(input_ids)

    @staticmethod
    def _to_token_ids(output_ids: Any) -> list[int]:
        tolist_fn = getattr(output_ids, "tolist", None)
        if callable(tolist_fn):
            rows = cast(list[list[int]], tolist_fn())
            return rows[0]

        rows = cast(list[list[int]], output_ids)
        return rows[0]
