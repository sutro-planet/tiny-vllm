from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

from tiny_vllm.sequence import SequenceState
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
                len(sequence.generated_token_ids)
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
        self._torch: Any | None = None
        eval_fn = getattr(self.model, "eval", None)
        if callable(eval_fn):
            eval_fn()

    def encode_prompt(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
        context_rows = [
            self._context_token_ids(
                sequence,
                num_scheduled_tokens=num_scheduled_tokens,
            )
            for sequence, num_scheduled_tokens in zip(
                batch.sequences, batch.num_scheduled_tokens, strict=True
            )
        ]
        input_rows, attention_rows = self._pad_context_rows(context_rows)
        input_ids = self._model_input_ids(input_rows)
        attention_mask = self._model_attention_mask(attention_rows)
        output = self._forward(input_ids=input_ids, attention_mask=attention_mask)
        logits = output.logits
        return ModelRunnerOutput(
            sampled_token_ids=[
                self._sample_next_token_id(
                    logits,
                    row_index=row_index,
                    token_index=sum(attention_row) - 1,
                )
                for row_index, attention_row in enumerate(attention_rows)
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

    @staticmethod
    def _context_token_ids(
        sequence: SequenceState, *, num_scheduled_tokens: int
    ) -> list[int]:
        return sequence.scheduled_context_token_ids(
            num_scheduled_tokens=num_scheduled_tokens
        )

    def _pad_context_rows(
        self, context_rows: list[list[int]]
    ) -> tuple[list[list[int]], list[list[int]]]:
        if not context_rows:
            return [], []
        if any(not row for row in context_rows):
            raise ValueError("cannot execute an empty token context")

        max_length = max(len(row) for row in context_rows)
        pad_token_id = self.tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        for row in context_rows:
            padding = max_length - len(row)
            input_rows.append([*row, *([pad_token_id] * padding)])
            attention_rows.append([*([1] * len(row)), *([0] * padding)])
        return input_rows, attention_rows

    def _model_input_ids(self, input_rows: list[list[int]]) -> Any:
        if not self.use_torch_inputs:
            return input_rows

        return self._get_torch().tensor(input_rows, device=self.device)

    def _model_attention_mask(self, attention_rows: list[list[int]]) -> Any:
        if not self.use_torch_inputs:
            return attention_rows

        return self._get_torch().tensor(attention_rows, device=self.device)

    def _forward(self, *, input_ids: Any, attention_mask: Any) -> Any:
        if not self.use_torch_inputs:
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

        with self._get_torch().no_grad():
            return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def _get_torch(self) -> Any:
        if self._torch is not None:
            return self._torch

        try:
            self._torch = import_module("torch")
        except ImportError as exc:
            raise RuntimeError("Install torch to run a transformers model") from exc
        return self._torch

    @staticmethod
    def _sample_next_token_id(
        logits: Any,
        *,
        row_index: int,
        token_index: int,
    ) -> int:
        try:
            token_logits = logits[row_index, token_index]
        except TypeError:
            token_logits = logits[row_index][token_index]

        argmax_fn = getattr(token_logits, "argmax", None)
        if callable(argmax_fn):
            token_id = argmax_fn(dim=-1)
            item_fn = getattr(token_id, "item", None)
            if callable(item_fn):
                return int(item_fn())
            return int(token_id)

        scores = cast(list[float], token_logits)
        return max(range(len(scores)), key=scores.__getitem__)
