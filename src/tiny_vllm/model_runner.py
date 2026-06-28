from importlib import import_module
from typing import Any, Protocol, cast

from tiny_vllm.request import GenerationOutput, GenerationRequest
from tiny_vllm.tokenizer import Tokenizer, TokenizerBackend


class ModelRunner(Protocol):
    def generate(self, request: GenerationRequest, max_new_tokens: int) -> GenerationOutput: ...


class MockModelRunner:
    def generate(self, request: GenerationRequest, max_new_tokens: int) -> GenerationOutput:
        generated = [f"<mock-{index}>" for index in range(max_new_tokens)]
        text = " ".join([request.prompt, *generated]) if request.prompt else " ".join(generated)
        return GenerationOutput(
            request_id=request.request_id,
            text=text,
            generated_tokens=max_new_tokens,
        )


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

    def generate(self, request: GenerationRequest, max_new_tokens: int) -> GenerationOutput:
        input_token_ids = self.tokenizer.encode(request.prompt)
        input_ids = self._model_input_ids(input_token_ids)
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": self._attention_mask(input_token_ids, input_ids),
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

        output_ids = self.model.generate(**generation_kwargs)
        output_token_ids = self._to_token_ids(output_ids)
        new_token_ids = output_token_ids[len(input_token_ids) :]

        return GenerationOutput(
            request_id=request.request_id,
            text=self.tokenizer.decode(new_token_ids),
            generated_tokens=len(new_token_ids),
        )

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
