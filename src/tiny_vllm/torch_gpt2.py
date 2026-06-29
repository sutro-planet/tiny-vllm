from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import torch
from torch import nn

from tiny_vllm.model_runner import ExecutionBatch, ModelRunnerOutput
from tiny_vllm.tokenizer import Tokenizer, TokenizerBackend

PastKeyValues = tuple[tuple[torch.Tensor, torch.Tensor], ...]
LayerKVCache = tuple[torch.Tensor, torch.Tensor]
LayerKVCaches = list[LayerKVCache]
GELU_NEW_COEFFICIENT = 0.7978845608028654


@dataclass(frozen=True)
class GPT2Config:
    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    num_layers: int
    num_heads: int
    intermediate_size: int | None = None
    layer_norm_epsilon: float = 1e-5
    activation_function: str = "gelu_new"

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        return self.hidden_size // self.num_heads

    @property
    def resolved_intermediate_size(self) -> int:
        return self.intermediate_size or 4 * self.hidden_size

    @classmethod
    def from_hf_config(cls, config: Any) -> GPT2Config:
        return cls(
            vocab_size=int(config.vocab_size),
            max_position_embeddings=int(config.n_positions),
            hidden_size=int(config.n_embd),
            num_layers=int(config.n_layer),
            num_heads=int(config.n_head),
            intermediate_size=(
                int(config.n_inner) if getattr(config, "n_inner", None) is not None else None
            ),
            layer_norm_epsilon=float(config.layer_norm_epsilon),
            activation_function=str(getattr(config, "activation_function", "gelu_new")),
        )


@dataclass
class GPT2ForwardOutput:
    logits: torch.Tensor
    past_key_values: PastKeyValues


@dataclass
class GPT2FlatForwardOutput:
    logits: torch.Tensor


class GPT2Conv1D(nn.Module):
    def __init__(self, out_features: int, in_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        output_shape = (*hidden_states.shape[:-1], self.bias.shape[0])
        hidden_states = torch.addmm(
            self.bias,
            hidden_states.reshape(-1, hidden_states.shape[-1]),
            self.weight,
        )
        return hidden_states.reshape(output_shape)


def _gelu_new(hidden_states: torch.Tensor) -> torch.Tensor:
    return (
        0.5
        * hidden_states
        * (
            1.0
            + torch.tanh(
                GELU_NEW_COEFFICIENT
                * (hidden_states + 0.044715 * torch.pow(hidden_states, 3.0))
            )
        )
    )


def _activation(name: str) -> nn.Module:
    if name == "gelu_new":
        return _FunctionalModule(_gelu_new)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"unsupported GPT-2 activation: {name}")


class _FunctionalModule(nn.Module):
    def __init__(self, fn: Any) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.fn(hidden_states))


class GPT2Attention(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.scale = config.head_dim**0.5
        self.c_attn = GPT2Conv1D(3 * config.hidden_size, config.hidden_size)
        self.c_proj = GPT2Conv1D(config.hidden_size, config.hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        query, key, value = self.c_attn(hidden_states).split(
            self.config.hidden_size,
            dim=2,
        )
        query = self._split_heads(query)
        key = self._split_heads(key)
        value = self._split_heads(value)

        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        attn_output = self._attend(query, key, value, position_ids=position_ids)
        attn_output = self._merge_heads(attn_output)
        return self.c_proj(attn_output), (key, value)

    def forward_flat(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor,
        req_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        kv_cache: LayerKVCache,
        block_size: int,
    ) -> torch.Tensor:
        query, key, value = self.c_attn(hidden_states).split(
            self.config.hidden_size,
            dim=-1,
        )
        query = self._split_heads_flat(query)
        key = self._split_heads_flat(key)
        value = self._split_heads_flat(value)

        key_cache, value_cache = kv_cache
        slot_block_ids = block_tables[req_indices, positions // block_size]
        slot_offsets = positions % block_size
        key_cache[slot_block_ids, :, slot_offsets, :] = key
        value_cache[slot_block_ids, :, slot_offsets, :] = value

        key_rows = self._gather_block_rows(
            key_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
        )
        value_rows = self._gather_block_rows(
            value_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            block_size=block_size,
        )
        attn_output = self._attend_flat(
            query,
            key_rows,
            value_rows,
            positions=positions,
            req_indices=req_indices,
            seq_lens=seq_lens,
        )
        return self.c_proj(self._merge_heads_flat(attn_output))

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        tensor = tensor.reshape(
            batch_size,
            sequence_length,
            self.config.num_heads,
            self.config.head_dim,
        )
        return tensor.permute(0, 2, 1, 3)

    def _split_heads_flat(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(
            tensor.shape[0],
            self.config.num_heads,
            self.config.head_dim,
        )

    @staticmethod
    def _merge_heads(tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        return tensor.view(tensor.shape[0], tensor.shape[1], -1)

    @staticmethod
    def _merge_heads_flat(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.shape[0], -1)

    def _attend(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / self.scale
        key_positions = torch.arange(key.shape[-2], device=query.device)
        causal_mask = key_positions.view(1, 1, 1, -1) <= position_ids.view(
            position_ids.shape[0],
            1,
            position_ids.shape[1],
            1,
        )
        attn_weights = attn_weights.masked_fill(
            ~causal_mask,
            torch.finfo(attn_weights.dtype).min,
        )
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, value)

    def _gather_block_rows(
        self,
        cache: torch.Tensor,
        *,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        max_seq_len = int(seq_lens.max().item())
        token_positions = torch.arange(max_seq_len, device=cache.device)
        block_offsets = token_positions % block_size
        rows: list[torch.Tensor] = []
        for row_index in range(seq_lens.shape[0]):
            block_ids = block_tables[row_index, token_positions // block_size]
            row = cache[block_ids, :, block_offsets, :].permute(1, 0, 2)
            rows.append(row)
        return torch.stack(rows)

    def _attend_flat(
        self,
        query: torch.Tensor,
        key_rows: torch.Tensor,
        value_rows: torch.Tensor,
        *,
        positions: torch.Tensor,
        req_indices: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> torch.Tensor:
        key = key_rows[req_indices]
        value = value_rows[req_indices]
        attn_weights = torch.einsum("thd,thsd->ths", query, key) / self.scale
        key_positions = torch.arange(key.shape[-2], device=query.device)
        visible_tokens = key_positions.view(1, 1, -1) <= positions.view(-1, 1, 1)
        valid_tokens = key_positions.view(1, 1, -1) < seq_lens[req_indices].view(
            -1,
            1,
            1,
        )
        attn_weights = attn_weights.masked_fill(
            ~(visible_tokens & valid_tokens),
            torch.finfo(attn_weights.dtype).min,
        )
        attn_weights = torch.softmax(attn_weights, dim=-1)
        return torch.einsum("ths,thsd->thd", attn_weights, value)



class GPT2MLP(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        intermediate_size = config.resolved_intermediate_size
        self.c_fc = GPT2Conv1D(intermediate_size, config.hidden_size)
        self.c_proj = GPT2Conv1D(config.hidden_size, intermediate_size)
        self.act = _activation(config.activation_function)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        activated = cast(torch.Tensor, self.act(self.c_fc(hidden_states)))
        return cast(torch.Tensor, self.c_proj(activated))


class GPT2Block(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        layer_past: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden_states
        attn_output, present = self.attn(
            self.ln_1(hidden_states),
            position_ids=position_ids,
            layer_past=layer_past,
        )
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = residual + self.mlp(self.ln_2(hidden_states))
        return hidden_states, present

    def forward_flat(
        self,
        hidden_states: torch.Tensor,
        *,
        positions: torch.Tensor,
        req_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        kv_cache: LayerKVCache,
        block_size: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = residual + self.attn.forward_flat(
            self.ln_1(hidden_states),
            positions=positions,
            req_indices=req_indices,
            seq_lens=seq_lens,
            block_tables=block_tables,
            kv_cache=kv_cache,
            block_size=block_size,
        )

        residual = hidden_states
        return residual + self.mlp(self.ln_2(hidden_states))


class GPT2Transformer(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList([GPT2Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: PastKeyValues | None,
    ) -> tuple[torch.Tensor, PastKeyValues]:
        hidden_states = self.wte(input_ids) + self.wpe(position_ids)
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index, block in enumerate(self.h):
            layer_past = (
                None if past_key_values is None else past_key_values[layer_index]
            )
            hidden_states, present = block(
                hidden_states,
                position_ids=position_ids,
                layer_past=layer_past,
            )
            presents.append(present)

        return self.ln_f(hidden_states), tuple(presents)

    def forward_flat(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        req_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        kv_caches: LayerKVCaches,
        block_size: int,
    ) -> torch.Tensor:
        hidden_states = self.wte(input_ids) + self.wpe(positions)
        for block, kv_cache in zip(self.h, kv_caches, strict=True):
            hidden_states = block.forward_flat(
                hidden_states,
                positions=positions,
                req_indices=req_indices,
                seq_lens=seq_lens,
                block_tables=block_tables,
                kv_cache=kv_cache,
                block_size=block_size,
            )

        return self.ln_f(hidden_states)


class TinyGPT2LMHeadModel(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()
        self.config = config
        self.transformer = GPT2Transformer(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.weight = self.transformer.wte.weight

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        past_key_values: PastKeyValues | None = None,
        use_cache: bool = True,
    ) -> GPT2ForwardOutput:
        if not use_cache:
            raise ValueError("TinyGPT2LMHeadModel currently requires use_cache=True")
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")

        if position_ids is None:
            past_length = 0
            if past_key_values is not None and past_key_values:
                past_length = past_key_values[0][0].shape[-2]
            position_ids = torch.arange(
                past_length,
                past_length + input_ids.shape[1],
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0)
            position_ids = position_ids.expand(input_ids.shape[0], -1)

        hidden_states, present = self.transformer(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
        )
        return GPT2ForwardOutput(logits=self.lm_head(hidden_states), past_key_values=present)

    def forward_flat(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        req_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_lens: torch.Tensor,
        block_tables: torch.Tensor,
        kv_caches: LayerKVCaches,
        block_size: int,
    ) -> GPT2FlatForwardOutput:
        if input_ids.ndim != 1:
            raise ValueError("flat input_ids must have shape [total_num_scheduled_tokens]")
        if query_start_loc.ndim != 1:
            raise ValueError("query_start_loc must have shape [num_reqs + 1]")
        if int(query_start_loc[-1].item()) != input_ids.shape[0]:
            raise ValueError("query_start_loc does not match flat input length")

        hidden_states = self.transformer.forward_flat(
            input_ids=input_ids,
            positions=positions,
            req_indices=req_indices,
            seq_lens=seq_lens,
            block_tables=block_tables,
            kv_caches=kv_caches,
            block_size=block_size,
        )
        return GPT2FlatForwardOutput(logits=self.lm_head(hidden_states))

    def load_hf_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            raise RuntimeError(
                "incompatible GPT-2 state_dict: "
                f"missing={incompatible.missing_keys}"
            )


class TorchGPT2ModelRunner:
    def __init__(
        self,
        model: TinyGPT2LMHeadModel,
        tokenizer: Tokenizer,
        *,
        device: str = "cpu",
        block_size: int = 16,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = block_size
        self._kv_caches: LayerKVCaches | None = None
        self._next_block_id = 0
        self._free_block_ids: list[int] = []
        self._req_to_blocks: dict[str, list[int]] = {}
        self._req_to_token_count: dict[str, int] = {}

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        *,
        device: str = "cpu",
        torch_dtype: str | torch.dtype | None = None,
        use_safetensors: bool | None = None,
        block_size: int = 16,
    ) -> TorchGPT2ModelRunner:
        try:
            transformers = cast(Any, import_module("transformers"))
        except ImportError as exc:
            raise RuntimeError(
                "Install tiny-vllm[transformers] to use TorchGPT2ModelRunner"
            ) from exc

        tokenizer_backend = transformers.AutoTokenizer.from_pretrained(model_name)
        hf_config = transformers.AutoConfig.from_pretrained(model_name)
        if getattr(hf_config, "model_type", None) != "gpt2":
            raise ValueError(
                f"TorchGPT2ModelRunner only supports GPT-2 models, got {hf_config.model_type}"
            )

        model_kwargs: dict[str, Any] = {}
        resolved_torch_dtype = cls._resolve_torch_dtype(torch_dtype)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_torch_dtype
        if use_safetensors is not None:
            model_kwargs["use_safetensors"] = use_safetensors

        hf_model = transformers.AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )
        model = TinyGPT2LMHeadModel(GPT2Config.from_hf_config(hf_config))
        model.load_hf_state_dict(hf_model.state_dict())
        del hf_model

        if resolved_torch_dtype is not None:
            model = model.to(dtype=resolved_torch_dtype)

        return cls(
            model=model,
            tokenizer=Tokenizer(cast(TokenizerBackend, tokenizer_backend)),
            device=device,
            block_size=block_size,
        )

    @staticmethod
    def _resolve_torch_dtype(torch_dtype: str | torch.dtype | None) -> torch.dtype | None:
        if torch_dtype is None:
            return None
        if isinstance(torch_dtype, str):
            return cast(torch.dtype, getattr(torch, torch_dtype))
        return torch_dtype

    def encode_prompt(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
        if not batch.sequences:
            return ModelRunnerOutput(sampled_token_ids=[])

        with torch.no_grad():
            flat_input_ids: list[int] = []
            flat_positions: list[int] = []
            flat_req_indices: list[int] = []
            query_start_loc: list[int] = [0]
            seq_lens: list[int] = []

            for sequence, num_scheduled_tokens in zip(
                batch.sequences,
                batch.num_scheduled_tokens,
                strict=True,
            ):
                row_index = len(seq_lens)
                start = sequence.num_computed_tokens
                end = start + num_scheduled_tokens
                input_token_ids = sequence.all_token_ids[start:end]
                if not input_token_ids:
                    raise ValueError(f"cannot execute empty token range: {sequence.request_id}")

                self._validate_cache_length(
                    sequence.request_id,
                    expected_tokens=start,
                )
                self._ensure_request_blocks(sequence.request_id, num_tokens=end)
                flat_input_ids.extend(input_token_ids)
                flat_positions.extend(range(start, end))
                flat_req_indices.extend([row_index] * num_scheduled_tokens)
                query_start_loc.append(query_start_loc[-1] + num_scheduled_tokens)
                seq_lens.append(end)

            kv_caches = self._ensure_kv_caches()
            block_tables = self._build_block_tables(batch.request_ids)
            output = self.model.forward_flat(
                input_ids=torch.tensor(flat_input_ids, device=self.device, dtype=torch.long),
                positions=torch.tensor(flat_positions, device=self.device, dtype=torch.long),
                req_indices=torch.tensor(flat_req_indices, device=self.device, dtype=torch.long),
                query_start_loc=torch.tensor(
                    query_start_loc,
                    device=self.device,
                    dtype=torch.long,
                ),
                seq_lens=torch.tensor(seq_lens, device=self.device, dtype=torch.long),
                block_tables=block_tables,
                kv_caches=kv_caches,
                block_size=self.block_size,
            )
            for sequence, seq_len in zip(batch.sequences, seq_lens, strict=True):
                self._req_to_token_count[sequence.request_id] = seq_len

            sampled_token_ids = [
                int(output.logits[token_index - 1].argmax(dim=-1).item())
                for token_index in query_start_loc[1:]
            ]

        return ModelRunnerOutput(sampled_token_ids=sampled_token_ids)

    def detokenize(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def release(self, request_id: str) -> None:
        block_ids = self._req_to_blocks.pop(request_id, [])
        self._free_block_ids.extend(block_ids)
        self._req_to_token_count.pop(request_id, None)

    def cache_token_count(self, request_id: str) -> int | None:
        return self._req_to_token_count.get(request_id)

    def _ensure_kv_caches(self) -> LayerKVCaches:
        if self._kv_caches is None:
            self._kv_caches = []
            dtype = next(self.model.parameters()).dtype
            for _ in range(self.model.config.num_layers):
                self._kv_caches.append(
                    (
                        torch.empty(
                            0,
                            self.model.config.num_heads,
                            self.block_size,
                            self.model.config.head_dim,
                            device=self.device,
                            dtype=dtype,
                        ),
                        torch.empty(
                            0,
                            self.model.config.num_heads,
                            self.block_size,
                            self.model.config.head_dim,
                            device=self.device,
                            dtype=dtype,
                        ),
                    )
                )
        return self._kv_caches

    def _ensure_request_blocks(self, request_id: str, *, num_tokens: int) -> None:
        required_blocks = (num_tokens + self.block_size - 1) // self.block_size
        blocks = self._req_to_blocks.setdefault(request_id, [])
        while len(blocks) < required_blocks:
            blocks.append(self._allocate_block())

    def _allocate_block(self) -> int:
        if self._free_block_ids:
            block_id = self._free_block_ids.pop()
            self._zero_block(block_id)
            return block_id

        block_id = self._next_block_id
        self._next_block_id += 1
        self._append_block()
        return block_id

    def _append_block(self) -> None:
        kv_caches = self._ensure_kv_caches()
        for layer_index, (key_cache, value_cache) in enumerate(kv_caches):
            key_block = key_cache.new_zeros(
                1,
                self.model.config.num_heads,
                self.block_size,
                self.model.config.head_dim,
            )
            value_block = value_cache.new_zeros(
                1,
                self.model.config.num_heads,
                self.block_size,
                self.model.config.head_dim,
            )
            kv_caches[layer_index] = (
                torch.cat((key_cache, key_block), dim=0),
                torch.cat((value_cache, value_block), dim=0),
            )

    def _zero_block(self, block_id: int) -> None:
        kv_caches = self._ensure_kv_caches()
        for key_cache, value_cache in kv_caches:
            key_cache[block_id].zero_()
            value_cache[block_id].zero_()

    def _build_block_tables(self, request_ids: list[str]) -> torch.Tensor:
        max_blocks = max(len(self._req_to_blocks[request_id]) for request_id in request_ids)
        block_tables = torch.zeros(
            len(request_ids),
            max_blocks,
            device=self.device,
            dtype=torch.long,
        )
        for row_index, request_id in enumerate(request_ids):
            block_ids = self._req_to_blocks[request_id]
            block_tables[row_index, : len(block_ids)] = torch.tensor(
                block_ids,
                device=self.device,
                dtype=torch.long,
            )
        return block_tables

    def _validate_cache_length(
        self,
        request_id: str,
        *,
        expected_tokens: int,
    ) -> None:
        cached_tokens = self._req_to_token_count.get(request_id)
        if cached_tokens is None:
            if expected_tokens == 0:
                return
            raise RuntimeError(
                f"missing KV cache for {request_id} at token {expected_tokens}"
            )
        if cached_tokens != expected_tokens:
            raise RuntimeError(
                f"KV cache for {request_id} has {cached_tokens} tokens, "
                f"expected {expected_tokens}"
            )
