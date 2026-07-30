from importlib.util import find_spec
from unittest import SkipTest

if find_spec("torch") is None:
    raise SkipTest("torch is required for tiny GPT-2 tests")

import torch

from tiny_vllm.torch_gpt2 import GPT2Config, TinyGPT2LMHeadModel


def test_tiny_gpt2_cached_forward_matches_full_forward_suffix() -> None:
    torch.manual_seed(0)
    model = TinyGPT2LMHeadModel(
        GPT2Config(
            vocab_size=17,
            max_position_embeddings=8,
            hidden_size=12,
            num_layers=2,
            num_heads=3,
        )
    )
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    with torch.no_grad():
        full_output = model(input_ids=input_ids, use_cache=True)
        prefix_output = model(input_ids=input_ids[:, :2], use_cache=True)
        suffix_output = model(
            input_ids=input_ids[:, 2:],
            position_ids=torch.tensor([[2, 3]], dtype=torch.long),
            past_key_values=prefix_output.past_key_values,
            use_cache=True,
        )

    assert len(full_output.past_key_values) == 2
    first_key, first_value = full_output.past_key_values[0]
    assert first_key.shape == (1, 3, 4, 4)
    assert first_value.shape == (1, 3, 4, 4)
    torch.testing.assert_close(
        full_output.logits[:, 2:, :],
        suffix_output.logits,
        rtol=1e-5,
        atol=1e-5,
    )


def test_tiny_gpt2_flat_block_forward_matches_full_forward() -> None:
    torch.manual_seed(4)
    model = TinyGPT2LMHeadModel(
        GPT2Config(
            vocab_size=17,
            max_position_embeddings=8,
            hidden_size=12,
            num_layers=2,
            num_heads=3,
        )
    )
    model.eval()
    first_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    second_ids = torch.tensor([[4, 5]], dtype=torch.long)
    block_size = 2
    num_blocks = 3
    kv_caches = [
        (
            torch.zeros(num_blocks, 3, block_size, 4),
            torch.zeros(num_blocks, 3, block_size, 4),
        )
        for _ in range(2)
    ]

    with torch.no_grad():
        first_output = model(input_ids=first_ids, use_cache=True)
        second_output = model(input_ids=second_ids, use_cache=True)
        flat_output = model.forward_flat(
            input_ids=torch.tensor([1, 2, 3, 4, 5], dtype=torch.long),
            positions=torch.tensor([0, 1, 2, 0, 1], dtype=torch.long),
            req_indices=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
            query_start_loc=torch.tensor([0, 3, 5], dtype=torch.long),
            seq_lens=torch.tensor([3, 2], dtype=torch.long),
            block_tables=torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
            kv_caches=kv_caches,
            block_size=block_size,
        )

    torch.testing.assert_close(
        flat_output.logits[:3],
        first_output.logits[0],
        rtol=1e-5,
        atol=1e-5,
    )
    torch.testing.assert_close(
        flat_output.logits[3:],
        second_output.logits[0],
        rtol=1e-5,
        atol=1e-5,
    )


def test_tiny_gpt2_loads_hf_style_state_dict() -> None:
    torch.manual_seed(1)
    source = TinyGPT2LMHeadModel(
        GPT2Config(
            vocab_size=11,
            max_position_embeddings=8,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
        )
    )
    target = TinyGPT2LMHeadModel(source.config)

    target.load_hf_state_dict(source.state_dict())

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        source_output = source(input_ids=input_ids, use_cache=True)
        target_output = target(input_ids=input_ids, use_cache=True)

    torch.testing.assert_close(source_output.logits, target_output.logits)


def test_tiny_gpt2_ignores_extra_hf_state_dict_buffers() -> None:
    torch.manual_seed(2)
    source = TinyGPT2LMHeadModel(
        GPT2Config(
            vocab_size=11,
            max_position_embeddings=8,
            hidden_size=8,
            num_layers=1,
            num_heads=2,
        )
    )
    target = TinyGPT2LMHeadModel(source.config)
    state_dict = source.state_dict()
    state_dict["transformer.h.0.attn.masked_bias"] = torch.tensor(-1e4)

    target.load_hf_state_dict(state_dict)

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        source_output = source(input_ids=input_ids, use_cache=True)
        target_output = target(input_ids=input_ids, use_cache=True)

    torch.testing.assert_close(source_output.logits, target_output.logits)
