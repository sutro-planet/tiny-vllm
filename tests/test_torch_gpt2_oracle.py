import os
from importlib.util import find_spec
from typing import Any, cast
from unittest import SkipTest

if find_spec("torch") is None:
    raise SkipTest("torch is required for Torch GPT-2 oracle tests")
if find_spec("transformers") is None:
    raise SkipTest("transformers is required for Torch GPT-2 oracle tests")

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from tiny_vllm.torch_gpt2 import GPT2Config, TinyGPT2LMHeadModel


def test_torch_gpt2_forward_matches_hf_oracle() -> None:
    model_name = os.environ.get("TINY_VLLM_INTEGRATION_MODEL")
    if not model_name:
        raise SkipTest("set TINY_VLLM_INTEGRATION_MODEL to run GPT-2 oracle test")

    hf_config = AutoConfig.from_pretrained(model_name)
    if hf_config.model_type != "gpt2":
        raise SkipTest("Torch GPT-2 oracle test only supports GPT-2 models")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = cast(torch.nn.Module, AutoModelForCausalLM.from_pretrained(model_name))
    hf_model.eval()
    model = TinyGPT2LMHeadModel(GPT2Config.from_hf_config(hf_config))
    model.load_hf_state_dict(hf_model.state_dict())
    model.eval()

    input_ids = tokenizer("Hello world", return_tensors="pt")["input_ids"]
    if input_ids.shape[1] < 2:
        raise SkipTest("oracle prompt must tokenize to at least two tokens")

    with torch.no_grad():
        hf_model_call = cast(Any, hf_model)
        hf_full = hf_model_call(input_ids=input_ids, use_cache=True)
        tiny_full = model(input_ids=input_ids, use_cache=True)

        prefix_ids = input_ids[:, :-1]
        suffix_ids = input_ids[:, -1:]
        hf_prefix = hf_model_call(input_ids=prefix_ids, use_cache=True)
        hf_suffix = hf_model_call(
            input_ids=suffix_ids,
            past_key_values=hf_prefix.past_key_values,
            use_cache=True,
        )
        tiny_prefix = model(input_ids=prefix_ids, use_cache=True)
        tiny_suffix = model(
            input_ids=suffix_ids,
            position_ids=torch.tensor([[prefix_ids.shape[1]]], dtype=torch.long),
            past_key_values=tiny_prefix.past_key_values,
            use_cache=True,
        )

    torch.testing.assert_close(tiny_full.logits, hf_full.logits, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(tiny_suffix.logits, hf_suffix.logits, rtol=1e-4, atol=1e-4)
