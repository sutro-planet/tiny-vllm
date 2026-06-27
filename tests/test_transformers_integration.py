import os

import pytest

from tiny_vllm.model_runner import TransformersModelRunner
from tiny_vllm.request import GenerationRequest


@pytest.mark.skipif(
    not os.environ.get("TINY_VLLM_INTEGRATION_MODEL"),
    reason="set TINY_VLLM_INTEGRATION_MODEL to run a real transformers smoke test",
)
def test_transformers_model_runner_real_model_smoke() -> None:
    runner = TransformersModelRunner.from_pretrained(os.environ["TINY_VLLM_INTEGRATION_MODEL"])

    output = runner.generate(
        GenerationRequest(request_id="integration", prompt="Hello", max_new_tokens=1),
        max_new_tokens=1,
    )

    assert output.request_id == "integration"
    assert output.generated_tokens == 1
    assert output.text
