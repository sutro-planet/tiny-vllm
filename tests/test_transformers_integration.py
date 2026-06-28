import os

import pytest

from tiny_vllm.model_runner import ExecutionBatch, TransformersModelRunner
from tiny_vllm.sequence import SequenceState


@pytest.mark.skipif(
    not os.environ.get("TINY_VLLM_INTEGRATION_MODEL"),
    reason="set TINY_VLLM_INTEGRATION_MODEL to run a real transformers smoke test",
)
def test_transformers_model_runner_real_model_smoke() -> None:
    runner = TransformersModelRunner.from_pretrained(os.environ["TINY_VLLM_INTEGRATION_MODEL"])
    sequence = SequenceState(
        request_id="integration",
        prompt="Hello",
        prompt_token_ids=runner.encode_prompt("Hello"),
        max_new_tokens=1,
    )

    token_id = runner.execute(
        ExecutionBatch(
            sequences=[sequence],
            num_scheduled_tokens=[len(sequence.prompt_token_ids)],
        )
    ).sampled_token_ids[0]
    sequence.append_prefill_token(token_id)
    text = runner.detokenize(sequence.generated_token_ids)

    assert sequence.request_id == "integration"
    assert len(sequence.generated_token_ids) == 1
    assert text
