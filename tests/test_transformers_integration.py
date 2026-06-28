import os
from unittest import SkipTest

from tiny_vllm.model_runner import ExecutionBatch, TransformersModelRunner
from tiny_vllm.sequence import SequenceState


def test_transformers_model_runner_real_model_smoke() -> None:
    model_name = os.environ.get("TINY_VLLM_INTEGRATION_MODEL")
    if not model_name:
        raise SkipTest("set TINY_VLLM_INTEGRATION_MODEL to run a real transformers smoke test")

    runner = TransformersModelRunner.from_pretrained(model_name)
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
    sequence.advance_computed_tokens(len(sequence.prompt_token_ids))
    sequence.append_sampled_token(token_id)
    text = runner.detokenize(sequence.generated_token_ids)

    assert sequence.request_id == "integration"
    assert len(sequence.generated_token_ids) == 1
    assert text
