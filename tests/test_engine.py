from tiny_vllm import Engine, EngineConfig, GenerationRequest
from tiny_vllm.model_runner import ExecutionBatch, ModelRunnerOutput
from tiny_vllm.sequence import SequencePhase


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[int], list[SequencePhase]]] = []

    def encode_prompt(self, prompt: str) -> list[int]:
        return list(range(max(1, len(prompt.split()))))

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
        self.calls.append(
            (
                [sequence.request_id for sequence in batch.sequences],
                batch.num_scheduled_tokens,
                [sequence.phase for sequence in batch.sequences],
            )
        )
        return ModelRunnerOutput(
            sampled_token_ids=[
                0
                if sequence.phase is SequencePhase.WAITING_PREFILL
                else len(sequence.generated_token_ids)
                for sequence in batch.sequences
            ]
        )

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(f"<recorded-{token_id}>" for token_id in token_ids)


class FailingExecuteRunner(RecordingRunner):
    def __init__(self, message: str, fail_on_phase: SequencePhase) -> None:
        super().__init__()
        self.message = message
        self.fail_on_phase = fail_on_phase

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
        output = super().execute(batch)
        if any(sequence.phase is self.fail_on_phase for sequence in batch.sequences):
            raise RuntimeError(self.message)
        return output


class FailingEncodeRunner(RecordingRunner):
    def encode_prompt(self, prompt: str) -> list[int]:
        raise RuntimeError("encode failed")


def test_engine_generates_deterministic_mock_outputs_and_releases_cache() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=4, block_size=4))

    engine.submit(GenerationRequest(request_id="req-a", prompt="alpha beta", max_new_tokens=3))
    engine.submit(GenerationRequest(request_id="req-b", prompt="gamma", max_new_tokens=2))

    outputs = engine.run_until_complete()

    outputs_by_id = {output.request_id: output for output in outputs}
    assert list(outputs_by_id) == ["req-b", "req-a"]
    assert outputs_by_id["req-a"].text == "alpha beta <mock-0> <mock-1> <mock-2>"
    assert outputs_by_id["req-b"].text == "gamma <mock-0> <mock-1>"
    assert engine.stats.completed_requests == 2
    assert engine.stats.generated_tokens == 5
    assert engine.kv_cache.available_blocks == 4


def test_engine_accepts_injected_model_runner() -> None:
    runner = RecordingRunner()
    engine = Engine(
        EngineConfig(max_batch_size=1, max_num_blocks=4, block_size=4),
        model_runner=runner,
    )

    engine.submit(GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=2))

    outputs = engine.run_until_complete()

    assert runner.calls == [
        (["req-a"], [1], [SequencePhase.WAITING_PREFILL]),
        (["req-a"], [1], [SequencePhase.DECODING]),
    ]
    assert outputs[0].text == "alpha <recorded-0> <recorded-1>"
    assert engine.stats.completed_requests == 1


def test_engine_prefill_can_complete_one_token_request() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=4, block_size=4))
    engine.submit(GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=1))

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["req-a"]
    assert outputs[0].text == "alpha <mock-0>"
    assert outputs[0].generated_tokens == 1
    assert engine.stats.completed_requests == 1
    assert engine.stats.generated_tokens == 1
    assert engine.kv_cache.available_blocks == 4


def test_engine_decodes_active_sequence_after_prefill() -> None:
    engine = Engine(EngineConfig(max_batch_size=1, max_num_blocks=4, block_size=4))
    engine.submit(GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=2))

    assert engine.run_step() == []

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["req-a"]
    assert outputs[0].text == "alpha <mock-0> <mock-1>"
    assert outputs[0].generated_tokens == 2
    assert engine.stats.completed_requests == 1
    assert engine.stats.generated_tokens == 2
    assert engine.kv_cache.available_blocks == 4


def test_engine_refills_slots_with_prefill_after_sequence_finishes() -> None:
    runner = RecordingRunner()
    engine = Engine(
        EngineConfig(max_batch_size=2, max_num_blocks=8, block_size=4),
        model_runner=runner,
    )
    engine.submit(GenerationRequest(request_id="short", prompt="a", max_new_tokens=1))
    engine.submit(GenerationRequest(request_id="long", prompt="b", max_new_tokens=3))
    engine.submit(GenerationRequest(request_id="next", prompt="c", max_new_tokens=1))

    first_outputs = engine.run_step()
    assert [output.request_id for output in first_outputs] == ["short"]

    second_outputs = engine.run_step()
    assert [output.request_id for output in second_outputs] == ["next"]
    assert len(runner.calls) == 2
    assert runner.calls[1] == (
        ["long", "next"],
        [1, 1],
        [SequencePhase.DECODING, SequencePhase.WAITING_PREFILL],
    )

    final_outputs = engine.run_until_complete()
    assert [output.request_id for output in final_outputs] == ["long"]


def test_engine_does_not_leak_request_id_when_kv_allocation_fails() -> None:
    engine = Engine(EngineConfig(max_batch_size=1, max_num_blocks=1, block_size=1))
    request = GenerationRequest(request_id="too-large", prompt="alpha beta", max_new_tokens=1)

    engine.submit(request)

    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "KV cache capacity" in str(exc)
    else:
        raise AssertionError("expected KV allocation to fail")

    assert engine.stats.completed_requests == 0
    assert engine.kv_cache.available_blocks == 1

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_requeues_unprocessed_batch_requests_after_failure() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=1))
    too_large = GenerationRequest(request_id="too-large", prompt="alpha beta", max_new_tokens=1)
    small = GenerationRequest(request_id="small", prompt="", max_new_tokens=1)

    engine.submit(too_large)
    engine.submit(small)

    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "KV cache capacity" in str(exc)
    else:
        raise AssertionError("expected KV allocation to fail")

    assert engine.scheduler.pending_count == 1

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["small"]
    assert outputs[0].text == "<mock-0>"


def test_engine_requeues_earlier_batch_requests_when_later_allocation_fails() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=1))
    small = GenerationRequest(request_id="small", prompt="", max_new_tokens=1)
    too_large = GenerationRequest(request_id="too-large", prompt="alpha beta", max_new_tokens=1)

    engine.submit(small)
    engine.submit(too_large)

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["small"]
    assert engine.stats.completed_requests == 1
    assert engine.scheduler.pending_count == 1

    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "KV cache capacity" in str(exc)
    else:
        raise AssertionError("expected KV allocation to fail")

    assert engine.stats.completed_requests == 1
    assert engine.scheduler.pending_count == 0


def test_engine_releases_request_id_when_prefill_allocation_fails() -> None:
    engine = Engine(EngineConfig(max_batch_size=1, max_num_blocks=1, block_size=1))
    request = GenerationRequest(request_id="too-large", prompt="alpha beta", max_new_tokens=1)

    engine.submit(request)

    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "KV cache capacity" in str(exc)
    else:
        raise AssertionError("expected KV allocation to fail")

    assert engine.stats.completed_requests == 0
    assert engine.kv_cache.available_blocks == 1

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_keeps_successfully_prefilled_sequences_when_later_admission_fails() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=1))
    small = GenerationRequest(request_id="small", prompt="", max_new_tokens=1)
    too_large = GenerationRequest(request_id="too-large", prompt="alpha beta", max_new_tokens=1)

    engine.submit(small)
    engine.submit(too_large)

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["small"]
    assert engine.stats.completed_requests == 1
    assert engine.scheduler.pending_count == 1

    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "KV cache capacity" in str(exc)
    else:
        raise AssertionError("expected KV allocation to fail")

    assert engine.stats.completed_requests == 1
    assert engine.scheduler.pending_count == 0


def test_engine_releases_prefill_sequences_when_runner_fails() -> None:
    runner = FailingExecuteRunner("prefill failed", SequencePhase.WAITING_PREFILL)
    engine = Engine(
        EngineConfig(max_batch_size=1, max_num_blocks=4, block_size=4),
        model_runner=runner,
    )
    request = GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=2)

    engine.submit(request)

    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "prefill failed" in str(exc)
    else:
        raise AssertionError("expected prefill failure")

    assert runner.calls == [(["req-a"], [1], [SequencePhase.WAITING_PREFILL])]
    assert engine.stats.completed_requests == 0
    assert engine.stats.generated_tokens == 0
    assert engine.kv_cache.available_blocks == 4
    assert engine.scheduler.pending_count == 0

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_releases_request_id_when_prompt_encoding_fails() -> None:
    runner = FailingEncodeRunner()
    engine = Engine(
        EngineConfig(max_batch_size=1, max_num_blocks=4, block_size=4),
        model_runner=runner,
    )
    request = GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=2)

    engine.submit(request)

    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "encode failed" in str(exc)
    else:
        raise AssertionError("expected encode failure")

    assert engine.stats.completed_requests == 0
    assert engine.stats.generated_tokens == 0
    assert engine.kv_cache.available_blocks == 4
    assert engine.scheduler.pending_count == 0

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_releases_decode_sequences_when_runner_fails() -> None:
    runner = FailingExecuteRunner("decode failed", SequencePhase.DECODING)
    engine = Engine(
        EngineConfig(max_batch_size=1, max_num_blocks=4, block_size=4),
        model_runner=runner,
    )
    request = GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=2)

    engine.submit(request)
    assert engine.run_step() == []

    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "decode failed" in str(exc)
    else:
        raise AssertionError("expected decode failure")

    assert runner.calls == [
        (["req-a"], [1], [SequencePhase.WAITING_PREFILL]),
        (["req-a"], [1], [SequencePhase.DECODING]),
    ]
    assert engine.stats.completed_requests == 0
    assert engine.stats.generated_tokens == 1
    assert engine.kv_cache.available_blocks == 4
    assert engine.scheduler.pending_count == 0

    engine.submit(request)
    assert engine.scheduler.pending_count == 1
