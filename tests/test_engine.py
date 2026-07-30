from tiny_vllm import Engine, EngineConfig, GenerationRequest
from tiny_vllm.kv_cache import KVBlock, KVBlockAllocator
from tiny_vllm.model_runner import ExecutionBatch, ModelRunnerOutput
from tiny_vllm.scheduler import Scheduler


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], list[int], list[int], list[int]]] = []

    def encode_prompt(self, prompt: str) -> list[int]:
        return list(range(max(1, len(prompt.split()))))

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
        self.calls.append(
            (
                [sequence.request_id for sequence in batch.sequences],
                batch.num_scheduled_tokens,
                [sequence.num_computed_tokens for sequence in batch.sequences],
                [sequence.num_tokens for sequence in batch.sequences],
            )
        )
        return ModelRunnerOutput(
            sampled_token_ids=[
                len(sequence.generated_token_ids)
                for sequence in batch.sequences
            ]
        )

    def detokenize(self, token_ids: list[int]) -> str:
        return " ".join(f"<recorded-{token_id}>" for token_id in token_ids)


class FailingExecuteRunner(RecordingRunner):
    def __init__(self, message: str, fail_on_call: int) -> None:
        super().__init__()
        self.message = message
        self.fail_on_call = fail_on_call

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
        output = super().execute(batch)
        if len(self.calls) == self.fail_on_call:
            raise RuntimeError(self.message)
        return output


class RecordingReleaseRunner(RecordingRunner):
    def __init__(self) -> None:
        super().__init__()
        self.release_calls: list[str] = []

    def release(self, request_id: str) -> None:
        self.release_calls.append(request_id)


class FailingExecuteReleaseRunner(FailingExecuteRunner):
    def __init__(self, message: str, fail_on_call: int) -> None:
        super().__init__(message=message, fail_on_call=fail_on_call)
        self.release_calls: list[str] = []

    def release(self, request_id: str) -> None:
        self.release_calls.append(request_id)


class FailingEncodeRunner(RecordingRunner):
    def encode_prompt(self, prompt: str) -> list[int]:
        raise RuntimeError("encode failed")


class SelectiveFailingEncodeRunner(RecordingRunner):
    def encode_prompt(self, prompt: str) -> list[int]:
        if prompt == "bad":
            raise RuntimeError("encode failed")
        return super().encode_prompt(prompt)


class EmptyPromptRunner(RecordingRunner):
    def encode_prompt(self, prompt: str) -> list[int]:
        if prompt == "":
            return []
        return super().encode_prompt(prompt)


class RecordingKVBlockAllocator(KVBlockAllocator):
    def __init__(self, num_blocks: int, block_size: int) -> None:
        super().__init__(num_blocks=num_blocks, block_size=block_size)
        self.ensure_calls: list[tuple[str, int]] = []

    def ensure_slots(self, request_id: str, num_tokens: int) -> list[KVBlock]:
        self.ensure_calls.append((request_id, num_tokens))
        return super().ensure_slots(request_id, num_tokens)


class FailingDecodeSlotAllocator(KVBlockAllocator):
    def ensure_slots(self, request_id: str, num_tokens: int) -> list[KVBlock]:
        if request_id == "decode" and num_tokens == 2:
            raise RuntimeError("decode slot failed")
        return super().ensure_slots(request_id, num_tokens)


class RecordingScheduler(Scheduler):
    def __init__(self, max_batch_size: int) -> None:
        super().__init__(max_batch_size=max_batch_size)
        self.pop_next_calls = 0

    def pop_next(self) -> GenerationRequest | None:
        self.pop_next_calls += 1
        return super().pop_next()


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
        (["req-a"], [1], [0], [1]),
        (["req-a"], [1], [1], [2]),
    ]
    assert outputs[0].text == "alpha <recorded-0> <recorded-1>"
    assert engine.stats.completed_requests == 1


def test_engine_releases_runner_cache_when_sequence_completes() -> None:
    runner = RecordingReleaseRunner()
    engine = Engine(
        EngineConfig(max_batch_size=1, max_num_blocks=4, block_size=4),
        model_runner=runner,
    )

    engine.submit(GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=1))
    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["req-a"]
    assert runner.release_calls == ["req-a"]


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
        [1, 0],
        [2, 1],
    )

    final_outputs = engine.run_until_complete()
    assert [output.request_id for output in final_outputs] == ["long"]


def test_engine_does_not_leak_request_id_when_kv_allocation_fails() -> None:
    engine = Engine(EngineConfig(max_batch_size=1, max_num_blocks=1, block_size=1))
    request = GenerationRequest(request_id="too-large", prompt="alpha beta", max_new_tokens=1)

    engine.submit(request)

    outputs = engine.run_step()

    assert len(outputs) == 1
    assert outputs[0].request_id == "too-large"
    assert outputs[0].error is not None
    assert "KV cache capacity" in outputs[0].error
    assert engine.stats.completed_requests == 0
    assert engine.stats.failed_requests == 1
    assert engine.kv_cache.available_blocks == 1

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_continues_admitting_after_request_rejection() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=1))
    too_large = GenerationRequest(
        request_id="too-large",
        prompt="alpha beta gamma",
        max_new_tokens=1,
    )
    small = GenerationRequest(request_id="small", prompt="", max_new_tokens=1)

    engine.submit(too_large)
    engine.submit(small)

    outputs = engine.run_step()

    outputs_by_id = {output.request_id: output for output in outputs}
    assert set(outputs_by_id) == {"too-large", "small"}
    assert outputs_by_id["too-large"].error is not None
    assert "KV cache capacity" in outputs_by_id["too-large"].error
    assert outputs_by_id["small"].error is None
    assert outputs_by_id["small"].text == "<mock-0>"
    assert engine.scheduler.pending_count == 0


def test_engine_rejects_later_impossible_request_after_scheduled_work() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=1))
    small = GenerationRequest(request_id="small", prompt="", max_new_tokens=1)
    too_large = GenerationRequest(
        request_id="too-large",
        prompt="alpha beta gamma",
        max_new_tokens=1,
    )

    engine.submit(small)
    engine.submit(too_large)

    outputs = engine.run_step()

    outputs_by_id = {output.request_id: output for output in outputs}
    assert set(outputs_by_id) == {"small", "too-large"}
    assert outputs_by_id["small"].error is None
    assert outputs_by_id["too-large"].error is not None
    assert "KV cache capacity" in outputs_by_id["too-large"].error
    assert engine.stats.completed_requests == 1
    assert engine.stats.failed_requests == 1
    assert engine.scheduler.pending_count == 0


def test_engine_releases_request_id_when_prefill_allocation_fails() -> None:
    engine = Engine(EngineConfig(max_batch_size=1, max_num_blocks=1, block_size=1))
    request = GenerationRequest(request_id="too-large", prompt="alpha beta", max_new_tokens=1)

    engine.submit(request)

    outputs = engine.run_step()

    assert outputs[0].request_id == "too-large"
    assert outputs[0].error is not None
    assert "KV cache capacity" in outputs[0].error
    assert engine.stats.completed_requests == 0
    assert engine.stats.failed_requests == 1
    assert engine.kv_cache.available_blocks == 1

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_keeps_successfully_prefilled_sequences_when_later_admission_fails() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=1))
    small = GenerationRequest(request_id="small", prompt="", max_new_tokens=1)
    too_large = GenerationRequest(
        request_id="too-large",
        prompt="alpha beta gamma",
        max_new_tokens=1,
    )

    engine.submit(small)
    engine.submit(too_large)

    outputs = engine.run_step()

    outputs_by_id = {output.request_id: output for output in outputs}
    assert set(outputs_by_id) == {"small", "too-large"}
    assert outputs_by_id["small"].error is None
    assert outputs_by_id["too-large"].error is not None
    assert "KV cache capacity" in outputs_by_id["too-large"].error
    assert engine.stats.completed_requests == 1
    assert engine.stats.failed_requests == 1
    assert engine.scheduler.pending_count == 0


def test_engine_releases_prefill_sequences_when_runner_fails() -> None:
    runner = FailingExecuteRunner("prefill failed", fail_on_call=1)
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

    assert runner.calls == [(["req-a"], [1], [0], [1])]
    assert engine.stats.completed_requests == 0
    assert engine.stats.generated_tokens == 0
    assert engine.kv_cache.available_blocks == 4
    assert engine.scheduler.pending_count == 0

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_uses_per_step_slots_instead_of_full_request_budget() -> None:
    engine = Engine(EngineConfig(max_batch_size=1, max_num_blocks=1, block_size=4))
    engine.submit(
        GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=4)
    )

    outputs = engine.run_until_complete()

    assert outputs[0].generated_tokens == 4
    assert engine.stats.completed_requests == 1
    assert engine.kv_cache.available_blocks == 1


def test_engine_chunks_prefill_without_sampling_until_prompt_is_computed() -> None:
    runner = RecordingRunner()
    engine = Engine(
        EngineConfig(
            max_batch_size=1,
            max_num_blocks=4,
            block_size=4,
            max_num_scheduled_tokens=2,
        ),
        model_runner=runner,
    )
    engine.submit(
        GenerationRequest(
            request_id="req-a",
            prompt="alpha beta gamma",
            max_new_tokens=1,
        )
    )

    first_outputs = engine.run_step()

    assert first_outputs == []
    assert runner.calls == [(["req-a"], [2], [0], [3])]
    sequence = engine._active_sequences["req-a"]
    assert sequence.num_computed_tokens == 2
    assert sequence.generated_token_ids == []
    assert engine.stats.generated_tokens == 0

    second_outputs = engine.run_step()

    assert [output.request_id for output in second_outputs] == ["req-a"]
    assert runner.calls[-1] == (["req-a"], [1], [2], [3])
    assert second_outputs[0].generated_tokens == 1
    assert engine.stats.generated_tokens == 1


def test_engine_token_budget_caps_sum_of_running_and_waiting_tokens() -> None:
    runner = RecordingRunner()
    engine = Engine(
        EngineConfig(
            max_batch_size=2,
            max_num_blocks=8,
            block_size=4,
            max_num_scheduled_tokens=2,
        ),
        model_runner=runner,
    )
    engine.submit(GenerationRequest(request_id="decode", prompt="alpha", max_new_tokens=2))

    assert engine.run_step() == []

    engine.submit(
        GenerationRequest(
            request_id="prefill",
            prompt="beta gamma delta",
            max_new_tokens=1,
        )
    )

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["decode"]
    assert runner.calls[-1] == (
        ["decode", "prefill"],
        [1, 1],
        [1, 0],
        [2, 3],
    )
    prefill = engine._active_sequences["prefill"]
    assert prefill.num_computed_tokens == 1
    assert prefill.generated_token_ids == []


def test_engine_extends_decode_slots_at_block_boundary() -> None:
    engine = Engine(EngineConfig(max_batch_size=1, max_num_blocks=2, block_size=4))
    allocator = RecordingKVBlockAllocator(num_blocks=2, block_size=4)
    engine.kv_cache = allocator
    engine.submit(
        GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=5)
    )

    outputs = engine.run_until_complete()

    assert outputs[0].generated_tokens == 5
    assert allocator.ensure_calls == [
        ("req-a", 1),
        ("req-a", 2),
        ("req-a", 3),
        ("req-a", 4),
        ("req-a", 5),
    ]
    assert allocator.available_blocks == 2


def test_engine_completes_decodes_when_capacity_requires_preemption() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=2))
    engine.submit(GenerationRequest(request_id="req-a", prompt="a", max_new_tokens=3))
    engine.submit(GenerationRequest(request_id="req-b", prompt="b", max_new_tokens=3))

    outputs = engine.run_until_complete()

    outputs_by_id = {output.request_id: output for output in outputs}
    assert set(outputs_by_id) == {"req-a", "req-b"}
    assert outputs_by_id["req-a"].generated_tokens == 3
    assert outputs_by_id["req-b"].generated_tokens == 3
    assert engine.stats.completed_requests == 2
    assert engine.stats.generated_tokens == 6
    assert engine.kv_cache.available_blocks == 2


def test_engine_recomputes_preempted_decode_before_it_runs_again() -> None:
    runner = RecordingRunner()
    engine = Engine(
        EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=2),
        model_runner=runner,
    )
    engine.submit(GenerationRequest(request_id="req-a", prompt="a", max_new_tokens=3))
    engine.submit(GenerationRequest(request_id="req-b", prompt="b", max_new_tokens=3))

    assert engine.run_step() == []
    assert engine.run_step() == []

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["req-a"]
    assert "req-b" not in engine._active_sequences
    preempted = engine._waiting_sequences[0]
    assert preempted.num_computed_tokens == 0
    assert preempted.generated_token_ids == [0, 1]

    runner.calls.clear()
    outputs = engine.run_step()

    assert runner.calls == [(["req-b"], [3], [0], [3])]
    assert [output.request_id for output in outputs] == ["req-b"]
    assert outputs[0].generated_tokens == 3


def test_engine_releases_runner_cache_when_sequence_is_preempted() -> None:
    runner = RecordingReleaseRunner()
    engine = Engine(
        EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=2),
        model_runner=runner,
    )
    engine.submit(GenerationRequest(request_id="req-a", prompt="a", max_new_tokens=3))
    engine.submit(GenerationRequest(request_id="req-b", prompt="b", max_new_tokens=3))

    assert engine.run_step() == []
    assert engine.run_step() == []
    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["req-a"]
    assert runner.release_calls == ["req-b", "req-a"]


def test_engine_rejects_request_that_can_never_fit_kv_capacity() -> None:
    runner = RecordingRunner()
    engine = Engine(
        EngineConfig(max_batch_size=1, max_num_blocks=1, block_size=2),
        model_runner=runner,
    )
    engine.submit(GenerationRequest(request_id="too-long", prompt="a", max_new_tokens=3))

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["too-long"]
    assert outputs[0].error is not None
    assert "KV cache capacity" in outputs[0].error
    assert "too-long" in outputs[0].error
    assert runner.calls == []
    assert engine.scheduler.pending_count == 0
    assert engine.kv_cache.available_blocks == 1


def test_engine_does_not_preempt_already_scheduled_decode() -> None:
    runner = RecordingRunner()
    engine = Engine(
        EngineConfig(max_batch_size=2, max_num_blocks=2, block_size=2),
        model_runner=runner,
    )
    engine.submit(GenerationRequest(request_id="req-a", prompt="a", max_new_tokens=3))
    engine.submit(
        GenerationRequest(request_id="req-b", prompt="b c", max_new_tokens=2)
    )

    assert engine.run_step() == []

    runner.calls.clear()
    assert engine.run_step() == []

    assert runner.calls == [(["req-a"], [1], [1], [2])]


def test_engine_releases_request_id_when_prompt_encoding_fails() -> None:
    runner = FailingEncodeRunner()
    engine = Engine(
        EngineConfig(max_batch_size=1, max_num_blocks=4, block_size=4),
        model_runner=runner,
    )
    request = GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=2)

    engine.submit(request)

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["req-a"]
    assert outputs[0].error == "encode failed"
    assert engine.stats.completed_requests == 0
    assert engine.stats.failed_requests == 1
    assert engine.stats.generated_tokens == 0
    assert engine.kv_cache.available_blocks == 4
    assert engine.scheduler.pending_count == 0

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_keeps_active_sequence_when_pending_prompt_encoding_fails() -> None:
    runner = SelectiveFailingEncodeRunner()
    engine = Engine(
        EngineConfig(max_batch_size=2, max_num_blocks=4, block_size=4),
        model_runner=runner,
    )
    engine.submit(GenerationRequest(request_id="good", prompt="good", max_new_tokens=3))

    assert engine.run_step() == []

    engine.submit(GenerationRequest(request_id="bad", prompt="bad", max_new_tokens=1))
    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["bad"]
    assert outputs[0].error == "encode failed"
    assert "good" in engine._active_sequences
    assert engine._active_sequences["good"].generated_token_ids == [0, 1]
    assert engine.scheduler.pending_count == 0

    final_outputs = engine.run_until_complete()

    assert [output.request_id for output in final_outputs] == ["good"]
    assert final_outputs[0].error is None
    assert final_outputs[0].generated_tokens == 3


def test_engine_rejects_empty_encoded_prompt_and_releases_request_id() -> None:
    runner = EmptyPromptRunner()
    engine = Engine(
        EngineConfig(max_batch_size=1, max_num_blocks=4, block_size=4),
        model_runner=runner,
    )
    request = GenerationRequest(request_id="empty", prompt="", max_new_tokens=1)

    engine.submit(request)
    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["empty"]
    assert outputs[0].error == "encoded prompt is empty"
    assert engine.scheduler.pending_count == 0
    assert engine.stats.failed_requests == 1

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_releases_decode_sequences_when_runner_fails() -> None:
    runner = FailingExecuteRunner("decode failed", fail_on_call=2)
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
        (["req-a"], [1], [0], [1]),
        (["req-a"], [1], [1], [2]),
    ]
    assert engine.stats.completed_requests == 0
    assert engine.stats.generated_tokens == 1
    assert engine.kv_cache.available_blocks == 4
    assert engine.scheduler.pending_count == 0

    engine.submit(request)
    assert engine.scheduler.pending_count == 1


def test_engine_releases_runner_cache_when_runner_fails() -> None:
    runner = FailingExecuteReleaseRunner("prefill failed", fail_on_call=1)
    engine = Engine(
        EngineConfig(max_batch_size=1, max_num_blocks=4, block_size=4),
        model_runner=runner,
    )
    engine.submit(GenerationRequest(request_id="req-a", prompt="alpha", max_new_tokens=1))

    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "prefill failed" in str(exc)
    else:
        raise AssertionError("expected prefill failure")

    assert runner.release_calls == ["req-a"]


def test_engine_keeps_prefills_pending_when_decode_slot_allocation_fails() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=4, block_size=4))
    engine.kv_cache = FailingDecodeSlotAllocator(num_blocks=4, block_size=4)
    engine.submit(GenerationRequest(request_id="decode", prompt="alpha", max_new_tokens=2))

    assert engine.run_step() == []

    engine.submit(GenerationRequest(request_id="prefill", prompt="beta", max_new_tokens=1))
    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "decode slot failed" in str(exc)
    else:
        raise AssertionError("expected decode slot allocation to fail")

    assert engine.scheduler.pending_count == 1
    assert engine.kv_cache.available_blocks == 4

    outputs = engine.run_step()

    assert [output.request_id for output in outputs] == ["prefill"]


def test_engine_does_not_pop_prefills_when_decode_slot_allocation_fails() -> None:
    scheduler = RecordingScheduler(max_batch_size=2)
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=4, block_size=4))
    engine.scheduler = scheduler
    engine.kv_cache = FailingDecodeSlotAllocator(num_blocks=4, block_size=4)
    engine.submit(GenerationRequest(request_id="decode", prompt="alpha", max_new_tokens=2))

    assert engine.run_step() == []

    scheduler.pop_next_calls = 0
    engine.submit(GenerationRequest(request_id="prefill", prompt="beta", max_new_tokens=1))
    try:
        engine.run_step()
    except RuntimeError as exc:
        assert "decode slot failed" in str(exc)
    else:
        raise AssertionError("expected decode slot allocation to fail")

    assert scheduler.pop_next_calls == 0
    assert engine.scheduler.pending_count == 1
