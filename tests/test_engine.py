from tiny_vllm import Engine, EngineConfig, GenerationRequest
from tiny_vllm.request import GenerationOutput


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def generate(self, request: GenerationRequest, max_new_tokens: int) -> GenerationOutput:
        self.calls.append((request.request_id, max_new_tokens))
        return GenerationOutput(
            request_id=request.request_id,
            text=f"{request.prompt} <recorded>",
            generated_tokens=max_new_tokens,
        )


def test_engine_generates_deterministic_mock_outputs_and_releases_cache() -> None:
    engine = Engine(EngineConfig(max_batch_size=2, max_num_blocks=4, block_size=4))

    engine.submit(GenerationRequest(request_id="req-a", prompt="alpha beta", max_new_tokens=3))
    engine.submit(GenerationRequest(request_id="req-b", prompt="gamma", max_new_tokens=2))

    outputs = engine.run_until_complete()

    assert [output.request_id for output in outputs] == ["req-a", "req-b"]
    assert outputs[0].text == "alpha beta <mock-0> <mock-1> <mock-2>"
    assert outputs[1].text == "gamma <mock-0> <mock-1>"
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

    assert runner.calls == [("req-a", 2)]
    assert outputs[0].text == "alpha <recorded>"
    assert engine.stats.completed_requests == 1


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
