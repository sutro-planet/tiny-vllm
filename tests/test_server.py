import asyncio
from importlib.util import find_spec
from typing import Any
from unittest import SkipTest

import pytest

if find_spec("aiohttp") is None:
    raise SkipTest("install tiny-vllm[server] for HTTP adapter tests")

from aiohttp.test_utils import TestClient, TestServer

from tiny_vllm import EngineConfig
from tiny_vllm.model_runner import ExecutionBatch, MockModelRunner, ModelRunnerOutput
from tiny_vllm.server import APIError, EngineService, create_app, parse_request


class RecordingRunner(MockModelRunner):
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
        self.batch_sizes.append(len(batch.sequences))
        return super().execute(batch)


def payload(**extra: Any) -> dict[str, Any]:
    return {"model": "mock-gemma", "prompt": "one two three", "max_tokens": 4, **extra}


def test_http_requests_share_engine_and_report_exact_usage_without_echo() -> None:
    async def scenario() -> None:
        runner = RecordingRunner()
        service = EngineService(
            EngineConfig(max_batch_size=4, max_num_scheduled_tokens=8),
            runner,
        )
        async with TestClient(TestServer(create_app(service))) as client:
            responses = await asyncio.gather(
                *[client.post("/v1/completions", json=payload()) for _ in range(6)]
            )
            for response in responses:
                assert response.status == 200
                result = await response.json()
                assert result["choices"][0]["text"] == "<mock-0> <mock-1> <mock-2> <mock-3>"
                assert result["usage"] == {
                    "prompt_tokens": 3,
                    "completion_tokens": 4,
                    "total_tokens": 7,
                }
            assert max(runner.batch_sizes) > 1
            assert service.engine.stats.completed_requests == 6
            assert service.engine.kv_cache.available_blocks == 16
            assert not service.pending

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "extra",
    [
        {"stream": True},
        {"temperature": 1},
        {"ignore_eos": False},
        {"stop": ["."]},
        {"max_tokens": True},
        {"max_tokens": 0},
        {"model": "other"},
        {"prompt": ["a", "b"]},
    ],
)
def test_unsupported_workload_is_rejected_instead_of_silently_changed(
    extra: dict[str, Any],
) -> None:
    with pytest.raises(APIError):
        parse_request(payload(**extra), "mock-gemma")


def test_context_overflow_fails_only_that_request() -> None:
    async def scenario() -> None:
        service = EngineService(EngineConfig(), MockModelRunner(), max_model_len=8)
        async with TestClient(TestServer(create_app(service))) as client:
            bad, good = await asyncio.gather(
                client.post("/v1/completions", json=payload(max_tokens=8)),
                client.post("/v1/completions", json=payload()),
            )
            assert bad.status == 400
            assert "max_model_len" in (await bad.json())["error"]["message"]
            assert good.status == 200
            assert service.engine.stats.completed_requests == 1

    asyncio.run(scenario())


def test_runner_failure_fails_pending_requests_and_marks_server_unhealthy() -> None:
    class FailingRunner(MockModelRunner):
        def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
            raise RuntimeError("injected failure")

    async def scenario() -> None:
        service = EngineService(EngineConfig(), FailingRunner())
        async with TestClient(TestServer(create_app(service))) as client:
            responses = await asyncio.gather(
                *[client.post("/v1/completions", json=payload()) for _ in range(3)]
            )
            assert all(response.status in (500, 503) for response in responses)
            assert (await client.get("/health")).status == 503
            assert (await client.post("/v1/completions", json=payload())).status == 503
            assert service.engine.kv_cache.available_blocks == 16

    asyncio.run(scenario())


def test_models_defaults_and_openai_errors() -> None:
    async def scenario() -> None:
        service = EngineService(EngineConfig(), MockModelRunner())
        async with TestClient(TestServer(create_app(service))) as client:
            models = await (await client.get("/v1/models")).json()
            assert models["object"] == "list"
            assert models["data"][0]["id"] == "mock-gemma"
            response = await client.post(
                "/v1/completions", json={"model": "mock-gemma", "prompt": "hi"}
            )
            result = await response.json()
            assert response.status == 200
            assert result["object"] == "text_completion"
            assert result["usage"]["completion_tokens"] == 16
            assert result["choices"][0]["finish_reason"] == "length"
            for body, status in ((payload(model="unknown"), 404), (payload(stream=True), 400)):
                response = await client.post("/v1/completions", json=body)
                assert response.status == status
                error = (await response.json())["error"]
                assert error["type"] == "invalid_request_error"
                assert error["param"] in ("model", "stream")
                assert "code" in error
            response = await client.post("/v1/completions", data="{broken")
            assert response.status == 400
            assert (await response.json())["error"]["message"] == "invalid JSON body"

    asyncio.run(scenario())


@pytest.mark.parametrize("decoded", ["  leading space\n", "", "one two three"])
def test_completion_preserves_decoded_text_and_counts_generated_ids(decoded: str) -> None:
    class TextRunner(MockModelRunner):
        def detokenize(self, token_ids: list[int]) -> str:
            return decoded

    async def scenario() -> None:
        service = EngineService(EngineConfig(), TextRunner())
        async with TestClient(TestServer(create_app(service))) as client:
            response = await client.post("/v1/completions", json=payload())
            result = await response.json()
            assert result["choices"][0]["text"] == decoded
            assert result["usage"]["completion_tokens"] == 4

    asyncio.run(scenario())


def test_worker_keeps_health_responsive_and_bounds_queue() -> None:
    from threading import Event

    started, release = Event(), Event()

    class SlowRunner(MockModelRunner):
        def execute(self, batch: ExecutionBatch) -> ModelRunnerOutput:
            started.set()
            if not release.wait(timeout=5):
                raise RuntimeError("test worker timed out")
            return super().execute(batch)

    async def scenario() -> None:
        service = EngineService(EngineConfig(), SlowRunner(), max_pending=1)
        async with TestClient(TestServer(create_app(service))) as client:
            first = asyncio.create_task(client.post("/v1/completions", json=payload()))
            try:
                assert await asyncio.to_thread(started.wait, 5)
                response = await asyncio.wait_for(client.get("/health"), 1)
                assert response.status == 200
                overflow = await asyncio.wait_for(client.post("/v1/completions", json=payload()), 1)
                assert overflow.status == 429
                assert (await overflow.json())["error"]["type"] == "rate_limit_error"
            finally:
                release.set()
            assert (await first).status == 200

    asyncio.run(scenario())


def test_cancelled_caller_and_shutdown_drain_owned_kv() -> None:
    from tiny_vllm import GenerationRequest

    async def scenario() -> None:
        service = EngineService(EngineConfig(), MockModelRunner())
        await service.start()
        request = asyncio.create_task(service.generate(GenerationRequest("cancelled", "hello", 4)))
        await asyncio.sleep(0)
        assert "cancelled" in service.pending
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        await service.close()
        assert not service.pending
        assert not service.prompt_counts
        assert service.engine.stats.completed_requests == 1
        assert service.engine.kv_cache.available_blocks == 16

    asyncio.run(scenario())
