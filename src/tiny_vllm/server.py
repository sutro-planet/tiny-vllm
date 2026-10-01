"""Serve the supported OpenAI Completions API subset on one shared Engine."""

import argparse
import asyncio
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from aiohttp import web

from tiny_vllm import Engine, EngineConfig, GenerationOutput, GenerationRequest
from tiny_vllm.model_runner import MockModelRunner, ModelRunner

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(self, message: str, *, status: int = 400, param: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.param = param


class EngineService:
    """Batch HTTP requests through one engine; keep model work off the event loop."""

    def __init__(
        self,
        config: EngineConfig,
        runner: ModelRunner,
        *,
        max_model_len: int = 1024,
        max_pending: int = 256,
    ) -> None:
        if max_model_len <= 0 or max_pending <= 0:
            raise ValueError("max_model_len and max_pending must be positive")
        self.engine = Engine(config, model_runner=runner)
        self.max_model_len = max_model_len
        self.max_pending = max_pending
        self.pending: dict[str, asyncio.Future[tuple[GenerationOutput, int]]] = {}
        self.incoming: deque[GenerationRequest] = deque()
        self.prompt_counts: dict[str, int] = {}
        self.wake = asyncio.Event()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tiny-engine")
        self.task: asyncio.Task[None] | None = None
        self.failed = False
        self.closing = False

    async def start(self) -> None:
        self.task = asyncio.create_task(self._run())

    async def close(self) -> None:
        # Drain accepted requests before shutting down the model worker.
        self.closing = True
        self.wake.set()
        if self.task is not None:
            await self.task
        await asyncio.to_thread(self.executor.shutdown, wait=True)

    async def generate(self, request: GenerationRequest) -> tuple[GenerationOutput, int]:
        if self.failed or self.closing:
            raise APIError("engine is unavailable", status=503)
        if len(self.pending) >= self.max_pending:
            raise APIError("request queue is full", status=429)
        future: asyncio.Future[tuple[GenerationOutput, int]] = (
            asyncio.get_running_loop().create_future()
        )
        self.pending[request.request_id] = future
        self.incoming.append(request)
        self.wake.set()
        # A disconnected caller cancels its future; the engine still drains its
        # request and releases KV blocks because Engine has no cancellation API.
        return await future

    async def _run(self) -> None:
        while True:
            await self.wake.wait()
            self.wake.clear()
            if self.closing and not self.pending:
                return
            incoming = list(self.incoming)
            self.incoming.clear()
            try:
                outputs = await asyncio.get_running_loop().run_in_executor(
                    self.executor, self._step, incoming
                )
            except Exception:
                logger.exception("engine worker failed")
                self.failed = True
                for future in self.pending.values():
                    if not future.done():
                        future.set_exception(APIError("engine execution failed", status=500))
                self.pending.clear()
                self.incoming.clear()
                self.prompt_counts.clear()
                return
            for output, prompt_tokens in outputs:
                future = self.pending.pop(output.request_id)
                if not future.done():
                    future.set_result((output, prompt_tokens))
            if self.pending or self.closing:
                self.wake.set()

    def _step(self, incoming: list[GenerationRequest]) -> list[tuple[GenerationOutput, int]]:
        results: list[tuple[GenerationOutput, int]] = []
        for request in incoming:
            try:
                # Preflight is server-owned. Engine tokenizes again at admission;
                # its public request/output contract stays unchanged.
                prompt_tokens = len(self.engine.model_runner.encode_prompt(request.prompt))
                if not prompt_tokens:
                    raise ValueError("encoded prompt is empty")
                max_tokens = request.max_new_tokens or self.engine.config.max_new_tokens
                if prompt_tokens + max_tokens > self.max_model_len:
                    raise ValueError(
                        f"prompt plus output exceeds max_model_len={self.max_model_len}"
                    )
            except Exception as exc:
                results.append((GenerationOutput(request.request_id, "", 0, str(exc)), 0))
                continue
            self.prompt_counts[request.request_id] = prompt_tokens
            self.engine.submit(request)
        for output in self.engine.run_step():
            results.append((output, self.prompt_counts.pop(output.request_id)))
        return results


SERVICE = web.AppKey("engine_service", EngineService)


def parse_request(payload: Any, model_name: str) -> GenerationRequest:
    if not isinstance(payload, dict):
        raise APIError("request must be a JSON object")
    allowed = {
        "model",
        "prompt",
        "max_tokens",
        "stream",
        "temperature",
        "top_p",
        "ignore_eos",
        "n",
        "echo",
        "stop",
        "logprobs",
        "presence_penalty",
        "frequency_penalty",
        "best_of",
        "user",
    }
    if unknown := payload.keys() - allowed:
        raise APIError(f"unsupported fields: {', '.join(sorted(unknown))}")
    if not isinstance(payload.get("model"), str):
        raise APIError("model must be a string", param="model")
    if payload["model"] != model_name:
        raise APIError(f"model {payload['model']!r} is not served", status=404, param="model")
    for name, boolean_default in (("stream", False), ("echo", False), ("ignore_eos", True)):
        if payload.get(name, boolean_default) is not boolean_default:
            raise APIError(f"only {name}={boolean_default} is supported", param=name)
    for name, default in (
        ("temperature", 0),
        ("top_p", 1),
        ("n", 1),
        ("best_of", 1),
        ("presence_penalty", 0),
        ("frequency_penalty", 0),
    ):
        value = payload.get(name, default)
        if type(value) not in (int, float) or value != default:
            raise APIError(f"only {name}={default} is supported", param=name)
    for name in ("stop", "logprobs"):
        if payload.get(name) is not None:
            raise APIError(f"{name} is not supported", param=name)
    if "user" in payload and not isinstance(payload["user"], str):
        raise APIError("user must be a string", param="user")
    prompt = payload.get("prompt")
    if isinstance(prompt, list) and len(prompt) == 1:
        prompt = prompt[0]
    if not isinstance(prompt, str) or not prompt:
        raise APIError(
            "prompt must be a non-empty string or one-element string list", param="prompt"
        )
    max_tokens = payload.get("max_tokens", 16)
    if type(max_tokens) is not int or max_tokens <= 0:
        raise APIError("max_tokens must be a positive integer", param="max_tokens")
    return GenerationRequest(f"cmpl-{uuid.uuid4().hex}", prompt, max_tokens)


@web.middleware
async def errors(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.StreamResponse:
    try:
        return await handler(request)
    except APIError as exc:
        status, message, param = exc.status, str(exc), exc.param
    except web.HTTPException as exc:
        status, message, param = exc.status, exc.reason, None
    error_type = (
        "server_error"
        if status >= 500
        else "rate_limit_error"
        if status == 429
        else "invalid_request_error"
    )
    return web.json_response(
        {"error": {"message": message, "type": error_type, "param": param, "code": None}},
        status=status,
    )


async def completions(request: web.Request) -> web.Response:
    service = request.app[SERVICE]
    try:
        payload = await request.json()
    except (ValueError, UnicodeError) as exc:
        raise APIError("invalid JSON body") from exc
    generation = parse_request(payload, service.engine.config.model_name)
    output, prompt_tokens = await service.generate(generation)
    if output.error is not None:
        raise APIError(output.error)
    # Engine's public text is prompt + one separator + decoded completion.
    # Remove exactly that prefix to preserve generated leading whitespace.
    text = output.text[len(generation.prompt) + 1 :] if output.text != generation.prompt else ""
    return web.json_response(
        {
            "id": output.request_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": service.engine.config.model_name,
            "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": output.generated_tokens,
                "total_tokens": prompt_tokens + output.generated_tokens,
            },
        }
    )


async def models(request: web.Request) -> web.Response:
    return web.json_response(
        {
            "object": "list",
            "data": [
                {
                    "id": request.app[SERVICE].engine.config.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "tiny-vllm",
                }
            ],
        }
    )


async def health(request: web.Request) -> web.Response:
    service = request.app[SERVICE]
    if service.failed or service.closing:
        raise APIError("engine is unavailable", status=503)
    return web.json_response({"status": "ok"})


async def lifecycle(app: web.Application) -> AsyncIterator[None]:
    await app[SERVICE].start()
    try:
        yield
    finally:
        await app[SERVICE].close()


def create_app(service: EngineService) -> web.Application:
    app = web.Application(client_max_size=1024 * 1024, middlewares=[errors])
    app[SERVICE] = service
    app.cleanup_ctx.append(lifecycle)
    app.add_routes(
        [
            web.post("/v1/completions", completions),
            web.get("/v1/models", models),
            web.get("/health", health),
        ]
    )
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--mock", action="store_true", help="Serve deterministic mock completions.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-dtype", default="float32")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-num-scheduled-tokens", type=int, default=64)
    parser.add_argument("--max-num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-pending", type=int, default=256)
    parser.add_argument("--no-safetensors", action="store_true")
    args = parser.parse_args()
    runner: ModelRunner
    max_model_len = 1024
    if args.mock:
        runner = MockModelRunner()
    else:
        from tiny_vllm.torch_gpt2 import TorchGPT2ModelRunner

        runner = TorchGPT2ModelRunner.from_pretrained(
            args.model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            block_size=args.block_size,
            use_safetensors=False if args.no_safetensors else None,
        )
        max_model_len = runner.model.config.max_position_embeddings
    config = EngineConfig(
        model_name=args.model,
        max_batch_size=args.max_batch_size,
        max_num_scheduled_tokens=args.max_num_scheduled_tokens,
        max_num_blocks=args.max_num_blocks,
        block_size=args.block_size,
    )
    service = EngineService(
        config, runner, max_model_len=max_model_len, max_pending=args.max_pending
    )
    web.run_app(create_app(service), host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
