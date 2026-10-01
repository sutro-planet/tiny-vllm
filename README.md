# tiny-vLLM

`tiny-vllm` is a learning-oriented LLM inference framework. The goal is to build the important pieces directly and keep the code easy to inspect: request lifecycle, scheduling, prefill/decode boundaries, KV cache management, observability, benchmarking, and correctness checks.

The current scaffold is CPU-only and deterministic by default. It tracks each request by token progress (`num_computed_tokens`) and executes one mixed scheduled-token batch per engine step so request state, scheduling, and KV cache ownership can stabilize before GPU kernels or paged attention are added.

## Current Scope

- Single-node, single-GPU target over time.
- One clear implementation path; avoid permanent compatibility layers.
- Deterministic smoke tests before GPU work.
- Engine steps schedule active sequences first, then waiting/preempted work and newly admitted requests into one runner `ExecutionBatch`.
- Per-request `num_scheduled_tokens` caps the token-position range computed in the step; prompt work may be chunked across multiple steps before sampling.
- Request-local admission failures, such as empty encoded prompts or impossible KV requirements, return a `GenerationOutput` with `error` set instead of failing unrelated active requests.
- GPT-2 real-model smoke uses tiny-vLLM's Torch forward path with flattened scheduled-token inputs and block-backed dense KV tensors; Hugging Face is used only for tokenizer/config/weight loading and oracle checks.
- GPU paged attention, fragmentation-aware physical block management, and non-GPT-2 architectures remain future work.
- Benchmarks should compare against vanilla vLLM when real model execution exists.
- A standalone OpenAI-compatible Completions server batches concurrent requests through one Engine.
- AIPerf uses that public endpoint; see [the benchmark workflow](benchmarks/README.md).
- [Kubernetes GPU profiling](benchmarks/kubernetes/README.md) packages CUDA images and compares tiny-vLLM with vLLM on the RTX 5090. The PR `/profile` workflow activates after it is merged into the default branch.

## Exploration State

Use `docs/vllm-gap-map.md` as the living exploration state map for tiny-vLLM versus vLLM. When researching a vLLM subsystem, update the closest node with observed vLLM behavior, tiny-vLLM implications, source pointers, and whether the idea is explored, candidate, deferred, not-now, or still unexplored.

Future Codex agents should also read `agent/skills/tiny-vllm/SKILL.md` before development work in this repo. The skill captures the project workflow: keep the implementation learning-oriented, use the gap map during research, compare against vanilla vLLM when practical, use `gpu1.sutroplanet.com` for GPU smoke/debug loops, keep PRs to a single commit, and proactively monitor Gemini/GitHub PR feedback.

## Quick Start

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 scripts/smoke.py
```

Optional real-model smoke path:

```bash
python3 -m pip install -e ".[dev,transformers]"
python3 scripts/run_tiny_model.py --model sshleifer/tiny-gpt2 --prompt "Hello" --max-new-tokens 4
python3 scripts/run_tiny_model.py --model gpt2 --prompt "Hello" --max-new-tokens 4 --no-safetensors
```

## API Server

Install the optional server and model dependencies, then start a local server:

```bash
python3 -m pip install -e '.[server,transformers]'
python3 -m tiny_vllm.server --model sshleifer/tiny-gpt2 --no-safetensors
```

The installed `tiny-vllm-serve` command is equivalent. Use `--model gpt2` for
GPT-2, `--device cuda --torch-dtype float16` for CUDA, or
`--mock --model mock-gemma` for deterministic transport tests without weights.

```bash
curl http://127.0.0.1:8000/v1/models
curl http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"sshleifer/tiny-gpt2","prompt":"Hello","max_tokens":16,"temperature":0}'
```

It also works with the OpenAI Python client (installed separately):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
result = client.completions.create(
    model="sshleifer/tiny-gpt2", prompt="Hello", max_tokens=16, temperature=0,
)
print(result.choices[0].text)
```

Supported routes: `GET /health`, `GET /v1/models`, and `POST /v1/completions`.
This first API slice supports one text prompt (or one-element string list),
non-streaming greedy generation, `n=1`, and fixed `max_tokens` output length
(default 16, EOS ignored). Omitted temperature defaults to zero. Unsupported
sampling, stop strings, chat completions, and streaming are not implemented;
unsupported completion options return JSON errors rather than being ignored.
Usage counts come from tokenizer input IDs and Engine output token counts.

One process owns one Engine and one model worker thread. Incoming requests join
between engine steps, allowing continuous batching while HTTP stays responsive.
`--max-pending` bounds accepted requests (default 256); excess requests receive
429. Context overflow is rejected per request, and fatal worker failure marks
health unavailable. Accepted work drains on shutdown; disconnected requests
continue to completion because the Engine has no cancellation API yet.
The service preflights prompt tokenization for context checks and usage counts;
Engine tokenizes again on admission. This cost is included in API latency.

Pass `http://127.0.0.1:8000/v1/completions` to AIPerf using the
[benchmark configs](benchmarks/README.md).

## Layout

- `src/tiny_vllm/config.py`: engine configuration and validation.
- `src/tiny_vllm/tokenizer.py`: tokenizer wrapper boundary.
- `src/tiny_vllm/request.py`: request and output data objects.
- `src/tiny_vllm/sequence.py`: active sequence lifecycle and token accounting.
- `src/tiny_vllm/scheduler.py`: FIFO request batching.
- `src/tiny_vllm/kv_cache.py`: deterministic KV block allocation.
- `src/tiny_vllm/model_runner.py`: mock and optional Transformers oracle execution-batch boundary.
- `src/tiny_vllm/torch_gpt2.py`: tiny Torch GPT-2 forward path and block-backed dense KV runner.
- `src/tiny_vllm/engine.py`: request lifecycle orchestration.
- `src/tiny_vllm/server.py`: optional OpenAI-compatible Completions HTTP server.
- `tests/`: CPU behavior and opt-in model integration tests.
- `benchmarks/`: AIPerf workload configs and optional result collection wrapper.
- `scripts/`: local smoke-test commands.
- `docs/vllm-gap-map.md`: living vLLM gap and exploration state map.
- `agent/skills/tiny-vllm/SKILL.md`: repo-specific Codex workflow and PR discipline.
