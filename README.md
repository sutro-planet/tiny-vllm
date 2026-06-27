# tiny-vLLM

`tiny-vllm` is a learning-oriented LLM inference framework. The goal is to build the important pieces directly and keep the code easy to inspect: request lifecycle, scheduling, prefill/decode boundaries, KV cache management, observability, benchmarking, and correctness checks.

The initial scaffold is CPU-only and deterministic. It uses a mock model runner so the request, scheduler, and KV cache boundaries can stabilize before GPU kernels or real model loading are added.

## Current Scope

- Single-node, single-GPU target over time.
- One clear implementation path; avoid permanent compatibility layers.
- Deterministic smoke tests before GPU work.
- Benchmarks should compare against vanilla vLLM when real model execution exists.

## Quick Start

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
python3 scripts/smoke.py
```

## Layout

- `src/tiny_vllm/config.py`: engine configuration and validation.
- `src/tiny_vllm/request.py`: request and output data objects.
- `src/tiny_vllm/scheduler.py`: FIFO request batching.
- `src/tiny_vllm/kv_cache.py`: deterministic KV block allocation.
- `src/tiny_vllm/model_runner.py`: mock generation boundary.
- `src/tiny_vllm/engine.py`: request lifecycle orchestration.
- `tests/`: CPU-only behavior tests.
- `benchmarks/`: placeholder benchmark entry points.
- `scripts/`: local smoke-test commands.
