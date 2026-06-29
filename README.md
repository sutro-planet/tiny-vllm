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
- The first Transformers path may replay full context internally; efficient `past_key_values`, KV page tables, and paged attention remain future work.
- Benchmarks should compare against vanilla vLLM when real model execution exists.

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

## Layout

- `src/tiny_vllm/config.py`: engine configuration and validation.
- `src/tiny_vllm/tokenizer.py`: tokenizer wrapper boundary.
- `src/tiny_vllm/request.py`: request and output data objects.
- `src/tiny_vllm/sequence.py`: active sequence lifecycle and token accounting.
- `src/tiny_vllm/scheduler.py`: FIFO request batching.
- `src/tiny_vllm/kv_cache.py`: deterministic KV block allocation.
- `src/tiny_vllm/model_runner.py`: mock and optional Transformers execution-batch boundary.
- `src/tiny_vllm/engine.py`: request lifecycle orchestration.
- `tests/`: CPU-only behavior tests.
- `benchmarks/`: placeholder benchmark entry points.
- `scripts/`: local smoke-test commands.
- `docs/vllm-gap-map.md`: living vLLM gap and exploration state map.
- `agent/skills/tiny-vllm/SKILL.md`: repo-specific Codex workflow and PR discipline.
