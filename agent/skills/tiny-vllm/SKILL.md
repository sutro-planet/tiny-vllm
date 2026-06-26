---
name: tiny-vllm
description: Guide tiny-vllm development as a learning-oriented modern LLM inference framework. Use when working in this repo on architecture, implementation, benchmarking, scheduling, continuous batching, KV cache management, paged attention, GPU kernels, research evaluation, ablation studies, or comparisons against vLLM.
---

# Tiny vLLM

## Project Intent

Build `tiny-vllm` to learn how modern LLM inference frameworks work by implementing the important pieces directly, while keeping the code simple, clean, and understandable.

Target a full-featured inference-framework shape over time: request lifecycle, tokenizer/model loading boundaries, prefill and decode execution, scheduling, continuous batching, KV cache management, block/page allocation, paged attention, observability, benchmarking, and correctness checks.

Prefer one clear implementation path. Avoid permanent configuration toggles, compatibility layers, or parallel implementations unless they are needed temporarily for an ablation. After measuring, keep the best path and remove the losing path.

## Default Scope

- Optimize first for a single-node, single-GPU learning system.
- Use the available smoke-test GPU host `gpu1.sutroplanet.com`, assumed to provide one RTX 5090-class GPU.
- Assume one autoregressive model family at first. Use Gemma 4 30B as the default exemplar from the project brief unless the repo or available model artifacts establish a different target.
- Specialize for one GPU architecture when that simplifies kernels, memory layout, or scheduling. Do not generalize early for portability.
- Favor readable structure and explicit data flow over clever abstractions. Add abstraction only when it reduces real repeated complexity.

## Research And Design Workflow

For performance-sensitive work, first check current research and production practice. Search papers, vLLM, SGLang, TensorRT-LLM, FlashInfer, CUDA kernels, and relevant vendor/library docs when the topic could have moved recently.

Before implementing a new optimization, pitch the idea briefly:

- Explain the mechanism and why it may improve latency, throughput, memory efficiency, or implementation clarity.
- State the expected tradeoff, such as complexity, memory overhead, numerical behavior, kernel specialization, or reduced portability.
- Define the ablation that will decide whether to keep it.

Run the smallest meaningful ablation before making an optimization the default. Adopt the measured best path, document the result near the benchmark or decision record if such a convention exists, and remove stale alternatives.

## Benchmarking Discipline

Always compare against vanilla vLLM when reasonable. Use the same model, precision, prompt/decode shape, concurrency, and GPU host. If vLLM is not feasible for the exact setup, use the closest reasonable baseline and state the difference.

Choose benchmark workloads that expose the behavior being changed:

- Scheduler or batching work: vary request arrival pattern, batch pressure, prompt length, decode length, and concurrency.
- KV cache or paged attention work: include long-context and fragmentation-heavy workloads.
- Kernel work: measure isolated kernel timing when possible, then end-to-end tokens/sec and latency impact.
- Correctness-sensitive work: compare generated tokens, logits where practical, cache layout invariants, and request completion behavior.

Report enough context for a future agent to reproduce the result: git state if available, host, GPU, model, precision, workload, baseline command, tiny-vllm command, and key metrics.

## Implementation Biases

- Keep modules small and names direct.
- Make data ownership explicit, especially for requests, sequences, KV blocks, and scheduler queues.
- Treat observability as part of the framework: expose enough counters and traces to explain batching, cache allocation, evictions, stalls, and throughput.
- Prefer deterministic smoke tests on the 5090 before broadening scope.
- Keep APIs internal and minimal until repeated use proves the shape.
- Do not preserve unused options for hypothetical future models, GPUs, or serving modes.
