# Unified Token-Step Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move tiny-vLLM from whole-request generation to vLLM-shaped token-step execution: one scheduler/model-runner execution per engine step, carrying both active decode work and newly admitted prefill work in the same batch.

**Architecture:** Requests are tokenized at admission into `SequenceState` objects. The engine schedules active sequences first, then preempted/waiting sequences and newly admitted requests, and builds one `ExecutionBatch` for the step. `ExecutionBatch.num_scheduled_tokens` records how many input tokens each row contributes to this step. The engine uses `num_computed_tokens` plus `num_scheduled_tokens` to decide whether the row reached the current end of the request and should append the sampled token.

This mirrors vLLM's control-flow shape: scheduling produces per-request scheduled-token counts, the model runner prepares one mixed input, and attention metadata can infer prompt/decode spans from token positions. Prefill and decode remain logically different, but they are not separate public runner calls.

## File Structure

- `src/tiny_vllm/sequence.py`: sequence lifecycle state and prompt/generated token accounting.
- `src/tiny_vllm/model_runner.py`: `ExecutionBatch`, `ModelRunnerOutput`, `ModelRunner.execute()`, and tokenizer/detokenizer boundaries.
- `src/tiny_vllm/engine.py`: one mixed execution call per step, decode-first ordering, output completion, and failure cleanup.
- `src/tiny_vllm/scheduler.py`: FIFO request admission.
- `src/tiny_vllm/kv_cache.py`: deterministic block ownership and release on completion.
- `tests/test_model_runner.py`: mock and fake Transformers unified execution coverage.
- `tests/test_engine.py`: prefill-only completion, decode-after-prefill, mixed decode+prefill batching, capacity handling, and failure cleanup.
- `tests/test_transformers_integration.py`: optional real-model smoke aligned with `execute()`.
- `README.md` and `docs/vllm-gap-map.md`: document the unified step model and remaining gaps.

## Design Notes

- A token step is one engine iteration that schedules some number of input tokens for each selected request.
- Decode rows usually schedule one input token: the latest generated token.
- Prefill rows schedule prompt tokens. In this scaffold, a prefill row returns the first generated token immediately; future chunked prefill may schedule only part of a prompt and suppress sampling for non-final chunks.
- The execution batch is a request-major structure today, not a real tensor. A later GPU runner should flatten scheduled tokens into one token tensor plus metadata such as request boundaries, positions, block tables, and logits indices.
- The public model-runner API must stay unified. Callers should not invoke separate `prefill()` and `decode()` methods.
- The first Transformers runner may replay full context internally for correctness. Efficient `past_key_values` reuse is future work.
- This phase should not implement paged attention, page tables, custom CUDA kernels, or efficient cache reuse.

## Task 1: Add Sequence State

- [x] Create `SequenceState`.
- [x] Track prompt token IDs, generated token IDs, token budget, latest token, and completion state.
- [x] Add focused sequence lifecycle tests, including empty-token failure clarity for `latest_token_id`.

## Task 2: Add Unified Runner Contract

- [x] Add `ExecutionBatch` with ordered `sequences` and aligned `num_scheduled_tokens`.
- [x] Add `ModelRunnerOutput` with sampled token IDs aligned to `ExecutionBatch.sequences`.
- [x] Replace public `prefill()` and `decode()` runner methods with `execute(batch)`.
- [x] Update `MockModelRunner` to return one deterministic sampled token per scheduled sequence.
- [x] Update `TransformersModelRunner` to use the unified public API.

## Task 3: Add Mixed Engine Step

- [x] Snapshot active decoding sequences before admission.
- [x] Admit queued requests into prefill slots using existing capacity checks.
- [x] Build one `ExecutionBatch` ordered decode rows first, prefill rows second.
- [x] Call `model_runner.execute()` at most once per non-empty engine step.
- [x] Append returned sampled tokens only when the scheduled range reaches the current request end.
- [x] Complete finished sequences and release KV blocks.
- [x] On runner failure, release active sequence blocks and finish scheduler ownership.

## Task 4: Test Behavior

- [x] Verify mock runner executes prefill and decode rows through the same public method.
- [x] Verify Transformers runner fake model path uses `execute()`.
- [x] Verify a single engine step can carry both an existing decode row and a newly admitted prefill row.
- [x] Verify runner failure cleanup for prompt and generated-token work.
- [x] Keep optional real Transformers smoke test aligned with `execute()`.

## Task 5: Document Gaps

- [x] Update README to describe the unified execution-batch model.
- [x] Update vLLM gap map to record that vLLM schedules per-request token counts and lets runner/attention metadata distinguish prefill from decode.
- [ ] Future: build real flattened tensor inputs and positions from `ExecutionBatch`.
- [ ] Future: add runner-owned KV cache handles and `past_key_values`/page-table semantics.
- [x] Add chunked prefill where a request may schedule multiple prompt chunks before sampling.
- [ ] Future: add fairness policy and backpressure beyond FIFO capacity.
