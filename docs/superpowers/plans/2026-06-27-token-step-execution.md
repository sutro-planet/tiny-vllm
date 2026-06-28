# Unified Token-Step Execution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move tiny-vLLM from whole-request generation to vLLM-shaped token-step execution: one scheduler/model-runner execution per engine step, carrying both active decode work and newly admitted prefill work in the same batch.

**Architecture:** Requests are tokenized at admission into `SequenceState` objects. The engine snapshots active decoding sequences, admits new prefill sequences into available slots, and builds one `ExecutionBatch` ordered as decode rows first and prefill rows second. `ExecutionBatch.num_scheduled_tokens` records how many input tokens each row contributes to this step. The model runner returns one sampled token per scheduled sequence through `ModelRunnerOutput`; the engine appends each sampled token according to that sequence's current phase.

This mirrors vLLM's control-flow shape: scheduling produces per-request scheduled-token counts, the model runner prepares one mixed input, and attention metadata distinguishes prefill from decode internally. Prefill and decode remain logically different, but they are not separate public runner calls.

## File Structure

- `src/tiny_vllm/sequence.py`: sequence lifecycle state, prompt/generated token accounting, and phase transitions.
- `src/tiny_vllm/model_runner.py`: `ExecutionBatch`, `ModelRunnerOutput`, `ModelRunner.execute()`, and tokenizer/detokenizer boundaries.
- `src/tiny_vllm/engine.py`: one mixed execution call per step, decode-first ordering, output completion, and failure cleanup.
- `src/tiny_vllm/scheduler.py`: FIFO request admission.
- `src/tiny_vllm/kv_cache.py`: upfront deterministic block reservation and release on completion.
- `tests/test_model_runner.py`: mock and fake Transformers unified execution coverage.
- `tests/test_engine.py`: prefill-only completion, decode-after-prefill, mixed decode+prefill batching, capacity handling, and failure cleanup.
- `tests/test_transformers_integration.py`: optional real-model smoke aligned with `execute()`.
- `README.md` and `docs/vllm-gap-map.md`: document the unified step model and remaining gaps.

## Design Notes

- A token step is one engine iteration that schedules some number of input tokens for each selected request.
- Decode rows usually schedule one input token: the latest generated token.
- Prefill rows schedule prompt tokens. In this scaffold, a prefill row returns the first generated token immediately; future chunked prefill may schedule only part of a prompt and suppress sampling for non-final chunks.
- The execution batch is a request-major structure today, not a real tensor. A later GPU runner should flatten scheduled tokens into one token tensor plus metadata such as request boundaries, positions, phase flags, block tables, and logits indices.
- The public model-runner API must stay unified. Private helpers may branch on phase, but callers should not invoke separate `prefill()` and `decode()` methods.
- The first Transformers runner may replay full context per sequence with `generate(max_new_tokens=1)` for correctness. Efficient batched tensors and `past_key_values` are future work.
- This phase should not implement paged attention, page tables, custom CUDA kernels, chunked prefill, or efficient cache reuse.

## Task 1: Add Sequence State

- [x] Create `SequencePhase` and `SequenceState`.
- [x] Track prompt token IDs, generated token IDs, token budget, latest token, and completion state.
- [x] Add focused sequence lifecycle tests, including empty-token failure clarity for `latest_token_id`.

## Task 2: Add Unified Runner Contract

- [x] Add `ExecutionBatch` with ordered `sequences` and aligned `num_scheduled_tokens`.
- [x] Add `ModelRunnerOutput` with sampled token IDs aligned to `ExecutionBatch.sequences`.
- [x] Replace public `prefill()` and `decode()` runner methods with `execute(batch)`.
- [x] Update `MockModelRunner` to return one deterministic sampled token per scheduled sequence.
- [x] Update `TransformersModelRunner` to use the unified public API while keeping simple internal per-sequence generation.

## Task 3: Add Mixed Engine Step

- [x] Snapshot active decoding sequences before admission.
- [x] Admit queued requests into prefill slots using existing capacity checks.
- [x] Build one `ExecutionBatch` ordered decode rows first, prefill rows second.
- [x] Call `model_runner.execute()` at most once per non-empty engine step.
- [x] Append returned sampled tokens according to each sequence's phase.
- [x] Complete finished sequences and release KV reservations.
- [x] On runner failure, release active sequence reservations and finish scheduler ownership.

## Task 4: Test Behavior

- [x] Verify mock runner executes prefill and decode rows through the same public method.
- [x] Verify Transformers runner fake model path uses `execute()`.
- [x] Verify a single engine step can carry both an existing decode row and a newly admitted prefill row.
- [x] Verify runner failure cleanup for both prefill and decode phases.
- [x] Keep optional real Transformers smoke test aligned with `execute()`.

## Task 5: Document Gaps

- [x] Update README to describe the unified execution-batch model.
- [x] Update vLLM gap map to record that vLLM schedules per-request token counts and lets runner/attention metadata distinguish prefill from decode.
- [ ] Future: build real flattened tensor inputs and positions from `ExecutionBatch`.
- [ ] Future: add runner-owned KV cache handles and `past_key_values`/page-table semantics.
- [ ] Future: add chunked prefill where a prefill request may schedule multiple prompt chunks before sampling.
- [ ] Future: add scheduler token budgets, fairness policy, and backpressure beyond FIFO capacity.
