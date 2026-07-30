# tiny-vLLM / vLLM Gap Map

This is the living map of feature gaps, implementation ideas, and research threads discovered while comparing `tiny-vllm` with vLLM. Keep it current whenever a vLLM area is researched, implemented, rejected, or deferred.

Status legend:

- `[explored]` Source has been inspected and concrete gaps are recorded.
- `[candidate]` Worth implementing or testing in tiny-vLLM.
- `[deferred]` Useful later, but not needed for the current learning path.
- `[not-now]` Intentionally out of scope until project goals change.
- `[unexplored]` Known framework area that still needs a vLLM comparison pass.

Update rule:

- Add new findings under the closest existing node.
- Include local vLLM source references when possible.
- Record the tiny-vLLM implication separately from the vLLM observation.
- Promote a node from `[candidate]` to implementation work only after a small design note or ablation plan exists.

## Gap Tree

- **Request lifecycle and API surface** `[unexplored]`
  - Request admission and validation
  - Prompt forms: text, token IDs, embeddings
  - Completion and chat serving paths
  - Error semantics and cancellation

- **Tokenization and detokenization** `[explored]`
  - **Current tiny-vLLM shape**
    - `TransformersModelRunner.from_pretrained()` calls `transformers.AutoTokenizer.from_pretrained()` directly.
    - `tiny_vllm.tokenizer.Tokenizer` exposes only `encode`, `decode`, and `eos_token_id`.
    - No tokenizer loader options, tokenizer modes, cached tokenizer wrapper, prompt-token-ID input path, or incremental detokenizer exist yet.
  - **vLLM tokenizer registry and modes** `[candidate]`
    - The local vLLM checkout routes tokenizer creation through `~/workspace/vllm/vllm/tokenizers/registry.py`.
    - Its built-in registry includes `hf`, `mistral`, `kimi_audio`, `deepseek_v32`, and `deepseek_v4`; `slow` resolves to `hf` with `use_fast=False`, and plugins can register additional modes.
    - `auto` falls back to `hf` but can select model-specific tokenizers when repo/config clues justify it.
    - tiny-VLLM implication: introduce a small `load_tokenizer()` helper before adding more model-runner paths. Keep only `hf` initially, but make the boundary explicit enough to support modes later.
  - **HF tokenizer wrapper** `[candidate]`
    - vLLM still uses `AutoTokenizer` for the HF path, but wraps it with `get_cached_tokenizer()`.
    - Cached properties include `all_special_ids`, `all_special_tokens`, `get_vocab()`, `len(tokenizer)`, `max_token_id`, and `max_chars_per_token`.
    - tiny-VLLM implication: add a minimal cached wrapper when these properties enter scheduler, logits, or structured-output paths. Avoid broad protocol expansion until callers need it.
  - **Tokenizer loading options and defaults** `[candidate]`
    - vLLM exposes `trust_remote_code`, `tokenizer_revision`, `download_dir`, `tokenizer_mode`, and slow/fast selection.
    - vLLM sets generation truncation to the left and pooling truncation to the right.
    - vLLM improves common custom-tokenizer errors by suggesting `trust_remote_code` or a newer transformers version.
    - tiny-VLLM implication: support `trust_remote_code` and `tokenizer_revision` in the first tokenizer loader. Add truncation policy only when prompt length enforcement exists.
  - **Thread-safe fast tokenizer pool** `[deferred]`
    - vLLM can wrap HF fast tokenizers with deep-copied tokenizer instances behind a small pool for thread-safe public calls.
    - tiny-VLLM implication: defer until the engine has actual concurrent tokenization or request processing threads.
  - **Optional fastokens backend** `[not-now]`
    - vLLM can use `VLLM_USE_FASTOKENS=1` to patch HF fast tokenizers with the `fastokens` backend.
    - tiny-VLLM implication: not useful until tokenizer CPU overhead is measured as a bottleneck.
  - **Model-specific tokenizers** `[deferred]`
    - Mistral uses `mistral_common` for official Mistral/Pixtral/Tekken behavior, chat-template validation, special-token handling, grammar support, and UTF-8 edge cases.
    - The local vLLM checkout has `deepseek_v32` and `deepseek_v4` wrappers that load HF fast tokenizers and override chat-template encoding.
    - Kimi audio has its own tokenizer mode.
    - tiny-VLLM implication: defer until we intentionally support chat templates or those model families. Do not add these modes as unused scaffolding.
  - **Incremental detokenization** `[candidate]`
    - vLLM has fast and slow incremental detokenizers for streaming output.
    - In the local vLLM checkout, the fast path uses `tokenizers.decoders.DecodeStream` for HF fast tokenizers when `tokenizers>=0.22.0`.
    - The slow path tracks token strings, prefix offsets, and read offsets to avoid repeatedly decoding whole output and to handle cleanup algorithms correctly.
    - Stop-string buffering, special-token spacing, invalid prefix recovery, and byte-fallback edge cases are handled in the detokenizer layer.
    - tiny-VLLM implication: implement this before adding streaming responses or step-wise decode output. Until then, full decode of new token IDs is acceptable.
  - **Tokenizer bypass** `[candidate]`
    - vLLM can skip tokenizer and detokenizer initialization when callers provide valid `prompt_token_ids`; generated output then includes token IDs.
    - tiny-VLLM implication: add `prompt_token_ids` to `GenerationRequest` when the scheduler and model runner start operating on token IDs instead of prompt strings.
  - **Tests observed in vLLM** `[candidate]`
    - vLLM tests cached tokenizer behavior, tokenizer registry behavior, Mistral tokenization, and incremental detokenization edge cases.
    - tiny-VLLM implication: when each feature lands, add focused unit tests with fake tokenizers first, then optional transformers integration tests.

- **Model loading and runner boundary** `[candidate]`
  - **Current tiny-vLLM shape**
    - `TorchGPT2ModelRunner.from_pretrained()` loads GPT-2 tokenizer/config/weights from Hugging Face, then executes with `TinyGPT2LMHeadModel`.
    - The local Torch GPT-2 implementation includes token/position embeddings, causal self-attention, MLP, final layer norm, tied LM head, and HF-compatible weight names.
    - `TransformersModelRunner` remains available as an oracle/test boundary; scripts use the Torch runner for GPT-2 smoke.
    - The runner packs one `ExecutionBatch` into flattened scheduled-token tensors: `input_ids`, `positions`, `req_indices`, and `query_start_loc`.
    - The runner owns global per-layer dense KV tensors and a `req_to_blocks` table keyed by request ID. HF-style `past_key_values` remain only for model/oracle tests.
  - **Remaining gaps** `[candidate]`
    - Only GPT-2-style causal LM checkpoints are supported.
    - The runner gathers block-backed KV into dense attention rows internally instead of using paged-attention kernels directly.
    - Weight formats, quantization, model registry, and non-HF loaders are deferred.
  - **Runner cache lifecycle**
    - The engine exposes runner cache release through an optional `release(request_id)` hook.
    - Completion, preemption, and engine failure cleanup release both logical KV blocks and runner-owned block-table entries.

- **Scheduling and continuous batching** `[explored]`
  - Request queues and fairness
  - Token-progress request state inside a unified execution step
  - Batch formation
  - Failure and backpressure behavior
  - **Unified scheduled-token execution** `[explored]`
    - vLLM schedules requests by per-request `num_scheduled_tokens`, not by calling separate public prefill and decode runner methods.
    - tiny-vLLM now mirrors that shape with one `ExecutionBatch` per engine step.
    - Scheduling runs active sequences first, then preempted/waiting sequences and newly admitted requests, mirroring vLLM's running-then-waiting queue shape.
    - Per-request `num_computed_tokens` and `num_scheduled_tokens` define the token-position range for the step; prompt chunks only sample when that range reaches the request's current end.
    - Current Torch GPT-2 implementation flattens scheduled tokens into one model forward and uses `req_indices`/`query_start_loc`/block tables to preserve request boundaries.
  - **Continuous batching scaffold** `[candidate]`
    - The engine keeps queued requests and active sequences separate.
    - Finished sequences release their KV blocks, allowing later steps to admit queued requests into freed slots.
    - Backpressure is still capacity-driven and FIFO; fairness and latency policy are deferred.

- **KV cache and block management** `[candidate]`
  - Block allocator behavior
  - Prefix cache
  - Eviction and fragmentation
  - Swap/offload policy
  - **Current KV behavior** `[candidate]`
    - KV blocks grow by scheduled-token demand: prompt/recompute chunks may reserve multiple slots, while normal decode usually reserves one additional computed slot per step.
    - The allocator only appends whole blocks when a request crosses a block boundary, then releases all owned blocks on completion.
    - Torch GPT-2 runner stores global per-layer dense KV tensors addressed by a request-to-block table.
    - Fragmentation-aware physical block management, page-table GPU kernels, cache handles, and eviction are deferred.

- **Paged attention and GPU kernels** `[unexplored]`
  - Page table layout
  - Kernel API shape
  - Flash attention integration
  - Kernel correctness and profiling harnesses

- **Sampling and logits processing** `[unexplored]`
  - Temperature, top-k, top-p, repetition penalties
  - Stop conditions
  - Logprobs
  - Structured output constraints

- **Streaming output** `[unexplored]`
  - Per-step output assembly
  - Delta vs full text
  - Client-visible stop behavior
  - Interaction with incremental detokenization

- **Observability and debugging** `[unexplored]`
  - Engine stats
  - Request traces
  - Scheduler/KV cache diagnostics
  - GPU profiling hooks

- **Benchmarking and vLLM parity checks** `[unexplored]`
  - Shared workloads
  - Vanilla vLLM baseline commands
  - tiny-vLLM comparison commands
  - Reproducibility metadata

- **Distributed and multi-GPU execution** `[not-now]`
  - Tensor parallelism
  - Pipeline parallelism
  - Data parallel serving
  - Multi-node orchestration
