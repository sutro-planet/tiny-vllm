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

- **Model loading and runner boundary** `[unexplored]`
  - HF model loading options
  - Device and dtype handling
  - Weight formats and quantization
  - Runner protocol shape

- **Scheduling and continuous batching** `[unexplored]`
  - Request queues and fairness
  - Prefill/decode separation
  - Batch formation
  - Failure and backpressure behavior

- **KV cache and block management** `[unexplored]`
  - Block allocator behavior
  - Prefix cache
  - Eviction and fragmentation
  - Swap/offload policy

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
