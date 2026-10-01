# tiny-vLLM / vLLM GPU profile

Source: `fe833e410b34cc53fd28540d4d6ebd5ff72d2115`. Run: `gpt2-20261001-r2`.
GPU: NVIDIA GeForce RTX 5090 on `sutro-gpu1`; CUDA, float16.
Model: `openai-community/gpt2` at `607a30d783dfa663caf39e06633721c8d4cfcd7e`.
3 repeats × 100 measured requests per scenario; warmup excluded. Values are medians across runs; percentile columns are medians of per-run percentiles, not pooled percentiles.

| Scenario | tiny p50 ms | vLLM p50 ms | tiny p95 ms | vLLM p95 ms | tiny output tok/s | vLLM output tok/s | vLLM / tiny tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| batch_c2 | 231.41 | 76.10 | 239.81 | 82.50 | 274.06 | 822.89 | 3.00× |
| batch_c4 | 268.47 | 78.76 | 280.89 | 84.33 | 469.89 | 1590.55 | 3.38× |
| decode_32_128 | 983.50 | 278.55 | 1036.59 | 287.03 | 516.90 | 1827.91 | 3.54× |
| latency_c1 | 196.20 | 72.44 | 223.89 | 75.67 | 159.88 | 434.27 | 2.72× |
| mixed_prefill | 272.24 | 79.08 | 286.74 | 83.48 | 463.73 | 1589.62 | 3.43× |
| prefill_512_16 | 255.23 | 77.07 | 269.22 | 82.28 | 244.06 | 799.80 | 3.28× |
| queue_c8 | 535.25 | 150.60 | 563.41 | 156.73 | 471.85 | 1671.81 | 3.54× |

Both servers ran sequentially on the same GPU. All measured requests succeeded with matching input/output lengths. Generation was greedy, fixed-length and non-streaming. TTFT and inter-token latency are unavailable.

vLLM 0.10.2 used FLASH_ATTN, eager mode, and disabled prefix caching. Both used 4 active requests, a 64-token step budget and 256 logical blocks of 16 tokens. This is a controlled eager baseline, not a comparison against vLLM's fastest default configuration.

This GPT-2 workload measures one model and serving configuration, including scheduling and kernel-launch overhead. It does not predict larger-model throughput. The node also hosts other CPU workloads; the GPU resource was exclusively reserved by each benchmark server.

## Images and environment

- tiny: `ghcr.io/sutro-planet/tiny-vllm@sha256:8756e12039c73e44903856202973a19c380fe0ee864daa95f37fe87476ff1c1c`; endpoint `http://tiny-gpt2-20261001-r2.tiny-vllm-profile.svc:8000/v1/completions`.
- vllm: `docker.io/vllm/vllm-openai@sha256:df2607b26bdda2875de4832f4d08da0055b4b6e3570347f3a849bcc652771dd6`; endpoint `http://vllm-gpt2-20261001-r2.tiny-vllm-profile.svc:8000/v1/completions`.

The adjacent `gpt2-evidence.json` retains the manifest, environment, generation probes and validated metrics. Raw exports and server logs are retained locally under `artifacts/gpu-profile/gpt2-20261001-r2/`. The listed endpoints were ephemeral and have been cleaned up.

## Measurement notes

Latency/batch/queue cells use 128 input and 32 output tokens. Prefill uses 512/16; decode uses 32/128. Mixed prefill uses equally weighted 32/256-token prompts and 32-token outputs. AIPerf 0.13.0, seed 42, eight excluded warmup requests per scenario before the first trial; later trials reuse the warm server.

Both CUDA generation probes produced `HelloHelloHelloHello` for 32 repeated `Hello` tokens, with four output tokens. This is a functional parity probe, not a comprehensive model-quality test.

The manual run used uncommitted harness edits later included in this PR; the manifest records the original harness revision and dirty status. The tested serving source at `fe833e410b34cc53fd28540d4d6ebd5ff72d2115` is identical to this PR's `src` tree (`0f168b16f26dbbd3f81a4d23d87e7b9a91486db9`). No inference-loop optimization was introduced for these measurements.

See [the separate tiny-gpt2 report](tiny-gpt2.md) for the originally requested toy-model numbers and the upstream head-size incompatibility. GitHub `/profile` event integration remains pending default-branch activation; this report validates the manual Kubernetes orchestration, not a completed Actions run.
