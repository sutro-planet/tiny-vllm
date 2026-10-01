# tiny-gpt2 on RTX 5090

Source: `fe833e410b34cc53fd28540d4d6ebd5ff72d2115`. Run: `local-20261001-r2`.
GPU: NVIDIA GeForce RTX 5090 on `sutro-gpu1`; CUDA, float16.
Model: `sshleifer/tiny-gpt2` at `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`.

**The paired comparison did not complete. No cross-engine speedup is available.**

Failure detail: vLLM 0.10.2 / PyTorch 2.8 FlexAttention rejected head_dim=1 on the first generation request. The run was interrupted after repeated engine crashes; no vLLM measurements are accepted. The server log reports:

```text
NotImplementedError: embedding dimension of the query, key, and value must be at least 16 but got E=1 and Ev=1
```

## tiny: validated measurements

3 repeats × 100 requests per scenario; warmup excluded. Values are medians of per-run metrics.

| Scenario | p50 ms | p95 ms | p99 ms | Requests/s | Output tok/s |
|---|---:|---:|---:|---:|---:|
| batch_c2 | 47.48 | 53.61 | 54.44 | 40.64 | 1300.48 |
| batch_c4 | 56.26 | 58.42 | 63.80 | 70.01 | 2240.27 |
| decode_32_128 | 199.61 | 202.87 | 204.66 | 20.01 | 2560.76 |
| latency_c1 | 39.54 | 42.15 | 44.27 | 24.88 | 796.13 |
| mixed_prefill | 56.88 | 59.57 | 64.62 | 69.01 | 2208.42 |
| prefill_512_16 | 54.58 | 57.93 | 67.27 | 71.59 | 1145.40 |
| queue_c8 | 112.72 | 116.81 | 124.60 | 70.15 | 2244.88 |

vllm: no complete, validated measurement matrix.

The adjacent `tiny-gpt2-evidence.json` retains the run manifest, environment and validated metrics. Local raw exports are retained under `artifacts/gpu-profile/local-20261001-r2/`. Non-streaming latency does not expose TTFT/ITL. Tiny GPT-2 primarily measures serving and kernel-launch overhead.

Latency/batch/queue cells use 128 input and 32 output tokens. Prefill uses 512/16; decode uses 32/128. Mixed prefill uses equally weighted 32/256-token prompts and 32-token outputs. Generation is greedy and fixed-length. AIPerf 0.13.0, seed 42, eight excluded warmup requests per scenario before the first trial; AIPerf skips warmup in subsequent trials. Both servers use four active requests, a 64-token scheduling budget, and 256 blocks of 16 tokens. The node also hosts unrelated CPU workloads.

The serving source is identical to the source consolidated into the CUDA profiling PR; no inference-loop optimization was made for this measurement. vLLM 0.10.2 ran from its official digest, with FlexAttention and enforce-eager. Tiny used Torch 2.8.0+cu128, Transformers 4.56.2 and float16.

The paired PR benchmark uses standard GPT-2 (124M) because the pinned vLLM supports its head size. No model padding or upstream source modifications are used.

The manual measurement used uncommitted orchestration changes; the evidence manifest records that harness state. The serving source itself matches the recorded commit.
