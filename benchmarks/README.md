# AIPerf benchmarks

AIPerf talks to a normal OpenAI-compatible `/v1/completions` endpoint. The
`tiny_vllm.server` module serves that API independently of AIPerf; the benchmark
client does not import the engine or require private server routes.

## Setup

From the tiny-vLLM checkout:

```bash
uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -e '.[dev,server,transformers]'
uv venv .venv-aiperf --python 3.12
uv pip install --python .venv-aiperf/bin/python -e ~/workspace/aiperf
```

AIPerf uses its native schema-2.0 YAML configuration. The local source references
are `~/workspace/aiperf/docs/tutorials/yaml-config.md` and
`~/workspace/aiperf/docs/tutorials/openai-text-endpoints.md`. These recipes were
checked with AIPerf 0.13.0 at `d66fc0b83`.

## Start a server and pass its endpoint to AIPerf

For local CPU validation:

```bash
.venv/bin/tiny-vllm-serve --model sshleifer/tiny-gpt2 \
  --device cpu --torch-dtype float32 --no-safetensors
```

From another terminal:

```bash
BENCH_URL=http://127.0.0.1:8000/v1/completions \
BENCH_MODEL=sshleifer/tiny-gpt2 BENCH_TOKENIZER=sshleifer/tiny-gpt2 \
BENCH_RUNS=1 BENCH_REQUESTS=8 \
BENCH_ARTIFACTS=artifacts/aiperf/cpu-baseline \
.venv-aiperf/bin/aiperf profile --config benchmarks/aiperf/baseline.yaml
```

This direct AIPerf command is sufficient. Change `BENCH_URL` to point at another
server. For a transport smoke test without weights, start
`tiny-vllm-serve --mock --model mock-gemma` and run:

```bash
.venv-aiperf/bin/aiperf profile --config benchmarks/aiperf/smoke.yaml
```

The smoke config sends two warmup and eight measured requests, each requesting
four output tokens. Mock prompt counts use whitespace tokenization; they are
not GPT-2 performance measurements.

## Workloads

`baseline.yaml` uses seed 42, three repeats, eight warmup requests excluded from
results, and 100 measured requests per cell:

| Cell | Input tokens | Output tokens | Concurrency | Purpose |
|---|---:|---:|---:|---|
| latency_c1 | 128 | 32 | 1 | Single-request latency |
| batch_c2 | 128 | 32 | 2 | Batch scaling |
| batch_c4 | 128 | 32 | 4 | Fill four active engine slots |
| queue_c8 | 128 | 32 | 8 | Queue pressure above engine capacity |
| prefill_512_16 | 512 | 16 | 4 | Chunked prefill |
| decode_32_128 | 32 | 128 | 4 | Decode and block growth |
| mixed_prefill | 32 / 256, equal weights | 32 | 4 | Mixed prompt lengths |

`BENCH_RUNS` and `BENCH_REQUESTS` override repeats and measured request counts.
Small runs validate functionality; their tail latencies are not stable estimates.
Synthetic lengths are targets; check the exported server-reported lengths.

`kv-pressure.yaml` uses 128 input tokens, 64 output tokens, and concurrency four.
Start tiny-vLLM with `--max-num-blocks 32 --block-size 16 --max-batch-size 4`:
each request fits alone, but four requests cannot retain their full KV together.
Use the same workload with 256 blocks as a capacity control. This compares
end-to-end behavior under KV pressure; AIPerf cannot verify internal preemption
counts. Engine and runner still allocate block IDs independently, and the
runner grows tensor storage dynamically. Logical block capacity is not GPU
memory allocation.

## GPU and vLLM comparison

For container builds, Kubernetes deployment on the RTX 5090, and automated
PR `/profile` reports, use [the cluster workflow](kubernetes/README.md).

Use the same model snapshot, precision, GPU, workload, and client location.
Run servers sequentially on `gpu1.sutroplanet.com`; a loopback client avoids
mixing network differences into the baseline. Record the host and driver:

```bash
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
CUDA_VISIBLE_DEVICES=0 .venv/bin/tiny-vllm-serve \
  --model gpt2 --device cuda --torch-dtype float16 \
  --max-batch-size 4 --max-num-scheduled-tokens 64 \
  --max-num-blocks 256 --block-size 16
```

After stopping tiny-vLLM, an eager vLLM comparison with prefix caching disabled:

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve gpt2 \
  --host 127.0.0.1 --port 8001 --dtype float16 \
  --max-model-len 1024 --max-num-seqs 4 --max-num-batched-tokens 64 \
  --enable-chunked-prefill --no-enable-prefix-caching --enforce-eager \
  --block-size 16 --num-gpu-blocks-override 256 --generation-config vllm
```

Run the same YAML with `BENCH_MODEL=gpt2 BENCH_TOKENIZER=gpt2` and the appropriate
`BENCH_URL`. Enable `BENCH_GPU_TELEMETRY=true` when AIPerf runs on the server host.
Both use `temperature=0` and `ignore_eos=true` for fixed output lengths. Equal
logical block counts do not imply equal physical memory use. Report a vLLM
run with its optimized defaults separately.

## Optional reproducibility wrapper

`benchmarks/run_aiperf.py` adds manifests and result checks around the same
AIPerf CLI. It works through the supplied endpoint with no server-specific hooks:

```bash
.venv/bin/python benchmarks/run_aiperf.py \
  --label tiny-vllm --model sshleifer/tiny-gpt2 \
  --url http://127.0.0.1:8000/v1/completions \
  --artifact-dir artifacts/aiperf/tiny-baseline
```

Use a fresh artifact directory. Optionally pass `--server-metadata server.json`
with the server launch command, model revision, dtype, GPU/driver, and version or
commit. Without that file the wrapper records only client/config provenance;
it cannot infer which server binary or GPU is behind an arbitrary endpoint.
The `--label` value is a user-provided label, not verified server identity.

The wrapper saves the input YAML, expanded plan, environment overrides, AIPerf
version/source revision, client package versions, local source hashes, and exit
status. It rejects missing/incomplete/cancelled results, request errors, and
output-length mismatches, including when AIPerf itself exits successfully.
AIPerf exports request JSONL, summary metrics, logs, and repeated-run aggregates;
`summary.json` collects headline metrics. Use immutable model snapshots and a
clean committed server checkout for reproducible performance comparisons.

Compare request latency p50/p95/p99, request throughput, output-token throughput,
errors, and actual sequence lengths. This server currently returns non-streaming
responses, so TTFT and inter-token latency cannot be measured. Derived phase
metrics do not measure internal prefill/decode timing. Internal Torch/CUDA
profiling and streaming are future slices. HF oracle tests remain the model
correctness check.

## Validation: 2026-10-01

The endpoint-only workflow passed locally on macOS/CPU with
`sshleifer/tiny-gpt2`, Torch 2.14.1, Transformers 5.18.0, and AIPerf 0.13.0:

- 82 tests passed; two opt-in model integration tests were skipped. Ruff and
  mypy passed. HTTP tests cover batching, usage, unsupported options, context
  limits, queue overflow, worker failure, responsiveness, and shutdown cleanup.
- OpenAI Python SDK 3.22.1 model discovery, completion, and error handling passed
  against the running real-model server.
- All three YAML configs validated. The native AIPerf CLI completed all seven
  baseline cells with eight measured requests each, zero errors, and matching
  output lengths. The optional wrapper completed the mock smoke test as well.

Artifacts are in the ignored `artifacts/validation/server-endpoint/` directory.
These short dirty-tree runs validate functionality, not GPU performance. GPU
measurements and a live vLLM comparison remain unverified; the target GPU host
was unreachable during the earlier setup probe.
