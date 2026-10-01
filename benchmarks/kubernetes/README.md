# RTX 5090 PR profiling

This workflow builds tiny-vLLM into a CUDA image, runs it and an upstream vLLM
image sequentially on Kubernetes node `sutro-gpu1`, and sends the same AIPerf
workload to their separate Completions endpoints. It generates `report.md`,
`comparison.json`, a provenance manifest, server environment/logs, and raw
AIPerf exports.

## Environment

- Namespace: `tiny-vllm-profile`; NVIDIA runtime class: `nvidia`.
- Node: `sutro-gpu1`, RTX 5090 (32 GB), driver 575.57.08 at initial inspection.
- CUDA 12.8 / PyTorch 2.8.0; image digests are pinned in `docker/` and `profile.json`.
- Default paired model: `openai-community/gpt2` (124M), revision
  `607a30d783dfa663caf39e06633721c8d4cfcd7e`, baked at `/models/gpt2`
  in the tiny-vLLM image; AIPerf has the matching tokenizer. The official vLLM
  image downloads that same revision at startup and uses the same served name.
- vLLM 0.10.2, eager mode, FlashAttention, no prefix caching. The upstream
  image and source are unchanged.
- `sshleifer/tiny-gpt2` at `5f91d94bd9cd7190a9f3216ff93cd1dd95f2c7be`
  is also baked into the images for smoke testing. On the RTX 5090, tiny-vLLM
  runs it successfully, but the pinned vLLM cannot generate with its head_dim=1:
  FlexAttention/PyTorch requires at least 16, and the other GPU backends also
  reject that head size. `--config benchmarks/kubernetes/tiny-gpt2.json`
  reproduces that diagnostic and retains tiny results in an incomplete report.
  It is deliberately not the default paired benchmark; no weights or dimensions
  are padded to manufacture a baseline.
- AIPerf 0.13.0, seed 42, three repeats and 100 measured requests per scenario.
  Eight warmup requests per scenario run before the first trial and are
  excluded from metrics; subsequent trials reuse the warm server. See
  `profile.json` and `../aiperf/baseline.yaml` for the exact workload.

Every server requests one GPU from Kubernetes. The workflow never scales or
stops another service, changes GPU sharing, or exposes devices without scheduler
allocation. If the GPU is occupied, pods wait until it becomes available or the
30-minute readiness timeout expires. The user coordinates GPU availability.
Other applications may still use node CPU; report that limitation.

## Local execution

Provision the namespace and namespace-scoped automation identity once:

```bash
kubectl apply -f benchmarks/kubernetes/bootstrap.yaml
```

Build for the node's architecture and push to the project registry:

```bash
docker buildx build --platform linux/amd64 -f docker/Dockerfile \
  --build-arg SOURCE_REVISION=<tested-sha> \
  -t ghcr.io/sutro-planet/tiny-vllm:<unique-tag> --push .
docker buildx build --platform linux/amd64 -f docker/Dockerfile.aiperf \
  -t ghcr.io/sutro-planet/tiny-vllm-aiperf:<unique-tag> --push .
```

An in-cluster rootless BuildKit builder can avoid downloading large CUDA layers
to an ARM laptop; use an explicitly named builder and remove it after building.
A registry pull Secret in the profiling namespace is needed for private images.
Do not put registry credentials in Dockerfiles, build arguments or committed files.

Run with Python 3.12 and kubectl, replacing image references with pushed digests.
The vLLM image defaults to the official digest in `profile.json`; no custom vLLM
build or registry copy is required:

```bash
python -m benchmarks.kubernetes.run \
  --tiny-image ghcr.io/sutro-planet/tiny-vllm@sha256:<digest> \
  --client-image ghcr.io/sutro-planet/tiny-vllm-aiperf@sha256:<digest> \
  --source-revision <tested-sha> --run-id local-<unique-id> \
  --pull-secret <registry-secret> --output artifacts/gpu-profile/<unique-id>
```

The orchestrator creates one endpoint and an AIPerf client pod, verifies CUDA and
the GPU name, executes a real prefill/decode probe, measures all scenarios,
retrieves results, and releases that server before starting the second. Services have distinct cluster DNS names recorded
in the manifest. Both servers and the client use the same node to avoid WAN
latency. Model/client pods do not mount Kubernetes API credentials.

Resources are labeled with the run ID and deleted in a `finally` block. After
an abrupt client-machine shutdown, remove only the abandoned run's resources:

```bash
kubectl delete deployment,service,pod -n tiny-vllm-profile -l profile-run=<run-id>
```

`--runs 1 --requests 8` is a functional smoke run, not a stable performance
estimate. The standard report uses three repeats × 100 requests × seven cells
for each engine. Percentiles are medians of per-run percentiles, not pooled
percentiles. Raw artifacts retain every run. Failed/incomplete exports or
mismatched token lengths fail the comparison. Server restarts/unhealthy states
also invalidate measurements. Failure reports retain only complete, independently
validated engine matrices and never compute a speedup against failed requests.

## `/profile` on a PR

The workflow `.github/workflows/profile.yml` must be merged into `main` first;
GitHub dispatches `issue_comment` workflows from the default branch. A PR alone
does not activate a new comment-triggered workflow.

Prerequisites:

1. The `sutro-k8s-runner-set` organization runner must be available to this repo.
   It needs Docker for builds and network access to the Kubernetes API. CUDA
   images are built there to avoid GitHub-hosted runner disk limits.
2. Repository secret `PROFILE_KUBECONFIG` authenticates as the `profiler` service
   account from `bootstrap.yaml`. Its Role is limited to this namespace and
   cannot access Secrets or deployments in other namespaces. The credential must
   be rotated/revoked by the operator; never use an administrator kubeconfig.
3. The workflow creates `ghcr.io/sutro-planet/tiny-vllm/profile-server` and
   `ghcr.io/sutro-planet/tiny-vllm/profile-client` with its repository Actions
   token, linking the packages to this repository on first publication. Keep
   these names reserved for Actions: manually published private packages may
   lack repository access even when they have a source label. If reusing an
   existing package, grant this repository write access in Manage Actions Access
   first. Registry credentials supplied to pods are short-lived per-run Secrets,
   removed during cleanup.

A user with current write/maintain/admin repository permission posts exactly:

```text
/profile
```

The workflow records the open PR's current head SHA, builds candidate package
sources using Docker definitions from the trusted default branch, and runs the
trusted orchestrator. It posts the report and a link to 30-day raw artifacts on
the requesting PR. Only same-repository PRs are supported initially. A PR update
after the trigger does not change the tested revision. Unauthorized requests do
not build images or access the cluster. Manual dispatch with a PR number uses
the same permission check.

The runner with cluster credentials checks out only the default branch. Candidate
code runs inside its model container without the service-account token or host
mounts. Keep this separation when extending the workflow.

## Interpretation

This is a controlled eager vLLM baseline, not its most optimized configuration.
Both engines use the same logical KV budget but different physical allocation
and attention implementations. GPT-2 124M provides a supported paired benchmark;
tiny-gpt2 primarily exposes service/scheduler/kernel-launch overhead. Neither
predicts larger-model throughput. Non-streaming API latency includes the service
loop and tokenization; it cannot expose TTFT, inter-token latency, or CUDA kernel timing. Internal GPU
traces are a separate diagnostic slice.

References: [vLLM FlexAttention](https://docs.vllm.ai/en/v0.10.2/api/vllm/v1/attention/backends/flex_attention.html),
[GitHub comment workflow events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#issue_comment).

## Measured reports

See [tiny-gpt2 on the RTX 5090](../reports/2026-10-01-rtx5090/tiny-gpt2.md)
for the first CUDA measurements and the upstream compatibility finding, and
[the GPT-2 paired report](../reports/2026-10-01-rtx5090/gpt2.md) for the
validated 42-run tiny-vLLM/vLLM comparison.
