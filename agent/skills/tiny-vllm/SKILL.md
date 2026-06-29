---
name: tiny-vllm
description: Guide tiny-vllm development as a learning-oriented modern LLM inference framework. Use when working in this repo on architecture, implementation, benchmarking, scheduling, continuous batching, KV cache management, paged attention, GPU kernels, GPU development/debug setup on gpu1.sutroplanet.com, research evaluation, ablation studies, comparisons against vLLM, PR single-commit hygiene, or addressing Gemini/GitHub PR review feedback.
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

## GPU Development Environment

Use `gpu1.sutroplanet.com` as the canonical remote smoke-test and debugging host. Treat the local repo and GitHub branch as the source of truth; use the remote checkout to reproduce, profile, and debug GPU behavior.

Prefer the repo-owned VS Code tasks for routine GPU smoke checks instead of manual SSH. Use `Tiny vLLM: GPU Smoke Current Branch` for reproducible PR validation after committing and pushing; use `Tiny vLLM: GPU Smoke Dirty Workspace` only for short-lived scratch debugging. The tasks call `scripts/dev/gpu_smoke.sh`, which also supports `gpt2` variants for model smoke testing.

First probe the host and record the result in any benchmark or debug report:

```bash
ssh gpu1.sutroplanet.com 'hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader; python3 --version'
```

Default setup on `gpu1`:

```bash
ssh gpu1.sutroplanet.com '
set -e
mkdir -p ~/workspace
if [ -d ~/workspace/tiny-vllm/.git ]; then
  cd ~/workspace/tiny-vllm
  git fetch origin
else
  git clone git@github.com:sutro-planet/tiny-vllm.git ~/workspace/tiny-vllm
  cd ~/workspace/tiny-vllm
fi
git checkout <branch>
git pull --ff-only
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m ruff check .
python -m mypy src tests
PYTHONPATH=src python scripts/smoke.py
'
```

Prefer commit-and-push, then `git pull --ff-only` on `gpu1`, for reproducible work. Use `rsync` only for short-lived uncommitted debugging, and never report benchmark numbers from an rsynced dirty tree without saying so.

For future Torch/vLLM GPU work, check Python compatibility before installing heavy dependencies. The host may expose a newer system Python than PyTorch or vLLM supports; if so, use a project-local Python 3.11/3.12 environment or a CUDA/PyTorch container rather than forcing packages into the scaffold venv.

Remote debug loop:

- Run local CPU tests first; use `gpu1` for CUDA behavior, memory pressure, kernel timing, and vLLM comparisons.
- Use `CUDA_VISIBLE_DEVICES=0` to keep runs single-GPU.
- For correctness bugs, rerun with `CUDA_LAUNCH_BLOCKING=1` and the smallest failing test or script.
- For memory/debug reports, capture `nvidia-smi`, git commit, command, model, precision, prompt/decode shape, and relevant logs.
- For performance changes, collect a vanilla vLLM baseline and a tiny-vLLM run on the same remote checkout and GPU.

## Research And Design Workflow

For performance-sensitive work, first check current research and production practice. Search papers, vLLM, SGLang, TensorRT-LLM, FlashInfer, CUDA kernels, and relevant vendor/library docs when the topic could have moved recently.

Use `docs/vllm-gap-map.md` as the living comparison map between tiny-vLLM and vLLM. When researching a vLLM subsystem, read that document first, update the relevant node with findings and source pointers, and keep implementation candidates separate from deferred or not-now ideas.

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

## PR Review Discipline

When a branch has an open PR, proactively inspect GitHub PR comments and review threads before finalizing, especially comments from Gemini Code Assist. Treat unresolved actionable Gemini feedback as part of the active task even if the user did not explicitly ask for each comment.

Gemini comments often appear a few minutes after a PR is opened or updated. After creating or pushing to a PR, run an immediate thread-aware check, then monitor again after a short delay before concluding that Gemini has no feedback.

Evaluate each suggestion technically before changing code. Implement comments that are correct for this codebase, add or update focused tests for behavior changes, and push a follow-up commit to the PR. If a Gemini suggestion is wrong, stale, ambiguous, or conflicts with the project scope, call that out with concise technical reasoning instead of applying it blindly.

Do not mark GitHub threads resolved or reply in the PR unless the user explicitly asks for that write action. Local code changes and pushed commits are the default response.

Always open PRs against `main`. Do not create stacked PRs whose base is another feature branch; if work depends on unmerged feature-branch changes, first consolidate those changes onto a new single-commit branch from `main` or get the prerequisite branch merged to `main`. Before opening any PR, verify the base branch is `main`, for example with `gh pr create --base main ...`.

Keep each PR to a single commit on top of the target branch. During review, amend or squash local changes into that commit instead of stacking fixup commits. Before rewriting a published PR branch, fetch the remote, verify the expected upstream branch, and push with `git push --force-with-lease`, not plain `--force`. If the branch contains commits by another author or unexpected remote changes, stop and ask before rewriting history.

## Implementation Biases

- Keep modules small and names direct.
- Make data ownership explicit, especially for requests, sequences, KV blocks, and scheduler queues.
- Treat observability as part of the framework: expose enough counters and traces to explain batching, cache allocation, evictions, stalls, and throughput.
- Prefer deterministic smoke tests on the 5090 before broadening scope.
- Keep APIs internal and minimal until repeated use proves the shape.
- Do not preserve unused options for hypothetical future models, GPUs, or serving modes.
