#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/dev/gpu_smoke.sh --mode branch|dirty --target smoke|gpt2

Runs tiny-vLLM smoke checks on gpu1.sutroplanet.com from VS Code or a shell.

Modes:
  --mode branch  Run the current committed branch from the remote git checkout.
  --mode dirty   Rsync the local workspace to a remote scratch checkout.

Targets:
  --target smoke Run tests, lint, typing, and scripts/smoke.py.
  --target gpt2  Run the smoke target plus a gpt2 tiny-model generation check.

Environment overrides:
  TINY_VLLM_GPU_HOST       SSH host, default: gpu1.sutroplanet.com
  TINY_VLLM_GPU_REPO_DIR   Remote branch checkout, default: ~/workspace/tiny-vllm
  TINY_VLLM_GPU_DIRTY_DIR  Remote dirty checkout, default: ~/workspace/tiny-vllm-dirty
EOF
}

mode=""
target="smoke"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      mode="${2:-}"
      shift 2
      ;;
    --target)
      target="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "$mode" != "branch" && "$mode" != "dirty" ]]; then
  echo "--mode must be branch or dirty" >&2
  usage >&2
  exit 2
fi

if [[ "$target" != "smoke" && "$target" != "gpt2" ]]; then
  echo "--target must be smoke or gpt2" >&2
  usage >&2
  exit 2
fi

host="${TINY_VLLM_GPU_HOST:-gpu1.sutroplanet.com}"
remote_repo_dir="${TINY_VLLM_GPU_REPO_DIR:-~/workspace/tiny-vllm}"
remote_dirty_dir="${TINY_VLLM_GPU_DIRTY_DIR:-~/workspace/tiny-vllm-dirty}"
repo_root="$(git rev-parse --show-toplevel)"
branch="$(git -C "$repo_root" branch --show-current)"
commit="$(git -C "$repo_root" rev-parse --short HEAD)"
repo_url="$(git -C "$repo_root" remote get-url origin)"

resolve_remote_dir() {
  local dir="$1"
  ssh "$host" bash -s -- "$dir" <<'REMOTE'
set -euo pipefail
dir="${1/#\~/$HOME}"
mkdir -p "$dir"
printf '%s\n' "$dir"
REMOTE
}

require_branch_ready() {
  if [[ -n "$(git -C "$repo_root" status --porcelain --untracked-files=all)" ]]; then
    echo "branch mode requires a clean local tree." >&2
    echo "Commit and push first, or run --mode dirty for an rsynced scratch run." >&2
    exit 1
  fi

  git -C "$repo_root" fetch --quiet origin

  local upstream
  upstream="$(git -C "$repo_root" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  if [[ -z "$upstream" ]]; then
    echo "branch mode requires an upstream branch. Push this branch first." >&2
    exit 1
  fi

  local behind ahead
  read -r behind ahead < <(git -C "$repo_root" rev-list --left-right --count "$upstream...HEAD")
  if [[ "$ahead" != "0" ]]; then
    echo "branch mode will run the remote branch, but local HEAD is $ahead commit(s) ahead." >&2
    echo "Push first, or run --mode dirty for an rsynced scratch run." >&2
    exit 1
  fi
}

prepare_branch_checkout() {
  local remote_dir
  remote_dir="$(resolve_remote_dir "$remote_repo_dir")"
  ssh "$host" bash -s -- "$remote_dir" "$repo_url" "$branch" <<'REMOTE' >&2
set -euo pipefail
remote_dir="$1"
repo_url="$2"
branch="$3"

mkdir -p "$(dirname "$remote_dir")"
if [[ -d "$remote_dir/.git" ]]; then
  cd "$remote_dir"
  if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
    echo "remote checkout is dirty: $remote_dir" >&2
    exit 1
  fi
  git fetch origin
else
  git clone "$repo_url" "$remote_dir"
  cd "$remote_dir"
fi

git checkout -B "$branch" "origin/$branch"
git pull --ff-only origin "$branch"
REMOTE
  printf '%s\n' "$remote_dir"
}

prepare_dirty_checkout() {
  local remote_dir
  remote_dir="$(resolve_remote_dir "$remote_dirty_dir")"
  echo "DIRTY RSYNC RUN - NOT REPRODUCIBLE" >&2
  echo "local branch=$branch commit=$commit" >&2
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.mypy_cache/' \
    --exclude '.pytest_cache/' \
    --exclude '.ruff_cache/' \
    "$repo_root/" "$host:$remote_dir/" >&2
  printf '%s\n' "$remote_dir"
}

run_remote_smoke() {
  local remote_dir="$1"
  ssh "$host" bash -s -- "$remote_dir" "$target" <<'REMOTE'
set -euo pipefail
remote_dir="$1"
target="$2"

cd "$remote_dir"
echo "remote_dir=$remote_dir"
hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 --version

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

. .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -m pytest tests -q
python -m ruff check .
python -m mypy src tests
PYTHONPATH=src python scripts/smoke.py

if [[ "$target" == "gpt2" ]]; then
  python -m pip install -e ".[transformers]"
  PYTHONPATH=src python scripts/run_tiny_model.py \
    --model gpt2 \
    --prompt "Hello" \
    --max-new-tokens 3 \
    --no-safetensors
fi
REMOTE
}

echo "tiny-vllm gpu smoke: mode=$mode target=$target host=$host branch=$branch commit=$commit"

if [[ "$mode" == "branch" ]]; then
  require_branch_ready
  remote_dir="$(prepare_branch_checkout)"
else
  remote_dir="$(prepare_dirty_checkout)"
fi

run_remote_smoke "$remote_dir"
