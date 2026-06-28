import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_vscode_gpu_smoke_tasks_are_wired_to_repo_script() -> None:
    tasks_path = REPO_ROOT / ".vscode" / "tasks.json"

    tasks_data = json.loads(tasks_path.read_text())
    tasks_by_label = {task["label"]: task for task in tasks_data["tasks"]}

    assert {
        "Tiny vLLM: GPU Smoke Current Branch",
        "Tiny vLLM: GPU Smoke Current Branch gpt2",
        "Tiny vLLM: GPU Smoke Dirty Workspace",
        "Tiny vLLM: GPU Smoke Dirty Workspace gpt2",
    }.issubset(tasks_by_label)

    for task in tasks_by_label.values():
        if not task["label"].startswith("Tiny vLLM: GPU Smoke"):
            continue
        assert task["command"] == "${workspaceFolder}/scripts/dev/gpu_smoke.sh"
        assert task["type"] == "shell"
        assert task["problemMatcher"] == []


def test_gpu_smoke_script_covers_reproducible_and_dirty_modes() -> None:
    script = (REPO_ROOT / "scripts" / "dev" / "gpu_smoke.sh").read_text()

    assert "gpu1.sutroplanet.com" in script
    assert "--mode branch" in script
    assert "--mode dirty" in script
    assert "DIRTY RSYNC RUN - NOT REPRODUCIBLE" in script
    assert "git pull --ff-only" in script
    assert "rsync" in script
    assert "nvidia-smi --query-gpu" in script
    assert "python -m pytest tests -q" in script
    assert "python -m ruff check ." in script
    assert "python -m mypy src tests" in script
    assert "PYTHONPATH=src python scripts/smoke.py" in script
    assert "PYTHONPATH=src python scripts/run_tiny_model.py" in script
