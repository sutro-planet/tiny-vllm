import copy
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.kubernetes.report import collect, write_failure_report, write_report
from benchmarks.kubernetes.run import check_server_status, client_resource, server_resources


def test_gpu_resources_are_scoped_and_use_explicit_cuda() -> None:
    config = json.loads(Path("benchmarks/kubernetes/profile.json").read_text())
    tiny, service = server_resources(config, "test-42", "tiny", "image@sha256:test", None)
    pod = tiny["spec"]["template"]["spec"]
    assert pod["nodeSelector"] == {"kubernetes.io/hostname": "sutro-gpu1"}
    assert not pod["automountServiceAccountToken"]
    assert pod["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] == "1"
    args = pod["containers"][0]["args"]
    assert args[args.index("--device") + 1] == "cuda"
    assert service["spec"]["selector"] == tiny["spec"]["selector"]["matchLabels"]
    assert tiny["metadata"]["namespace"] == "tiny-vllm-profile"
    vllm, _ = server_resources(config, "test-42", "vllm", config["vllm_image"], None)
    vllm_args = vllm["spec"]["template"]["spec"]["containers"][0]["args"]
    assert vllm_args[vllm_args.index("--model") + 1] == config["model_source"]
    assert vllm_args[vllm_args.index("--revision") + 1] == config["model_revision"]
    assert vllm_args[vllm_args.index("--served-model-name") + 1] == config["model"]
    client = client_resource(config, "test-42", "tiny", "client-image", None)
    env = {e["name"]: e["value"] for e in client["spec"]["containers"][0]["env"]}
    assert env["BENCH_URL"] == "http://tiny-test-42:8000/v1/completions"
    assert "nvidia.com/gpu" not in client["spec"]["containers"][0]["resources"]["limits"]


def exports(root: Path) -> None:
    for scenario in range(7):
        for trial in range(3):
            path = root / f"cell-{scenario}" / "profile_runs" / f"run_{trial + 1:04d}"
            path.mkdir(parents=True)
            report = {
                "is_complete": True,
                "was_cancelled": False,
                "error_summary": [],
                "request_error_rate": {"avg": 0},
                "request_count": {"avg": 100},
                "osl_mismatch_count": {"avg": 0},
                "request_latency": {"p50": 10 + trial, "p95": 20 + trial, "p99": 30 + trial},
                "request_throughput": {"avg": 20 + trial},
                "output_token_throughput": {"avg": 640 + trial},
                "input_sequence_length": {"avg": 128},
                "output_sequence_length": {"avg": 32},
            }
            (path / "profile_export_aiperf.json").write_text(json.dumps(report))


def test_report_requires_all_repeats_and_rejects_failed_results(tmp_path: Path) -> None:
    exports(tmp_path)
    result = collect(tmp_path, 3, 100)
    assert result["cell-0"]["latency_p50_ms"] == 11
    path = next(tmp_path.rglob("profile_export_aiperf.json"))
    valid = json.loads(path.read_text())
    for key, value in [
        ("request_error_rate", {"avg": 1}),
        ("request_count", {"avg": 8}),
        ("osl_mismatch_count", {"avg": 1}),
        ("is_complete", False),
    ]:
        failed = copy.deepcopy(valid)
        failed[key] = value
        path.write_text(json.dumps(failed))
        with pytest.raises(ValueError):
            collect(tmp_path, 3, 100)
    path.unlink()
    with pytest.raises(ValueError, match="repeats"):
        collect(tmp_path, 3, 100)


def test_report_compares_lengths_before_publishing(tmp_path: Path) -> None:
    config = json.loads(Path("benchmarks/kubernetes/profile.json").read_text())
    manifest = {"config": config, "run_id": "test-42", "source_revision": "test", "servers": {}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for engine in ("tiny", "vllm"):
        exports(tmp_path / engine / "results")
    write_report(tmp_path)
    assert "1.00×" in (tmp_path / "report.md").read_text()
    path = next((tmp_path / "vllm" / "results").rglob("profile_export_aiperf.json"))
    report = json.loads(path.read_text())
    report["output_sequence_length"]["avg"] = 31
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="output_tokens differs"):
        write_report(tmp_path)


def test_report_reads_published_aiperf_trial_first_layout(tmp_path: Path) -> None:
    exports(tmp_path)
    for path in list(tmp_path.rglob("profile_export_aiperf.json")):
        relative = path.relative_to(tmp_path)
        destination = (
            tmp_path
            / "profile_runs"
            / relative.parts[2].replace("run_", "trial_")
            / relative.parts[0]
            / path.name
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.rename(destination)
    result = collect(tmp_path, 3, 100)
    assert len(result) == 7
    assert result["cell-0"]["runs"] == 3


def test_failed_baseline_retains_validated_tiny_numbers(tmp_path: Path) -> None:
    config = json.loads(Path("benchmarks/kubernetes/profile.json").read_text())
    manifest = {
        "config": config,
        "run_id": "test-42",
        "source_revision": "test",
        "error": "Unsupported attention head size",
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    exports(tmp_path / "tiny" / "results")
    write_failure_report(tmp_path)
    report = (tmp_path / "report.md").read_text()
    assert "tiny: validated measurements" in report
    assert "Unsupported attention head size" in report
    assert "vllm: no complete, validated measurement matrix" in report
    assert "No cross-engine speedup is available" in report
    assert "1.00×" not in report


def test_restarted_or_unhealthy_server_invalidates_profile() -> None:
    status: dict[str, Any] = {
        "phase": "Running",
        "containerStatuses": [{"ready": True, "restartCount": 0}],
    }
    check_server_status(status)
    for change in ({"ready": False}, {"restartCount": 1}):
        broken = copy.deepcopy(status)
        broken["containerStatuses"][0].update(change)
        with pytest.raises(RuntimeError, match="unhealthy"):
            check_server_status(broken)
