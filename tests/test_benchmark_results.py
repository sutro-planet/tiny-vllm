import json
from pathlib import Path

from benchmarks.run_aiperf import validate_exports


def test_benchmark_acceptance_rejects_failed_or_wrong_length_results(tmp_path: Path) -> None:
    report = {
        "is_complete": True,
        "was_cancelled": False,
        "error_summary": [],
        "request_count": {"avg": 8},
        "request_error_rate": {"avg": 0},
        "osl_mismatch_count": {"avg": 0},
    }
    path = tmp_path / "profile_export_aiperf.json"
    path.write_text(json.dumps(report))
    errors, summaries = validate_exports(tmp_path)
    assert errors == []
    assert len(summaries) == 1
    for change in (
        {"is_complete": False},
        {"was_cancelled": True},
        {"request_error_rate": {"avg": 25}},
        {"osl_mismatch_count": {"avg": 1}},
        {"request_count": {"avg": 0}},
        {"error_summary": [{"message": "failure"}]},
    ):
        path.write_text(json.dumps({**report, **change}))
        assert validate_exports(tmp_path)[0]
    path.unlink()
    assert validate_exports(tmp_path)[0]
