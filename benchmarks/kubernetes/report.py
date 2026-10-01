"""Compare repeated AIPerf exports without mixing warmup or failed runs."""

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

METRICS = {
    "latency_p50_ms": ("request_latency", "p50"),
    "latency_p95_ms": ("request_latency", "p95"),
    "latency_p99_ms": ("request_latency", "p99"),
    "requests_per_second": ("request_throughput", "avg"),
    "output_tokens_per_second": ("output_token_throughput", "avg"),
}


def collect(root: Path, expected_runs: int, expected_requests: int) -> dict[str, Any]:
    cells: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(root.rglob("profile_export_aiperf.json")):
        relative = path.relative_to(root)
        if "phases" in relative.parts:
            continue
        report = json.loads(path.read_text())
        if (
            not report.get("is_complete")
            or report.get("was_cancelled")
            or report.get("error_summary")
            or report.get("request_error_rate", {}).get("avg") != 0
            or report.get("request_count", {}).get("avg") != expected_requests
            or report.get("osl_mismatch_count", {}).get("avg", 0) != 0
        ):
            raise ValueError(f"Incomplete, failed or wrong-length run: {path}")
        # AIPerf 0.13.0 publishes trial-first paths; newer source builds use
        # scenario-first paths. Phase exports were excluded above.
        if relative.parts[0] == "profile_runs":
            if len(relative.parts) != 4 or not relative.parts[1].startswith("trial_"):
                raise ValueError(f"Unexpected AIPerf trial layout: {path}")
            scenario = relative.parts[2]
        else:
            scenario = relative.parts[0]
        cells.setdefault(scenario, []).append(report)
    if len(cells) != 7:
        raise ValueError(f"Expected seven baseline scenarios, got {list(cells)}")
    result: dict[str, Any] = {}
    for name, runs in cells.items():
        if len(runs) != expected_runs:
            raise ValueError(f"{name}: expected {expected_runs} repeats, got {len(runs)}")
        result[name] = {
            key: median(run[metric][stat] for run in runs)
            for key, (metric, stat) in METRICS.items()
        }
        result[name]["runs"] = len(runs)
        result[name]["requests_per_run"] = expected_requests
        result[name]["input_tokens"] = [run["input_sequence_length"]["avg"] for run in runs]
        result[name]["output_tokens"] = [run["output_sequence_length"]["avg"] for run in runs]
    return result


def write_report(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]
    tiny = collect(directory / "tiny" / "results", config["runs"], config["requests"])
    vllm = collect(directory / "vllm" / "results", config["runs"], config["requests"])
    for cell in tiny:
        for length in ("input_tokens", "output_tokens"):
            if tiny[cell][length] != vllm[cell][length]:
                raise ValueError(f"{cell}: {length} differs between servers")
    comparison = {"tiny": tiny, "vllm": vllm}
    (directory / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
    lines = [
        "# tiny-vLLM / vLLM GPU profile",
        "",
        f"Source: `{manifest['source_revision']}`. Run: `{manifest['run_id']}`.",
        f"GPU: {config['gpu']} on `{config['node']}`; CUDA, {config['dtype']}.",
        f"Model: `{config['model_source']}` at `{config['model_revision']}`.",
        f"{config['runs']} repeats × {config['requests']} measured requests per scenario; "
        "warmup excluded. Values are medians across runs; percentile columns are medians "
        "of per-run percentiles, not pooled percentiles.",
        "",
        "| Scenario | tiny p50 ms | vLLM p50 ms | tiny p95 ms | vLLM p95 ms | "
        "tiny output tok/s | vLLM output tok/s | vLLM / tiny tok/s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell, t in tiny.items():
        v = vllm[cell]
        ratio = v["output_tokens_per_second"] / t["output_tokens_per_second"]
        lines.append(
            f"| {cell} | {t['latency_p50_ms']:.2f} | {v['latency_p50_ms']:.2f} | "
            f"{t['latency_p95_ms']:.2f} | {v['latency_p95_ms']:.2f} | "
            f"{t['output_tokens_per_second']:.2f} | {v['output_tokens_per_second']:.2f} | "
            f"{ratio:.2f}× |"
        )
    lines += [
        "",
        "Both servers ran sequentially on the same GPU. All measured requests "
        "succeeded with matching input/output lengths. Generation was greedy, fixed-length "
        "and non-streaming. TTFT and inter-token latency are unavailable.",
        "",
        f"vLLM {config['vllm_version']} used {config['attention_backend']}, eager mode, "
        f"and disabled prefix caching. Both used {config['max_batch_size']} active requests, "
        f"a {config['max_num_scheduled_tokens']}-token step budget and "
        f"{config['max_num_blocks']} logical blocks of {config['block_size']} tokens. "
        "This is a controlled eager baseline, "
        "not a comparison against vLLM's fastest default configuration.",
        "",
        "This GPT-2 workload measures one model and serving configuration, including "
        "scheduling and kernel-launch overhead. It does not predict larger-model throughput. "
        "The node also hosts other CPU workloads; the GPU resource was exclusively reserved "
        "by each benchmark server.",
        "",
        "## Images and environment",
        "",
    ]
    for name, server in manifest["servers"].items():
        lines.append(f"- {name}: `{server['image_id']}`; endpoint `{server['endpoint']}`.")
    lines += [
        "",
        "See `manifest.json`, each server's `environment.json` and `server.log`, "
        "and raw AIPerf exports for exact versions, GPU state and request metrics.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))


def write_failure_report(directory: Path) -> None:
    """Retain independently validated numbers without claiming a comparison."""
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["config"]
    lines = [
        "# GPU profile incomplete",
        "",
        f"Source: `{manifest['source_revision']}`. Run: `{manifest['run_id']}`.",
        f"GPU: {config['gpu']} on `{config['node']}`; CUDA, {config['dtype']}.",
        f"Model: `{config['model_source']}` at `{config['model_revision']}`.",
        "",
        "**The paired comparison did not complete. No cross-engine speedup is available.**",
        "",
        "Failure detail (see server logs for the underlying cause):",
        "",
        "```text",
        manifest.get("error", "Unknown failure").replace("```", "'''"),
        "```",
    ]
    for engine in ("tiny", "vllm"):
        try:
            result = collect(directory / engine / "results", config["runs"], config["requests"])
        except (ValueError, KeyError):
            lines += ["", f"{engine}: no complete, validated measurement matrix."]
            continue
        (directory / engine / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
        lines += [
            "",
            f"## {engine}: validated measurements",
            "",
            f"{config['runs']} repeats × {config['requests']} requests per scenario; "
            "warmup excluded. Values are medians of per-run metrics.",
            "",
            "| Scenario | p50 ms | p95 ms | p99 ms | Requests/s | Output tok/s |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for name, row in result.items():
            lines.append(
                f"| {name} | {row['latency_p50_ms']:.2f} | {row['latency_p95_ms']:.2f} | "
                f"{row['latency_p99_ms']:.2f} | {row['requests_per_second']:.2f} | "
                f"{row['output_tokens_per_second']:.2f} |"
            )
    lines += [
        "",
        "See `manifest.json`, environment records, server logs and raw AIPerf "
        "exports for provenance. Non-streaming latency does not expose TTFT/ITL. "
        "Results apply only to the recorded model and serving configuration.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    write_report(parser.parse_args().directory)
