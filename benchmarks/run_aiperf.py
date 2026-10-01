"""Run an AIPerf config with a reproducibility manifest alongside its exports."""

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def command_output(command: list[str], *, cwd: Path = ROOT) -> str:
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=30)
        return (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return str(exc)


def validate_exports(directory: Path) -> tuple[list[str], list[dict[str, object]]]:
    errors: list[str] = []
    summaries: list[dict[str, object]] = []
    exports = sorted(directory.rglob("profile_export_aiperf.json"))
    if not exports:
        errors.append("no AIPerf summary exports were produced")
    for path in exports:
        report = json.loads(path.read_text())
        if (
            not report.get("is_complete")
            or report.get("was_cancelled")
            or report.get("error_summary")
            or report.get("request_error_rate", {}).get("avg", 0) != 0
            or report.get("request_count", {}).get("avg", 0) <= 0
            or report.get("osl_mismatch_count", {}).get("avg", 0) != 0
        ):
            errors.append(f"incomplete run, request errors, or output-length mismatch: {path}")
        if "phases" not in path.relative_to(directory).parts:
            summaries.append(
                {
                    "path": str(path.relative_to(directory)),
                    **{
                        key: report.get(key)
                        for key in (
                            "request_latency",
                            "request_throughput",
                            "output_token_throughput",
                            "input_sequence_length",
                            "output_sequence_length",
                            "request_error_rate",
                        )
                    },
                }
            )
    return errors, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "benchmarks/aiperf/baseline.yaml")
    parser.add_argument("--aiperf", type=Path, default=ROOT / ".venv-aiperf/bin/aiperf")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--label", choices=("tiny-vllm", "vllm", "mock"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/completions")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--server-metadata",
        type=Path,
        help="Optional JSON with the server command, model revision, dtype, GPU and version.",
    )
    args = parser.parse_args()
    config = args.config.resolve()
    executable = str(args.aiperf.resolve())
    directory = args.artifact_dir.resolve()
    env = dict(os.environ)
    env.update(
        {
            "BENCH_URL": args.url,
            "BENCH_MODEL": args.model,
            "BENCH_TOKENIZER": args.tokenizer or args.model,
            "BENCH_ARTIFACTS": str(directory / "results"),
        }
    )
    subprocess.run([executable, "config", "validate", str(config)], env=env, check=True)
    server_info = json.loads(args.server_metadata.read_text()) if args.server_metadata else None
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config, directory / "input.yaml")
    with (directory / "expanded-plan.json").open("w") as output:
        subprocess.run(
            [executable, "config", "expand", str(config), "--full", "--format", "json"],
            env=env,
            stdout=output,
            check=True,
        )
    source_files = command_output(["git", "ls-files", "--cached", "--others", "--exclude-standard"])
    hashes = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        for name in source_files.splitlines()
        if (ROOT / name).is_file()
    }
    command = [executable, "profile", "--config", str(config)]
    client_python = str(args.aiperf.resolve().parent / "python")
    direct_url_text = command_output(
        [
            client_python,
            "-c",
            "import importlib.metadata as m; "
            "print(m.distribution('aiperf').read_text('direct_url.json') or '{}')",
        ]
    )
    direct_url = json.loads(direct_url_text)
    source_url = urlsplit(direct_url.get("url", ""))
    source_commit = None
    if source_url.scheme == "file":
        source_commit = command_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(unquote(source_url.path))
        )
    (directory / "aiperf-packages.txt").write_text(
        command_output(
            [
                "uv",
                "pip",
                "freeze",
                "--python",
                client_python,
            ]
        )
        + "\n"
    )
    manifest = {
        "started_at": datetime.now(UTC).isoformat(),
        "label": args.label,
        "endpoint": args.url,
        "client_host": platform.node(),
        "client_python": platform.python_version(),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--porcelain"]),
        "source_sha256": hashes,
        "server": server_info,
        "command": command,
        "aiperf_version": command_output([executable, "--version"]),
        "aiperf_install": direct_url,
        "aiperf_source_commit": source_commit,
        "client_gpu": command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "benchmark_environment": {k: v for k, v in env.items() if k.startswith("BENCH_")},
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    result = subprocess.run(command, env=env)
    errors, summaries = validate_exports(directory / "results")
    manifest["validation_errors"] = errors
    manifest["aiperf_exit_code"] = result.returncode
    manifest["exit_code"] = result.returncode or (1 if errors else 0)
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    (directory / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    for error in errors:
        print(error, file=sys.stderr)
    sys.exit(result.returncode or (1 if errors else 0))


if __name__ == "__main__":
    main()
