"""Run isolated CUDA endpoints and AIPerf sequentially on one Kubernetes GPU."""

import argparse
import json
import re
import subprocess
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from benchmarks.kubernetes.report import collect, write_failure_report, write_report

ENVIRONMENT = """import json, subprocess, torch, importlib.metadata as m
x = torch.ones(1, device="cuda")
torch.cuda.synchronize()
print(json.dumps({"torch": torch.__version__, "cuda": torch.version.cuda,
 "device": str(x.device), "gpu": torch.cuda.get_device_name(0),
 "capability": torch.cuda.get_device_capability(0),
 "packages": {n: m.version(n) for n in ("transformers",)},
 "nvidia_smi": subprocess.check_output([
 "nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used,utilization.gpu",
 "--format=csv,noheader"], text=True)}, indent=2))"""

GENERATION_PROBE = """import json, sys, urllib.error, urllib.request
payload = {"model": sys.argv[1], "prompt": "Hello" * 32,
           "max_tokens": 4, "temperature": 0, "ignore_eos": True, "stream": False}
request = urllib.request.Request("http://127.0.0.1:8000/v1/completions",
    data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
except urllib.error.HTTPError as error:
    raise RuntimeError(error.read().decode()) from error
assert result["usage"]["prompt_tokens"] == 32, result
assert result["usage"]["completion_tokens"] == 4, result
assert result["choices"][0]["finish_reason"] == "length", result
print(json.dumps(result, indent=2))"""


def kubectl(
    *args: str, payload: dict[str, Any] | None = None, check: bool = True, timeout: int = 120
) -> str:
    result = subprocess.run(
        ["kubectl", *args],
        input=json.dumps(payload) if payload else None,
        text=True,
        capture_output=True,
        timeout=timeout,
    )
    if check and result.returncode:
        raise RuntimeError(f"kubectl {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout


def apply(resource: dict[str, Any]) -> None:
    kubectl("apply", "-f", "-", payload=resource)


def server_resources(
    config: dict[str, Any],
    run_id: str,
    engine: str,
    image: str,
    pull_secret: str | None,
) -> list[dict[str, Any]]:
    name = f"{engine}-{run_id}"
    labels = {
        "app.kubernetes.io/name": "tiny-vllm-profile",
        "profile-run": run_id,
        "engine": engine,
    }
    if engine == "tiny":
        command = ["tiny-vllm-serve"]
        args = [
            "--model",
            config["model"],
            "--host",
            "0.0.0.0",
            "--device",
            "cuda",
            "--torch-dtype",
            config["dtype"],
            "--no-safetensors",
            "--max-batch-size",
            str(config["max_batch_size"]),
            "--max-num-scheduled-tokens",
            str(config["max_num_scheduled_tokens"]),
            "--max-num-blocks",
            str(config["max_num_blocks"]),
            "--block-size",
            str(config["block_size"]),
        ]
    else:
        command = ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args = [
            "--model",
            config["model_source"],
            "--revision",
            config["model_revision"],
            "--tokenizer-revision",
            config["model_revision"],
            "--served-model-name",
            config["model"],
            "--host",
            "0.0.0.0",
            "--dtype",
            config["dtype"],
            "--max-model-len",
            "1024",
            "--max-num-seqs",
            str(config["max_batch_size"]),
            "--max-num-batched-tokens",
            str(config["max_num_scheduled_tokens"]),
            "--block-size",
            str(config["block_size"]),
            "--num-gpu-blocks-override",
            str(config["max_num_blocks"]),
            "--enable-chunked-prefill",
            "--no-enable-prefix-caching",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.25",
            "--generation-config",
            "vllm",
        ]
    pod_spec: dict[str, Any] = {
        "automountServiceAccountToken": False,
        "runtimeClassName": "nvidia",
        "nodeSelector": {"kubernetes.io/hostname": config["node"]},
        "containers": [
            {
                "name": "server",
                "image": image,
                "imagePullPolicy": "IfNotPresent",
                "command": command,
                "args": args,
                "env": [
                    {"name": "VLLM_ATTENTION_BACKEND", "value": config["attention_backend"]},
                    {"name": "TOKENIZERS_PARALLELISM", "value": "false"},
                    {"name": "OMP_NUM_THREADS", "value": "1"},
                ],
                "resources": {
                    "requests": {"cpu": "2", "memory": "4Gi", "nvidia.com/gpu": "1"},
                    "limits": {"cpu": "4", "memory": "8Gi", "nvidia.com/gpu": "1"},
                },
                "ports": [{"containerPort": 8000}],
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": 8000},
                    "periodSeconds": 5,
                },
                "volumeMounts": [{"name": "shm", "mountPath": "/dev/shm"}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
            }
        ],
        "volumes": [{"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "1Gi"}}],
    }
    if pull_secret:
        pod_spec["imagePullSecrets"] = [{"name": pull_secret}]
    metadata = {"name": name, "namespace": config["namespace"], "labels": labels}
    return [
        {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": metadata,
            "spec": {
                "replicas": 1,
                "strategy": {"type": "Recreate"},
                "selector": {"matchLabels": labels},
                "template": {"metadata": {"labels": labels}, "spec": pod_spec},
            },
        },
        {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": metadata,
            "spec": {"selector": labels, "ports": [{"port": 8000, "targetPort": 8000}]},
        },
    ]


def client_resource(
    config: dict[str, Any],
    run_id: str,
    engine: str,
    image: str,
    pull_secret: str | None,
) -> dict[str, Any]:
    env = {
        "BENCH_URL": f"http://{engine}-{run_id}:8000/v1/completions",
        "BENCH_MODEL": config["model"],
        "BENCH_TOKENIZER": config["model"],
        "BENCH_ARTIFACTS": "/results/results",
        "BENCH_RUNS": str(config["runs"]),
        "BENCH_REQUESTS": str(config["requests"]),
        "BENCH_GPU_TELEMETRY": "false",
    }
    spec: dict[str, Any] = {
        "automountServiceAccountToken": False,
        "restartPolicy": "Never",
        "nodeSelector": {"kubernetes.io/hostname": config["node"]},
        "containers": [
            {
                "name": "client",
                "image": image,
                "command": ["sh", "-c"],
                "args": [
                    "aiperf profile --config /workloads/baseline.yaml; "
                    "echo $? > /results/exit-code; touch /results/done; sleep infinity"
                ],
                "env": [{"name": k, "value": v} for k, v in env.items()],
                "resources": {
                    "requests": {"cpu": "2", "memory": "2Gi"},
                    "limits": {"cpu": "4", "memory": "4Gi"},
                },
                "volumeMounts": [{"name": "results", "mountPath": "/results"}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                },
            }
        ],
        "volumes": [{"name": "results", "emptyDir": {}}],
    }
    if pull_secret:
        spec["imagePullSecrets"] = [{"name": pull_secret}]
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": f"client-{engine}-{run_id}",
            "namespace": config["namespace"],
            "labels": {"app.kubernetes.io/name": "tiny-vllm-profile", "profile-run": run_id},
        },
        "spec": spec,
    }


def wait_ready(namespace: str, name: str, *, client: bool, timeout: int) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_state = ""
    while time.monotonic() < deadline:
        pods = json.loads(kubectl("get", "pods", "-n", namespace, "-o", "json"))["items"]
        matches = (
            [p for p in pods if p["metadata"]["name"] == name]
            if client
            else [
                p
                for p in pods
                if p["metadata"]["name"].startswith(name + "-")
                and not p["metadata"].get("deletionTimestamp")
            ]
        )
        if matches:
            pod = matches[0]
            statuses = pod["status"].get("containerStatuses", [])
            state = json.dumps(
                {"phase": pod["status"]["phase"], "containers": [s.get("state") for s in statuses]}
            )
            if state != last_state:
                print(f"{name}: {state}", flush=True)
                last_state = state
            if pod["status"]["phase"] in ("Failed", "Succeeded") or any(
                s.get("state", {}).get("waiting", {}).get("reason")
                in ("CrashLoopBackOff", "RunContainerError")
                for s in statuses
            ):
                raise RuntimeError(f"{name} failed before readiness: {state}")
            if statuses and all(s.get("ready") for s in statuses):
                return cast(dict[str, Any], pod)
        time.sleep(5)
    raise TimeoutError(f"{name} did not become ready; check GPU availability and pod events")


def check_server_status(status: dict[str, Any]) -> None:
    """Reject measurements spanning a crashed, restarted or unhealthy server."""
    containers = status.get("containerStatuses", [])
    if (
        status.get("phase") != "Running"
        or not containers
        or any(c.get("restartCount", 0) or not c.get("ready") for c in containers)
    ):
        raise RuntimeError(f"Server became unhealthy during profiling: {status}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiny-image", required=True)
    parser.add_argument("--vllm-image", help="Override the upstream digest pinned in profile.json")
    parser.add_argument("--client-image", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pull-secret")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("profile.json"))
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--runs", type=int)
    parser.add_argument("--requests", type=int)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,36}[a-z0-9]", args.run_id):
        parser.error("run-id must be 2-38 lowercase letters, digits or internal hyphens")
    config = json.loads(args.config.read_text())
    args.vllm_image = args.vllm_image or config["vllm_image"]
    for name in ("runs", "requests"):
        if getattr(args, name) is not None:
            config[name] = getattr(args, name)
        if config[name] <= 0:
            parser.error(f"{name} must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    namespace = config["namespace"]
    manifest: dict[str, Any] = {
        "run_id": args.run_id,
        "source_revision": args.source_revision,
        "harness_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "harness_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
        "started_at": datetime.now(UTC).isoformat(),
        "config": config,
        "images": {"tiny": args.tiny_image, "vllm": args.vllm_image, "client": args.client_image},
        "servers": {},
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    try:
        for engine, image in (("tiny", args.tiny_image), ("vllm", args.vllm_image)):
            directory = args.output / engine
            directory.mkdir()
            name = f"{engine}-{args.run_id}"
            print(f"Starting {engine}: {image}", flush=True)
            resources = server_resources(config, args.run_id, engine, image, args.pull_secret)
            (directory / "deployment.json").write_text(json.dumps(resources, indent=2) + "\n")
            for resource in resources:
                apply(resource)
            pod = wait_ready(namespace, name, client=False, timeout=args.timeout)
            pod_name = pod["metadata"]["name"]
            environment = json.loads(
                kubectl("exec", "-n", namespace, pod_name, "--", "python3", "-c", ENVIRONMENT)
            )
            (directory / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
            if environment["gpu"] != config["gpu"] or environment["device"] != "cuda:0":
                raise RuntimeError(f"Unexpected CUDA device: {environment}")
            manifest["servers"][engine] = {
                "pod": pod_name,
                "node": pod["spec"]["nodeName"],
                "image_id": pod["status"]["containerStatuses"][0]["imageID"],
                "endpoint": f"http://{name}.{namespace}.svc:8000/v1/completions",
            }
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            # /health does not exercise attention kernels. Compile/execute a real
            # prefill and decode before spending time on the benchmark matrix.
            (directory / "generation-probe.json").write_text(
                kubectl(
                    "exec",
                    "-n",
                    namespace,
                    pod_name,
                    "--",
                    "python3",
                    "-c",
                    GENERATION_PROBE,
                    config["model"],
                    timeout=330,
                )
            )
            client = client_resource(
                config, args.run_id, engine, args.client_image, args.pull_secret
            )
            (directory / "client.json").write_text(json.dumps(client, indent=2) + "\n")
            apply(client)
            client_name = client["metadata"]["name"]
            client_pod = wait_ready(namespace, client_name, client=True, timeout=args.timeout)
            manifest["servers"][engine]["client_image_id"] = client_pod["status"][
                "containerStatuses"
            ][0]["imageID"]
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                check_server_status(
                    json.loads(kubectl("get", "pod", pod_name, "-n", namespace, "-o", "json"))[
                        "status"
                    ]
                )
                done = kubectl(
                    "exec",
                    "-n",
                    namespace,
                    client_name,
                    "--",
                    "sh",
                    "-c",
                    "test -f /results/done && cat /results/exit-code",
                    check=False,
                ).strip()
                if done:
                    break
                client_status = json.loads(
                    kubectl("get", "pod", client_name, "-n", namespace, "-o", "json")
                )["status"]
                if client_status["phase"] in ("Failed", "Succeeded"):
                    raise RuntimeError(f"AIPerf client exited without exporting results: {engine}")
                time.sleep(10)
            else:
                raise TimeoutError(f"AIPerf timed out for {engine}")
            (directory / "client.log").write_text(kubectl("logs", "-n", namespace, client_name))
            (directory / "server.log").write_text(kubectl("logs", "-n", namespace, pod_name))
            archive = directory / "results.tar"
            with archive.open("wb") as output:
                subprocess.run(
                    [
                        "kubectl",
                        "exec",
                        "-n",
                        namespace,
                        client_name,
                        "--",
                        "tar",
                        "cf",
                        "-",
                        "-C",
                        "/results",
                        ".",
                    ],
                    stdout=output,
                    check=True,
                    timeout=120,
                )
            with tarfile.open(archive) as tar:
                tar.extractall(directory, filter="data")
            archive.unlink()
            if done != "0":
                raise RuntimeError(f"AIPerf exited {done} for {engine}; see client.log")
            summary = collect(directory / "results", config["runs"], config["requests"])
            (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            manifest["servers"][engine]["status"] = "passed"
            print(f"Finished {engine}; releasing GPU before the next server", flush=True)
            kubectl("delete", "deployment", name, "-n", namespace, "--wait=true", "--timeout=90s")
            kubectl("delete", "pod", client_name, "-n", namespace, "--wait=true", "--timeout=90s")
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        write_report(args.output)
        manifest["status"] = "passed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc) or type(exc).__name__
        pods = json.loads(
            kubectl(
                "get", "pods", "-n", namespace, "-l", f"profile-run={args.run_id}", "-o", "json"
            )
        )["items"]
        for pod in pods:
            name = pod["metadata"]["name"]
            (args.output / f"{name}.log").write_text(
                kubectl("logs", "-n", namespace, name, check=False)
            )
            if any(s.get("restartCount", 0) for s in pod["status"].get("containerStatuses", [])):
                (args.output / f"{name}.previous.log").write_text(
                    kubectl("logs", "-n", namespace, name, "--previous", check=False)
                )
        (args.output / "events.txt").write_text(
            kubectl("get", "events", "-n", namespace, "--sort-by=.lastTimestamp")
        )
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        write_failure_report(args.output)
        raise
    finally:
        manifest["finished_at"] = datetime.now(UTC).isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        kubectl(
            "delete",
            "deployment,service,pod",
            "-n",
            namespace,
            "-l",
            f"profile-run={args.run_id}",
            "--wait=true",
            "--timeout=90s",
            check=False,
        )


if __name__ == "__main__":
    main()
