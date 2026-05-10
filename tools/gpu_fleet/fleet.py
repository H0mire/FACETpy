#!/usr/bin/env python3
"""Small local scheduler for sharing a few SSH GPUs across many model jobs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - depends on caller environment.
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "tools" / "gpu_fleet" / "workers.local.yaml"
DEFAULT_STATE = REPO_ROOT / ".facet_gpu_fleet" / "queue.json"
SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(
    args: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    return subprocess.run(
        args,
        cwd=REPO_ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=full_env,
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Worker config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        data = yaml.safe_load(text) or {}
    else:
        data = parse_simple_workers_yaml(text)
    if not isinstance(data.get("workers"), dict):
        raise SystemExit(f"Worker config must contain a 'workers' mapping: {path}")
    return data


def read_training_config_device(config_path: Path) -> str | None:
    """Return the model.device value from a training YAML, or None if absent."""
    if not config_path.exists():
        raise SystemExit(f"Training config not found: {config_path}")
    if yaml is None:
        # Fallback: line-grep for `device:` under `model:` block.
        in_model = False
        for raw_line in config_path.read_text(encoding="utf-8").splitlines():
            stripped = raw_line.split("#", 1)[0].rstrip()
            if not stripped.strip():
                continue
            if stripped == "model:":
                in_model = True
                continue
            if in_model and stripped and not stripped.startswith(" "):
                in_model = False
            if in_model and stripped.lstrip().startswith("device:"):
                return stripped.split(":", 1)[1].strip().strip("'\"") or None
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    model = data.get("model") if isinstance(data, dict) else None
    if isinstance(model, dict):
        device = model.get("device")
        if device is None:
            return None
        return str(device).strip()
    return None


def parse_simple_workers_yaml(text: str) -> dict[str, Any]:
    """Parse the simple workers.local.yaml format without requiring PyYAML."""
    data: dict[str, Any] = {"workers": {}}
    current_worker: dict[str, Any] | None = None
    in_workers = False

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line == "workers:":
            in_workers = True
            continue
        if not in_workers:
            continue
        if line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            name = line.strip()[:-1]
            current_worker = {}
            data["workers"][name] = current_worker
            continue
        if line.startswith("    ") and current_worker is not None and ":" in line:
            key, value = line.strip().split(":", 1)
            value = value.strip().strip("'\"")
            if value.isdigit():
                current_worker[key] = int(value)
            else:
                current_worker[key] = value

    return data


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "jobs": []}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)
        handle.write("\n")
    tmp.replace(path)


def sanitize_session_name(name: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", name.strip())
    normalized = normalized.strip("_")
    if not normalized:
        normalized = f"facet_job_{uuid.uuid4().hex[:8]}"
    if not SESSION_PATTERN.match(normalized):
        raise SystemExit(f"Invalid session name after normalization: {normalized}")
    return normalized[:80]


@dataclass(frozen=True)
class Worker:
    name: str
    ssh: str
    port: int
    remote_repo: str
    gpu: int
    identity_file: str | None = None

    @classmethod
    def from_config(cls, name: str, raw: dict[str, Any]) -> "Worker":
        return cls(
            name=name,
            ssh=str(raw["ssh"]),
            port=int(raw.get("port", 22)),
            remote_repo=str(raw.get("remote_repo", "/workspace/facetpy")),
            gpu=int(raw.get("gpu", 0)),
            identity_file=str(raw["identity_file"]) if raw.get("identity_file") else None,
        )

    def ssh_args(self) -> list[str]:
        args = ["ssh", "-p", str(self.port), "-o", "StrictHostKeyChecking=accept-new"]
        if self.identity_file:
            args.extend(["-i", self.identity_file, "-o", "IdentitiesOnly=yes"])
        args.append(self.ssh)
        return args

    def script_env(self) -> dict[str, str]:
        if not self.identity_file:
            return {}
        return {"FACET_GPU_FLEET_SSH_KEY": self.identity_file}


def workers_from_config(path: Path) -> dict[str, Worker]:
    data = load_yaml(path)
    return {
        name: Worker.from_config(name, raw)
        for name, raw in data["workers"].items()
    }


def tmux_sessions(worker: Worker) -> set[str]:
    result = run(
        [
            *worker.ssh_args(),
            "tmux",
            "ls",
            "-F",
            "#{session_name}",
        ],
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def remote_exit_code(worker: Worker, session: str) -> int | None:
    result = run(
        [
            *worker.ssh_args(),
            "cat",
            f"{worker.remote_repo}/remote_logs/{session}.exitcode",
        ],
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def refresh_state(state: dict[str, Any], workers: dict[str, Worker]) -> bool:
    changed = False
    sessions_by_worker: dict[str, set[str]] = {}
    for job in state["jobs"]:
        if job["status"] not in {"running", "finished_unknown"}:
            continue
        worker_name = job.get("worker")
        if worker_name not in workers:
            continue
        if worker_name not in sessions_by_worker:
            sessions_by_worker[worker_name] = tmux_sessions(workers[worker_name])
        session_exists = job["session"] in sessions_by_worker[worker_name]
        if session_exists and job["status"] == "finished_unknown":
            job["status"] = "running"
            job.pop("finished_at", None)
            changed = True
            continue
        if session_exists:
            continue

        exit_code = remote_exit_code(workers[worker_name], job["session"])
        if exit_code is not None:
            job["status"] = "finished" if exit_code == 0 else "failed"
            job["exit_code"] = exit_code
            job["finished_at"] = utc_now()
            changed = True
            continue

        started_at = job.get("started_at")
        if started_at:
            started = datetime.fromisoformat(started_at)
            elapsed_seconds = (datetime.now(timezone.utc) - started).total_seconds()
            if elapsed_seconds < 60:
                continue
        if job["status"] == "running":
            job["status"] = "finished_unknown"
            job["finished_at"] = utc_now()
            changed = True
    return changed


def worker_is_available(worker_name: str, worker: Worker, state: dict[str, Any]) -> bool:
    for job in state["jobs"]:
        if job["status"] == "running" and job.get("worker") == worker_name:
            return False
    return True


def cmd_submit(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    session = sanitize_session_name(args.session or args.name)
    if any(job["session"] == session and job["status"] in {"pending", "running"} for job in state["jobs"]):
        raise SystemExit(f"An active job with session '{session}' already exists")

    config_path = REPO_ROOT / args.training_config
    device = read_training_config_device(config_path)
    if device is not None and device.lower() != "cuda" and not args.allow_cpu:
        raise SystemExit(
            f"Refusing to submit: training config '{args.training_config}' has model.device='{device}'. "
            "GPU workers run CUDA only. Either set 'device: cuda' in the config or pass --allow-cpu "
            "to override (intended for debugging on CPU-only workers)."
        )
    if device is None:
        print(
            f"warning: training config '{args.training_config}' has no model.device set; "
            "trusting the model factory to default to cuda on GPU workers"
        )

    job = {
        "id": uuid.uuid4().hex[:12],
        "name": args.name,
        "session": session,
        "config": args.training_config,
        "worktree": args.worktree,
        "prepare_command": args.prepare_command,
        "preferred_worker": args.worker,
        "status": "pending",
        "created_at": utc_now(),
    }
    state["jobs"].append(job)
    save_state(args.state, state)
    print(f"submitted {job['id']} {job['session']}")
    return 0


def dispatch_one_job(job: dict[str, Any], worker_name: str, worker: Worker) -> None:
    run(
        [
            "tools/gpu_fleet/sync_worktree_to_runpod.sh",
            job["worktree"],
            worker.ssh,
            worker.remote_repo,
            str(worker.port),
        ],
        env=worker.script_env(),
    )
    run(
        [
            "tools/gpu_fleet/run_remote_training.sh",
            worker.ssh,
            job["config"],
            worker.remote_repo,
            job["session"],
            str(worker.port),
            str(worker.gpu),
            job.get("prepare_command") or "",
        ],
        env=worker.script_env(),
    )
    job["status"] = "running"
    job["worker"] = worker_name
    job["started_at"] = utc_now()


def cmd_dispatch(args: argparse.Namespace) -> int:
    workers = workers_from_config(args.config)
    while True:
        state = load_state(args.state)
        changed = refresh_state(state, workers)

        for job in state["jobs"]:
            if job["status"] != "pending":
                continue
            candidate_names = [job["preferred_worker"]] if job.get("preferred_worker") else list(workers)
            for worker_name in candidate_names:
                if worker_name not in workers:
                    raise SystemExit(f"Unknown preferred worker '{worker_name}' for job {job['id']}")
                worker = workers[worker_name]
                if not worker_is_available(worker_name, worker, state):
                    continue
                print(f"dispatch {job['id']} -> {worker_name}")
                dispatch_one_job(job, worker_name, worker)
                changed = True
                break

        if changed:
            save_state(args.state, state)
        if not args.loop:
            return 0
        time.sleep(args.interval)


def cmd_status(args: argparse.Namespace) -> int:
    workers = workers_from_config(args.config)
    state = load_state(args.state)
    if refresh_state(state, workers):
        save_state(args.state, state)

    print("workers")
    for worker_name, worker in workers.items():
        running = [
            job for job in state["jobs"]
            if job["status"] == "running" and job.get("worker") == worker_name
        ]
        label = running[0]["session"] if running else "idle"
        print(f"  {worker_name}: {label} ({worker.ssh}, gpu={worker.gpu})")

    print("\njobs")
    for job in state["jobs"]:
        worker = job.get("worker") or job.get("preferred_worker") or "any"
        prepare = " prepare" if job.get("prepare_command") else ""
        print(f"  {job['id']} {job['status']:16} {worker:8} {job['session']}{prepare}")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    workers = workers_from_config(args.config)
    names = [args.worker] if args.worker else list(workers)
    for worker_name in names:
        worker = workers[worker_name]
        print(f"fetch {worker_name}")
        run(
            [
                "tools/gpu_fleet/fetch_runpod_results.sh",
                worker.ssh,
                worker.remote_repo,
                ".",
                str(worker.port),
            ],
            env=worker.script_env(),
        )
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    for job in state["jobs"]:
        if job["id"] == args.job_id:
            if job["status"] == "running":
                raise SystemExit("Refusing to cancel a running remote tmux session automatically")
            job["status"] = "cancelled"
            job["finished_at"] = utc_now()
            save_state(args.state, state)
            print(f"cancelled {job['id']}")
            return 0
    raise SystemExit(f"Job not found: {args.job_id}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    sub = parser.add_subparsers(dest="command", required=True)

    submit = sub.add_parser("submit", help="Add a training job to the local queue")
    submit.add_argument("--name", required=True)
    submit.add_argument("--worktree", required=True)
    submit.add_argument("--training-config", required=True)
    submit.add_argument(
        "--prepare-command",
        help="Optional command executed on the RunPod after sync and before facet-train.",
    )
    submit.add_argument("--session")
    submit.add_argument("--worker", help="Optional preferred worker name")
    submit.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Override the GPU-only guard. Use only when intentionally running CPU jobs.",
    )
    submit.set_defaults(func=cmd_submit)

    dispatch = sub.add_parser("dispatch", help="Start pending jobs on idle workers")
    dispatch.add_argument("--loop", action="store_true")
    dispatch.add_argument("--interval", type=int, default=60)
    dispatch.set_defaults(func=cmd_dispatch)

    status = sub.add_parser("status", help="Show workers and queued jobs")
    status.set_defaults(func=cmd_status)

    fetch = sub.add_parser("fetch", help="Fetch results from all workers or one worker")
    fetch.add_argument("--worker")
    fetch.set_defaults(func=cmd_fetch)

    cancel = sub.add_parser("cancel", help="Cancel a pending job")
    cancel.add_argument("job_id")
    cancel.set_defaults(func=cmd_cancel)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stdout)
        sys.stderr.write(exc.stderr)
        return exc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
