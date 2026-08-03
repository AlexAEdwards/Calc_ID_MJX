#!/usr/bin/env python3
"""Run ProcessData as disjoint subject shards sharing one CUDA GPU.

Each shard is an independent process with its own capped JAX memory pool.
Subjects, rather than individual trials, are kept together so ProcessData's
subject-level model preparation and post-processing cannot overlap writes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import IO, Any
from paths import artifact, dataset  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
PROCESS_DATA = REPO_ROOT / "ProcessData.py"
NOISED_REQUIRED = (
    "Trimming_Traceability.json",
    "pos_inputs.npy",
    "vel_inputs.npy",
    "acc_inputs.npy",
    "pelvis_rot_matrix.npy",
    "pos_mjx.npy",
    "qvel_mjx.npy",
    "qacc_mjx.npy",
    "WorldToGroundAlignedCalcnRotation.npy",
    "Jacobian.npy",
    "ankle_heights.npy",
    "COM_r.npy",
    "COM_l.npy",
    "COM_Acc_Global.npy",
    "forwardVel.npy",
    "Foot_ProgressionAngle.npy",
    "CalcnToFloor_AngleDeg.npy",
    "qfrc_inverse.npy",
    "COP_Cleaned_Relative.npy",
    "COP_CalcFrame_GroundAligned.npy",
    "COP_CalcFrame_GroundAligned_GRFNorm.npy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.30)
    parser.add_argument(
        "--python",
        default="/home/mobl/miniconda3/envs/myoconverter/bin/python",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=artifact("output") / "processdata_gpu_shards",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Launch each ProcessData shard with --dry-run.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Do not add --only-new. Use only when overwriting is intentional.",
    )
    parser.add_argument(
        "--max-worker-rss-gb",
        type=float,
        default=18.0,
        help=(
            "Gracefully recycle a shard when its host resident memory reaches this "
            "limit. The restarted shard uses --only-new and resumes completed work. "
            "Use 0 to disable. Default: 18 GB."
        ),
    )
    parser.add_argument(
        "--min-system-available-gb",
        type=float,
        default=8.0,
        help=(
            "Emergency recycle the largest shard when Linux MemAvailable falls below "
            "this value. Use 0 to disable. Default: 8 GB."
        ),
    )
    parser.add_argument(
        "--status-interval-seconds",
        type=float,
        default=5.0,
        help="Worker RSS/status sampling interval. Default: 5 seconds.",
    )
    parser.add_argument(
        "--recycle-grace-seconds",
        type=float,
        default=20.0,
        help="Seconds allowed for SIGINT shutdown before SIGTERM. Default: 20.",
    )
    parser.add_argument(
        "--max-unexpected-restarts",
        type=int,
        default=5,
        help="Maximum automatic restarts after non-recycle worker failures.",
    )
    parser.add_argument(
        "processdata_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded after --, such as -- --UseNoised --COP_EdgeHold.",
    )
    return parser.parse_args()


def suffixed_filename(filename: str) -> str:
    path = Path(filename)
    return f"{path.stem}_noised{path.suffix}"


def noised_bundle_complete(processed: Path) -> bool:
    return all(
        (processed / suffixed_filename(filename)).is_file()
        for filename in NOISED_REQUIRED
    )


def subject_work(subject: Path, use_noised: bool) -> tuple[int, int]:
    total = 0
    pending = 0
    for trial in subject.iterdir():
        if not (trial.is_dir() and trial.name.startswith("Trial")):
            continue
        total += 1
        processed = trial / "ProcessedData"
        clean_complete = (processed / "pos_inputs.npy").is_file()
        complete = clean_complete and (not use_noised or noised_bundle_complete(processed))
        pending += int(not complete)
    return pending, total


def build_shards(
    dataset: Path,
    shard_count: int,
    use_noised: bool,
) -> list[dict]:
    subjects = []
    for subject in dataset.iterdir():
        if not subject.is_dir() or subject.name.startswith("."):
            continue
        pending, total = subject_work(subject, use_noised)
        if total:
            subjects.append(
                {
                    "subject": subject.name,
                    "pending_trials": pending,
                    "active_trials": total,
                    # A pending trial performs clean + noised core processing and
                    # later post-processing; an existing trial still participates
                    # in the final calc-frame/FPA post-pass.
                    "estimated_work_units": pending * 5 + total,
                }
            )

    shards = [
        {
            "subjects": [],
            "pending_trials": 0,
            "active_trials": 0,
            "estimated_work_units": 0,
        }
        for _ in range(shard_count)
    ]
    # Greedy longest-processing-time allocation balances expected remaining work.
    for item in sorted(
        subjects,
        key=lambda row: (
            row["estimated_work_units"],
            row["pending_trials"],
            row["active_trials"],
            row["subject"],
        ),
        reverse=True,
    ):
        target = min(
            shards,
            key=lambda shard: (
                shard["estimated_work_units"],
                shard["pending_trials"],
                shard["active_trials"],
                len(shard["subjects"]),
            ),
        )
        target["subjects"].append(item["subject"])
        target["pending_trials"] += item["pending_trials"]
        target["active_trials"] += item["active_trials"]
        target["estimated_work_units"] += item["estimated_work_units"]
    for shard in shards:
        shard["subjects"].sort()
    return shards


def process_rss_bytes(pid: int) -> int | None:
    """Read a process's resident host memory from Linux /proc."""
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
        return None
    return None


def system_available_bytes() -> int | None:
    """Read Linux MemAvailable, which accounts for reclaimable page cache."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def format_gib(value: int | None) -> str:
    return "?" if value is None else f"{value / (1024 ** 3):.1f}GiB"


def stop_process_group(
    child: subprocess.Popen,
    interrupt_grace_seconds: float,
    *,
    reason: str,
) -> int:
    """Stop a shard, escalating from SIGINT to SIGTERM to SIGKILL."""
    if child.poll() is not None:
        return int(child.returncode)
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        return int(child.wait())

    deadline = time.monotonic() + interrupt_grace_seconds
    while time.monotonic() < deadline:
        returncode = child.poll()
        if returncode is not None:
            return int(returncode)
        time.sleep(0.25)

    print(
        f"[worker pid={child.pid}] SIGINT timeout during {reason}; sending SIGTERM",
        flush=True,
    )
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return int(child.wait())
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        returncode = child.poll()
        if returncode is not None:
            return int(returncode)
        time.sleep(0.25)

    print(
        f"[worker pid={child.pid}] SIGTERM timeout during {reason}; sending SIGKILL",
        flush=True,
    )
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    return int(child.wait())


@dataclass
class WorkerState:
    index: int
    shard: dict[str, Any]
    command: list[str]
    env: dict[str, str]
    log_path: Path
    log_handle: IO[str]
    child: subprocess.Popen | None = None
    generation: int = 0
    recycle_count: int = 0
    unexpected_restart_count: int = 0
    completed: bool = False
    failed: bool = False
    final_returncode: int | None = None
    last_rss_bytes: int | None = None
    events: list[dict[str, Any]] = field(default_factory=list)


def launch_worker(
    worker: WorkerState,
    *,
    reason: str,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    worker.generation += 1
    marker = (
        f"\n{'=' * 78}\n"
        f"[launcher] starting shard {worker.index} generation {worker.generation}; "
        f"reason={reason}; time={datetime.now().isoformat(timespec='seconds')}\n"
        f"{'=' * 78}\n"
    )
    worker.log_handle.write(marker)
    worker.log_handle.flush()
    worker.child = subprocess.Popen(
        worker.command,
        cwd=REPO_ROOT,
        env=worker.env,
        stdout=worker.log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": "launch",
        "reason": reason,
        "generation": worker.generation,
        "pid": worker.child.pid,
    }
    worker.events.append(event)
    manifest["worker_events"][str(worker.index)] = worker.events
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[shard {worker.index}] generation={worker.generation} pid={worker.child.pid} "
        f"reason={reason} subjects={len(worker.shard['subjects'])} "
        f"pending_at_start={worker.shard['pending_trials']} log={worker.log_path}",
        flush=True,
    )


def recycle_worker(
    worker: WorkerState,
    *,
    reason: str,
    rss_bytes: int | None,
    interrupt_grace_seconds: float,
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    if worker.child is None or worker.child.poll() is not None:
        return
    old_pid = worker.child.pid
    print(
        f"[shard {worker.index}] recycling pid={old_pid}: {reason}; "
        f"rss={format_gib(rss_bytes)}",
        flush=True,
    )
    worker.log_handle.write(
        f"\n[launcher] recycling pid={old_pid}; reason={reason}; "
        f"rss={format_gib(rss_bytes)}; "
        f"time={datetime.now().isoformat(timespec='seconds')}\n"
    )
    worker.log_handle.flush()
    returncode = stop_process_group(
        worker.child,
        interrupt_grace_seconds,
        reason=reason,
    )
    worker.recycle_count += 1
    event = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "event": "recycle",
        "reason": reason,
        "generation": worker.generation,
        "pid": old_pid,
        "rss_bytes": rss_bytes,
        "returncode": returncode,
    }
    worker.events.append(event)
    manifest["worker_events"][str(worker.index)] = worker.events
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    # Give CUDA and the OS a moment to release the old process's allocations.
    time.sleep(1.0)
    launch_worker(
        worker,
        reason=f"resume_after_recycle_{worker.recycle_count}",
        manifest_path=manifest_path,
        manifest=manifest,
    )


def main() -> int:
    args = parse_args()
    dataset = args.data_root
    if not dataset.is_absolute():
        dataset = (REPO_ROOT / dataset).resolve()
    if not dataset.is_dir():
        raise FileNotFoundError(dataset)
    if args.shards < 1:
        raise ValueError("--shards must be at least 1")
    if not (0.0 < args.gpu_memory_fraction < 1.0):
        raise ValueError("--gpu-memory-fraction must be between 0 and 1")
    total_fraction = args.shards * args.gpu_memory_fraction
    if total_fraction > 0.90 + 1e-9:
        raise ValueError(
            "The total reserved fraction must be <= 0.90 to leave display/context headroom; "
            f"got {args.shards} × {args.gpu_memory_fraction} = {total_fraction:.3f}"
        )
    if args.max_worker_rss_gb < 0:
        raise ValueError("--max-worker-rss-gb must be nonnegative")
    if args.min_system_available_gb < 0:
        raise ValueError("--min-system-available-gb must be nonnegative")
    if args.status_interval_seconds <= 0:
        raise ValueError("--status-interval-seconds must be positive")
    if args.recycle_grace_seconds < 0:
        raise ValueError("--recycle-grace-seconds must be nonnegative")
    if args.max_unexpected_restarts < 0:
        raise ValueError("--max-unexpected-restarts must be nonnegative")
    if args.allow_overwrite and (
        args.max_worker_rss_gb > 0 or args.min_system_available_gb > 0
    ):
        raise ValueError(
            "RSS recycling requires the default --only-new behavior; "
            "do not combine memory recycling with --allow-overwrite"
        )

    forwarded = list(args.processdata_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    forbidden = {"--data-root", "--subjects", "--subject", "--workers", "--device"}
    if any(arg.split("=", 1)[0] in forbidden for arg in forwarded):
        raise ValueError(
            "Do not forward data-root/subject/worker/device selectors; the launcher owns them."
        )
    use_noised = "--UseNoised" in forwarded or "--OnlyProcessNoised" in forwarded
    shards = build_shards(dataset, args.shards, use_noised)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.log_dir
    if not run_dir.is_absolute():
        run_dir = (REPO_ROOT / run_dir).resolve()
    run_dir = run_dir / run_stamp
    run_dir.mkdir(parents=True, exist_ok=False)

    base_command = [
        str(Path(args.python).resolve()),
        str(PROCESS_DATA),
        "--data-root",
        str(dataset),
        "--device",
        "gpu",
        "--workers",
        "1",
    ]
    if not args.allow_overwrite:
        base_command.append("--only-new")
    if args.dry_run:
        base_command.append("--dry-run")

    manifest = {
        "schema_version": "1.1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": str(dataset),
        "shard_count": args.shards,
        "gpu_memory_fraction_per_shard": args.gpu_memory_fraction,
        "gpu_memory_fraction_total": total_fraction,
        "only_new": not args.allow_overwrite,
        "dry_run": args.dry_run,
        "memory_recycling": {
            "max_worker_rss_gb": args.max_worker_rss_gb,
            "min_system_available_gb": args.min_system_available_gb,
            "status_interval_seconds": args.status_interval_seconds,
            "recycle_grace_seconds": args.recycle_grace_seconds,
            "max_unexpected_restarts": args.max_unexpected_restarts,
        },
        "shards": shards,
        "worker_events": {},
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    workers: list[WorkerState] = []
    try:
        for index, shard in enumerate(shards, start=1):
            if not shard["subjects"]:
                continue
            command = base_command + [
                "--subjects",
                ",".join(shard["subjects"]),
                *forwarded,
            ]
            log_path = run_dir / f"shard_{index}.log"
            log_handle = log_path.open("w", buffering=1)
            env = os.environ.copy()
            env.pop("LD_LIBRARY_PATH", None)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
            env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.gpu_memory_fraction)
            env["XLA_PYTHON_CLIENT_ALLOCATOR"] = "default"
            worker = WorkerState(
                index=index,
                shard=shard,
                command=command,
                env=env,
                log_path=log_path,
                log_handle=log_handle,
            )
            workers.append(worker)
            launch_worker(
                worker,
                reason="initial",
                manifest_path=manifest_path,
                manifest=manifest,
            )

        max_rss_bytes = (
            int(args.max_worker_rss_gb * (1024 ** 3))
            if args.max_worker_rss_gb > 0 else None
        )
        min_available_bytes = (
            int(args.min_system_available_gb * (1024 ** 3))
            if args.min_system_available_gb > 0 else None
        )

        while not all(worker.completed for worker in workers):
            # First handle normal and unexpected child exits.
            for worker in workers:
                if worker.completed or worker.child is None:
                    continue
                returncode = worker.child.poll()
                if returncode is None:
                    worker.last_rss_bytes = process_rss_bytes(worker.child.pid)
                    continue
                if returncode == 0:
                    worker.completed = True
                    worker.final_returncode = 0
                    event = {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "event": "completed",
                        "generation": worker.generation,
                        "pid": worker.child.pid,
                        "returncode": 0,
                    }
                    worker.events.append(event)
                    print(
                        f"[shard {worker.index}] completed successfully after "
                        f"{worker.recycle_count} recycle(s)",
                        flush=True,
                    )
                elif worker.unexpected_restart_count < args.max_unexpected_restarts:
                    worker.unexpected_restart_count += 1
                    event = {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "event": "unexpected_exit",
                        "generation": worker.generation,
                        "pid": worker.child.pid,
                        "returncode": int(returncode),
                        "restart_number": worker.unexpected_restart_count,
                    }
                    worker.events.append(event)
                    print(
                        f"[shard {worker.index}] unexpected exit={returncode}; "
                        f"restarting ({worker.unexpected_restart_count}/"
                        f"{args.max_unexpected_restarts})",
                        flush=True,
                    )
                    time.sleep(1.0)
                    launch_worker(
                        worker,
                        reason=(
                            f"resume_after_unexpected_exit_{returncode}_"
                            f"{worker.unexpected_restart_count}"
                        ),
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )
                else:
                    worker.completed = True
                    worker.failed = True
                    worker.final_returncode = int(returncode)
                    event = {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "event": "failed",
                        "generation": worker.generation,
                        "pid": worker.child.pid,
                        "returncode": int(returncode),
                        "unexpected_restarts_exhausted": True,
                    }
                    worker.events.append(event)
                    print(
                        f"[shard {worker.index}] FAILED exit={returncode}; "
                        f"unexpected restart limit exhausted",
                        flush=True,
                    )
                manifest["worker_events"][str(worker.index)] = worker.events
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

            # Recycle any worker that crosses its individual RSS limit.
            if max_rss_bytes is not None:
                for worker in workers:
                    if (
                        worker.completed
                        or worker.child is None
                        or worker.child.poll() is not None
                    ):
                        continue
                    rss_bytes = process_rss_bytes(worker.child.pid)
                    worker.last_rss_bytes = rss_bytes
                    if rss_bytes is not None and rss_bytes >= max_rss_bytes:
                        recycle_worker(
                            worker,
                            reason=(
                                f"worker RSS {format_gib(rss_bytes)} reached "
                                f"limit {args.max_worker_rss_gb:.1f}GiB"
                            ),
                            rss_bytes=rss_bytes,
                            interrupt_grace_seconds=args.recycle_grace_seconds,
                            manifest_path=manifest_path,
                            manifest=manifest,
                        )

            # Protect against other applications consuming enough RAM to make
            # the per-worker threshold insufficient. Recycle only the largest
            # live shard, then reassess on the next status interval.
            available_bytes = system_available_bytes()
            if (
                min_available_bytes is not None
                and available_bytes is not None
                and available_bytes < min_available_bytes
            ):
                live_workers = [
                    worker
                    for worker in workers
                    if (
                        not worker.completed
                        and worker.child is not None
                        and worker.child.poll() is None
                    )
                ]
                if live_workers:
                    for worker in live_workers:
                        worker.last_rss_bytes = process_rss_bytes(worker.child.pid)
                    largest = max(
                        live_workers,
                        key=lambda item: item.last_rss_bytes or 0,
                    )
                    recycle_worker(
                        largest,
                        reason=(
                            f"system MemAvailable {format_gib(available_bytes)} "
                            f"fell below {args.min_system_available_gb:.1f}GiB"
                        ),
                        rss_bytes=largest.last_rss_bytes,
                        interrupt_grace_seconds=args.recycle_grace_seconds,
                        manifest_path=manifest_path,
                        manifest=manifest,
                    )
                    available_bytes = system_available_bytes()

            statuses = []
            for worker in workers:
                if worker.completed:
                    status = (
                        f"failed={worker.final_returncode}"
                        if worker.failed
                        else "complete"
                    )
                elif worker.child is None:
                    status = "not-started"
                elif worker.child.poll() is None:
                    worker.last_rss_bytes = process_rss_bytes(worker.child.pid)
                    status = (
                        f"running pid={worker.child.pid} "
                        f"rss={format_gib(worker.last_rss_bytes)} "
                        f"recycles={worker.recycle_count}"
                    )
                else:
                    status = f"exit={worker.child.returncode}"
                statuses.append(f"shard{worker.index}:{status}")
            print(
                f"[status] {', '.join(statuses)}; "
                f"system_available={format_gib(available_bytes)}",
                flush=True,
            )
            if not all(worker.completed for worker in workers):
                time.sleep(args.status_interval_seconds)
    except KeyboardInterrupt:
        print("\nStopping all GPU shards...", flush=True)
        for worker in workers:
            if worker.child is not None and worker.child.poll() is None:
                stop_process_group(
                    worker.child,
                    args.recycle_grace_seconds,
                    reason="launcher interrupted by user",
                )
        manifest["interrupted_at"] = datetime.now().isoformat(timespec="seconds")
        manifest["result"] = "interrupted"
        for worker in workers:
            manifest["worker_events"][str(worker.index)] = worker.events
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        return 130
    finally:
        for worker in workers:
            worker.log_handle.close()

    failures = [worker for worker in workers if worker.failed]
    manifest["completed_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["result"] = "failed" if failures else "success"
    manifest["worker_summary"] = {
        str(worker.index): {
            "generations": worker.generation,
            "recycles": worker.recycle_count,
            "unexpected_restarts": worker.unexpected_restart_count,
            "final_returncode": worker.final_returncode,
            "failed": worker.failed,
        }
        for worker in workers
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        for worker in failures:
            print(
                f"[shard {worker.index}] FAILED exit={worker.final_returncode}; "
                f"see {worker.log_path}"
            )
        return 1
    print(f"All GPU shards completed successfully. Logs: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
