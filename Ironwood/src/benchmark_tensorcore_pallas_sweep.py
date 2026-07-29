#!/usr/bin/env python3
"""Sweep exact-shape MXU cases in one Falcon TPU pod.

Each case runs in a fresh Python subprocess so that libtpu can bind
``--xla_mosaic_dump_to`` to a case-specific directory at initialization.
There is still only one Falcon workload pod and one TPU allocation for the
whole sweep. A failed shape is recorded and the remaining shapes continue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parse_shape(value: str) -> tuple[int, int, int]:
    fields = value.split(",")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError(
            f"shape must be M,K,N, received {value!r}"
        )
    try:
        dimensions = tuple(int(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"shape must contain integers, received {value!r}"
        ) from error
    if any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError(
            f"shape dimensions must be positive, received {value!r}"
        )
    m, k, n = dimensions
    return m, k, n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an exact-shape MXU LLO sweep in one TPU pod."
    )
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        required=True,
        help="repeatable M,K,N shape",
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--case-timeout-seconds", type=int, default=600)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.shape)) != len(args.shape):
        parser.error("--shape entries must be unique")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.case_timeout_seconds <= 0:
        parser.error("--case-timeout-seconds must be positive")
    return args


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _libtpu_args(raw_dir: Path) -> str:
    inherited = shlex.split(os.environ.get("LIBTPU_INIT_ARGS", ""))
    retained = [
        flag
        for flag in inherited
        if not flag.startswith("--xla_mosaic_dump_to=")
    ]
    required = (
        "--xla_enable_custom_call_region_trace=true",
        "--xla_xprof_register_llo_debug_info=true",
    )
    for flag in required:
        if flag not in retained:
            retained.append(flag)
    retained.append(f"--xla_mosaic_dump_to={raw_dir}")
    return shlex.join(retained)


def _index_llo(raw_dir: Path, case_id: str) -> dict[str, Any]:
    files = []
    candidates = (
        candidate for candidate in raw_dir.rglob("*") if candidate.is_file()
    )
    for path in sorted(candidates):
        files.append(
            {
                "path": str(path.relative_to(raw_dir)),
                "pass": path.stem,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return {
        "schema_version": 1,
        "case_id": case_id,
        "raw_directory": str(raw_dir),
        "file_count": len(files),
        "files": files,
    }


def _run_case(
    benchmark_script: Path,
    artifact_root: Path,
    shape: tuple[int, int, int],
    warmup: int,
    repeat: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    m, k, n = shape
    case_id = f"m{m}_k{k}_n{n}"
    case_dir = artifact_root / "cases" / case_id
    raw_dir = case_dir / "compiler" / "llo" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(benchmark_script),
        "--m",
        str(m),
        "--k",
        str(k),
        "--n",
        str(n),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--artifact-dir",
        str(artifact_root),
    ]
    environment = os.environ.copy()
    environment["LIBTPU_INIT_ARGS"] = _libtpu_args(raw_dir)
    environment["MXU_ENABLE_XPROF"] = "0"

    started_at = dt.datetime.now(dt.timezone.utc)
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            env=environment,
            text=True,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = 124
        stdout = _as_text(error.stdout)
        stderr = _as_text(error.stderr) + (
            f"\ncase timed out after {timeout_seconds} seconds\n"
        )
    finished_at = dt.datetime.now(dt.timezone.utc)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "run.stdout.txt").write_text(
        stdout, encoding="utf-8"
    )
    (case_dir / "run.stderr.txt").write_text(
        stderr, encoding="utf-8"
    )

    file_index = _index_llo(raw_dir, case_id)
    index_path = case_dir / "compiler" / "llo" / "file_index.json"
    _write_json(index_path, file_index)
    final_candidates = sorted(raw_dir.glob("*post-finalize-llo.txt"))
    metrics_path = case_dir / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else None
    )

    if timed_out:
        status = "failed"
        reason = "case_timeout"
    elif returncode != 0:
        status = "failed"
        reason = "benchmark_process_failed"
    elif not final_candidates:
        status = "failed"
        reason = "final_llo_missing"
    elif not metrics or not metrics["correctness"]["passed"]:
        status = "failed"
        reason = "correctness_failed"
    else:
        status = "succeeded"
        reason = None

    result = {
        "schema_version": 1,
        "case_id": case_id,
        "shape": {"m": m, "k": k, "n": n},
        "status": status,
        "reason": reason,
        "returncode": returncode,
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "command": command,
        "libtpu_init_args": environment["LIBTPU_INIT_ARGS"],
        "llo_file_count": file_index["file_count"],
        "final_llo": (
            {
                "path": str(final_candidates[-1].relative_to(artifact_root)),
                "size_bytes": final_candidates[-1].stat().st_size,
                "sha256": _sha256(final_candidates[-1]),
            }
            if final_candidates
            else None
        ),
        "correctness": metrics["correctness"] if metrics else None,
    }
    _write_json(case_dir / "status.json", result)
    return result


def _write_compatibility_view(
    artifact_root: Path, results: list[dict[str, Any]]
) -> None:
    rank_dir = artifact_root / "rank-0"
    metrics_lines = []
    for result in results:
        case_id = result["case_id"]
        case_dir = artifact_root / "cases" / case_id
        raw_dir = case_dir / "compiler" / "llo" / "raw"
        target = rank_dir / "compiler" / "llo" / case_id
        if raw_dir.is_dir():
            shutil.copytree(raw_dir, target, dirs_exist_ok=True)
        metrics_path = case_dir / "metrics.json"
        if metrics_path.is_file():
            metrics_lines.append(metrics_path.read_text(encoding="utf-8").strip())
    benchmark_dir = rank_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "metrics.jsonl").write_text(
        "\n".join(metrics_lines) + ("\n" if metrics_lines else ""),
        encoding="utf-8",
    )


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_dir.expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    benchmark_script = (
        Path(__file__).resolve().parent / "benchmark_tensorcore_pallas.py"
    )

    results = []
    for shape in args.shape:
        result = _run_case(
            benchmark_script,
            artifact_root,
            shape,
            args.warmup,
            args.repeat,
            args.case_timeout_seconds,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    _write_compatibility_view(artifact_root, results)
    summary = {
        "schema_version": 1,
        "artifact_contract": "tensorcore_mxu_sweep.v1",
        "experiment_id": os.environ.get("FALCON_EXP_ID"),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "repository": os.environ.get("MXU_SOURCE_REPOSITORY"),
            "ref": os.environ.get("MXU_SOURCE_REF"),
            "commit": os.environ.get("MXU_SOURCE_COMMIT"),
        },
        "warmup": args.warmup,
        "repeat": args.repeat,
        "case_timeout_seconds": args.case_timeout_seconds,
        "case_count": len(results),
        "succeeded": sum(result["status"] == "succeeded" for result in results),
        "failed": sum(result["status"] == "failed" for result in results),
        "cases": results,
    }
    _write_json(artifact_root / "manifest.json", summary)
    _write_json(artifact_root / "sweep_summary.json", summary)
    _write_json(
        artifact_root / "sweep_environment.json",
        {
            "schema_version": 1,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "falcon_exp_id": os.environ.get("FALCON_EXP_ID"),
            "falcon_job_id": os.environ.get("FALCON_JOB_ID"),
        },
    )
    print(json.dumps(summary, sort_keys=True), flush=True)

    correctness_failures = [
        result
        for result in results
        if result["reason"] == "correctness_failed"
    ]
    if correctness_failures:
        raise RuntimeError(
            "correctness failed for "
            + ", ".join(result["case_id"] for result in correctness_failures)
        )


if __name__ == "__main__":
    main()
