#!/usr/bin/env python3
"""Sweep rank-3 Pallas matmuls in one Falcon TPU pod.

Each case uses a fresh subprocess so ``xla_jf_dump_to`` can point at a
case-specific temporary directory.  Only the selected final MXU bundle is
copied into the durable Falcon artifact and the local-export archive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any

from benchmark_tensorcore_pallas_sweep import (
    _as_text,
    _index_llo,
    _libtpu_args,
    _select_final_bundle,
    _sha256,
    _write_json,
)


def _parse_case(value: str) -> tuple[int, int, int, int]:
    fields = value.split(",")
    if len(fields) != 4:
        raise argparse.ArgumentTypeError(
            f"case must be MB,M,K,N, received {value!r}"
        )
    try:
        dimensions = tuple(int(field) for field in fields)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"case must contain integers, received {value!r}"
        ) from error
    if any(dimension <= 0 for dimension in dimensions):
        raise argparse.ArgumentTypeError(
            f"case dimensions must be positive, received {value!r}"
        )
    mb, m, k, n = dimensions
    return mb, m, k, n


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run rank-3 batched MXU LLO cases in one TPU pod."
    )
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        required=True,
        help="repeatable MB,M,K,N case",
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--case-timeout-seconds", type=int, default=600)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--export-archive",
        type=Path,
        default=Path(
            "/tmp/tpu_logs/mxu-batched-final-bundles.tar.gz"
        ),
    )
    parser.add_argument(
        "--export-hold-seconds",
        type=int,
        default=0,
        help="keep the Falcon workload alive for exp cp after export",
    )
    parser.add_argument(
        "--export-release-file",
        type=Path,
        default=Path("/tmp/tpu_logs/mxu-batched-export.release"),
    )
    args = parser.parse_args()
    if len(set(args.case)) != len(args.case):
        parser.error("--case entries must be unique")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    if args.case_timeout_seconds <= 0:
        parser.error("--case-timeout-seconds must be positive")
    if args.export_hold_seconds < 0:
        parser.error("--export-hold-seconds must be non-negative")
    return args


def _run_case(
    benchmark_script: Path,
    artifact_root: Path,
    dimensions: tuple[int, int, int, int],
    warmup: int,
    repeat: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    mb, m, k, n = dimensions
    case_id = f"mb{mb}_m{m}_k{k}_n{n}"
    case_dir = artifact_root / "cases" / case_id
    dump_root = Path(
        os.environ.get(
            "MXU_BATCHED_LLO_DUMP_ROOT",
            "/tmp/tpu_logs/mxu-batched-llo-dumps",
        )
    )
    raw_dir = dump_root / case_id
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(benchmark_script),
        "--mb",
        str(mb),
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
    (case_dir / "run.stdout.txt").write_text(stdout, encoding="utf-8")
    (case_dir / "run.stderr.txt").write_text(stderr, encoding="utf-8")
    file_index = _index_llo(raw_dir, case_id)
    selected_bundle = _select_final_bundle(raw_dir)
    final_bundle = case_dir / "compiler" / "llo" / "final_bundle.llo"
    if selected_bundle is not None:
        final_bundle.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(selected_bundle, final_bundle)
        file_index["selected_final_bundle"] = {
            "source_path": str(selected_bundle.relative_to(raw_dir)),
            "artifact_path": str(final_bundle.relative_to(artifact_root)),
            "size_bytes": final_bundle.stat().st_size,
            "sha256": _sha256(final_bundle),
        }
    else:
        file_index["selected_final_bundle"] = None
    _write_json(case_dir / "compiler" / "llo" / "file_index.json", file_index)

    known_report_abort = (
        returncode == -6
        and selected_bundle is not None
        and "vmem_report_header.tmpl" in stderr
    )
    if timed_out:
        status, reason = "failed", "case_timeout"
    elif returncode != 0 and not known_report_abort:
        status, reason = "failed", "benchmark_process_failed"
    elif selected_bundle is None:
        status, reason = "failed", "final_bundle_missing"
    else:
        status, reason = "succeeded", None

    result = {
        "schema_version": 1,
        "case_id": case_id,
        "shape": {"mb": mb, "m": m, "k": k, "n": n},
        "status": status,
        "reason": reason,
        "returncode": returncode,
        "dump_warning": (
            "libtpu aborted after writing final bundles because the public "
            "image lacks vmem_report_header.tmpl"
            if known_report_abort
            else None
        ),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "llo_file_count": file_index["file_count"],
        "final_bundle": (
            {
                "path": str(final_bundle.relative_to(artifact_root)),
                "source_dump": str(selected_bundle.relative_to(raw_dir)),
                "size_bytes": final_bundle.stat().st_size,
                "sha256": _sha256(final_bundle),
            }
            if selected_bundle is not None
            else None
        ),
    }
    _write_json(case_dir / "status.json", result)
    return result


def _write_rank_view(
    artifact_root: Path, results: list[dict[str, Any]]
) -> None:
    rank_root = artifact_root / "rank-0"
    metrics_lines = []
    for result in results:
        case_id = result["case_id"]
        case_root = artifact_root / "cases" / case_id
        source = case_root / "compiler" / "llo" / "final_bundle.llo"
        if source.is_file():
            target = rank_root / "compiler" / "llo" / case_id / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        metrics_path = case_root / "metrics.json"
        if metrics_path.is_file():
            metrics_lines.append(metrics_path.read_text(encoding="utf-8").strip())
    benchmark_dir = rank_root / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "metrics.jsonl").write_text(
        "\n".join(metrics_lines) + ("\n" if metrics_lines else ""),
        encoding="utf-8",
    )


def _write_export_archive(
    artifact_root: Path,
    results: list[dict[str, Any]],
    archive_path: Path,
) -> int:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with tarfile.open(archive_path, "w:gz") as archive:
        for result in results:
            source = (
                artifact_root
                / "cases"
                / result["case_id"]
                / "compiler"
                / "llo"
                / "final_bundle.llo"
            )
            if not source.is_file():
                continue
            archive.add(
                source,
                arcname=(
                    f"cases/{result['case_id']}/compiler/llo/"
                    "final_bundle.llo"
                ),
            )
            exported += 1
    return exported


def _hold_for_export(
    hold_seconds: int, release_file: Path, archive_path: Path
) -> None:
    if hold_seconds <= 0:
        return
    release_file.unlink(missing_ok=True)
    deadline = time.monotonic() + hold_seconds
    print(
        "MXU_BATCHED_EXPORT_READY "
        f"archive={archive_path} release={release_file} "
        f"hold_seconds={hold_seconds}",
        flush=True,
    )
    while time.monotonic() < deadline and not release_file.exists():
        time.sleep(5)
    print(
        "MXU_BATCHED_EXPORT_RELEASED"
        if release_file.exists()
        else "MXU_BATCHED_EXPORT_HOLD_EXPIRED",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_dir.expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    benchmark_script = (
        Path(__file__).resolve().parent
        / "benchmark_tensorcore_batched_pallas.py"
    )

    results = []
    for dimensions in args.case:
        result = _run_case(
            benchmark_script,
            artifact_root,
            dimensions,
            args.warmup,
            args.repeat,
            args.case_timeout_seconds,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    _write_rank_view(artifact_root, results)
    summary = {
        "schema_version": 1,
        "artifact_contract": "tensorcore_mxu_batched_final_bundle_sweep.v1",
        "experiment_id": os.environ.get("FALCON_EXP_ID"),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {
            "repository": os.environ.get("MXU_SOURCE_REPOSITORY"),
            "ref": os.environ.get("MXU_SOURCE_REF"),
            "commit": os.environ.get("MXU_SOURCE_COMMIT"),
        },
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

    exported = _write_export_archive(
        artifact_root, results, args.export_archive
    )
    print(
        json.dumps(
            {
                "export_archive": str(args.export_archive),
                "exported_final_bundles": exported,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    _hold_for_export(
        args.export_hold_seconds,
        args.export_release_file,
        args.export_archive,
    )

    failures = [result for result in results if result["status"] == "failed"]
    if failures:
        raise RuntimeError(
            "batched MXU sweep failed for "
            + ", ".join(
                f"{result['case_id']} ({result['reason']})"
                for result in failures
            )
        )


if __name__ == "__main__":
    main()
