#!/usr/bin/env python3
"""Sweep exact-shape MXU cases in one Falcon TPU pod.

Each case runs in a fresh Python subprocess so that libtpu can bind
``--xla_jf_dump_to`` to a case-specific directory at initialization. There is
still only one Falcon workload pod and one TPU allocation for the whole sweep.
JF text dumping and annotations are enabled so the backend emits final bundled
MXU LLO. Only the largest final bundle containing real MXU instructions is
copied into the Falcon artifact. A failed shape is recorded and the remaining
shapes continue.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
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
        if not flag.startswith(
            (
                "--xla_jf_debug_level=",
                "--xla_jf_dump_to=",
                "--xla_jf_dump_llo_text=",
                "--xla_jf_emit_annotations=",
                "--xla_enable_custom_call_region_trace=",
                "--xla_xprof_register_llo_debug_info=",
                "--xla_mosaic_enable_dump_debug_info=",
                "--xla_mosaic_dump_to=",
            )
        )
    ]
    retained.extend(
        (
            f"--xla_jf_dump_to={raw_dir}",
            "--xla_jf_dump_llo_text=true",
            "--xla_jf_emit_annotations=true",
            "--xla_enable_custom_call_region_trace=true",
            "--xla_xprof_register_llo_debug_info=true",
        )
    )
    return shlex.join(retained)


def _pass_ordinal(path: Path) -> int:
    matches = re.findall(r"-(\d+)-", path.name)
    return int(matches[-1]) if matches else -1


def _select_final_bundle(raw_dir: Path) -> Path | None:
    mxu_candidates = []
    for path in raw_dir.rglob("*"):
        if not path.is_file() or not path.name.endswith("-final_bundles.txt"):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if re.search(
            r"\b(?:vmat(?:mul|push\d*|prep(?:\.[a-z0-9]+)*|res))\b",
            text,
            flags=re.IGNORECASE,
        ):
            mxu_candidates.append(path)
    if not mxu_candidates:
        return None
    return max(
        mxu_candidates,
        key=lambda path: (
            path.stat().st_size,
            _pass_ordinal(path),
            path.stat().st_mtime_ns,
            path.name,
        ),
    )


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
    dump_root = Path(
        os.environ.get("MXU_LLO_DUMP_ROOT", "/tmp/tpu_logs/mxu-llo-dumps")
    )
    raw_dir = dump_root / case_id
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
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
    index_path = case_dir / "compiler" / "llo" / "file_index.json"
    _write_json(index_path, file_index)
    metrics_path = case_dir / "metrics.json"
    metrics = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics_path.is_file()
        else None
    )
    known_report_abort = (
        returncode == -6
        and selected_bundle is not None
        and "vmem_report_header.tmpl" in stderr
    )

    if timed_out:
        status = "failed"
        reason = "case_timeout"
    elif returncode != 0 and not known_report_abort:
        status = "failed"
        reason = "benchmark_process_failed"
    elif selected_bundle is None:
        status = "failed"
        reason = "final_bundle_missing"
    elif metrics and not metrics["correctness"]["passed"]:
        status = "failed"
        reason = "correctness_failed"
    elif not metrics and not known_report_abort:
        status = "failed"
        reason = "metrics_missing"
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
        "benchmark_executed": returncode == 0,
        "dump_warning": (
            "libtpu aborted after writing final bundles because the public "
            "image lacks vmem_report_header.tmpl"
            if known_report_abort
            else None
        ),
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "command": command,
        "libtpu_init_args": environment["LIBTPU_INIT_ARGS"],
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
        target = rank_dir / "compiler" / "llo" / case_id
        final_bundle = case_dir / "compiler" / "llo" / "final_bundle.llo"
        if final_bundle.is_file():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(final_bundle, target / final_bundle.name)
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
        "artifact_contract": "tensorcore_mxu_final_bundle_sweep.v3",
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

    failed_results = [
        result for result in results if result["status"] == "failed"
    ]
    if failed_results:
        raise RuntimeError(
            "MXU sweep failed for "
            + ", ".join(
                f"{result['case_id']} ({result['reason']})"
                for result in failed_results
            )
        )


if __name__ == "__main__":
    main()
