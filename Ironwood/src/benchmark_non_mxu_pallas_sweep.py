#!/usr/bin/env python3
"""Run non-MXU Pallas probes with one JF dump directory per subprocess."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from benchmark_non_mxu_pallas import CASES


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", choices=sorted(CASES))
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--case-timeout-seconds", type=int, default=420)
    args = parser.parse_args()
    args.case = args.case or sorted(CASES)
    if len(set(args.case)) != len(args.case):
        parser.error("--case entries must be unique")
    return args


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _libtpu_args(raw_dir: Path) -> str:
    replaced = (
        "--xla_jf_dump_to=",
        "--xla_jf_dump_llo_text=",
        "--xla_jf_dump_llo_proto=",
        "--xla_jf_dump_isa_program_proto=",
        "--xla_jf_emit_annotations=",
    )
    inherited = [
        flag
        for flag in shlex.split(os.environ.get("LIBTPU_INIT_ARGS", ""))
        if not flag.startswith(replaced)
    ]
    inherited.extend(
        (
            f"--xla_jf_dump_to={raw_dir}",
            "--xla_jf_dump_llo_text=true",
            "--xla_jf_dump_llo_proto=true",
            "--xla_jf_dump_isa_program_proto=true",
            "--xla_jf_emit_annotations=true",
        )
    )
    return shlex.join(inherited)


def _pass_ordinal(path: Path) -> int:
    matches = re.findall(r"-(\d+)-", path.name)
    return int(matches[-1]) if matches else -1


def _final_candidates(raw_dir: Path, kernel_name: str) -> list[Path]:
    candidates = []
    marker = f"entry bundle: %{kernel_name}"
    for path in raw_dir.rglob("*-final_bundles.txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if marker in text:
            candidates.append(path)
    return sorted(
        candidates,
        key=lambda path: (
            path.stat().st_size,
            _pass_ordinal(path),
            path.stat().st_mtime_ns,
            path.name,
        ),
        reverse=True,
    )


def _index_raw(raw_dir: Path) -> dict[str, Any]:
    files = []
    for path in sorted(candidate for candidate in raw_dir.rglob("*") if candidate.is_file()):
        files.append(
            {
                "path": str(path.relative_to(raw_dir)),
                "size_bytes": path.stat().st_size,
            }
        )
    proto_files = [
        item
        for item in files
        if "proto" in Path(item["path"]).name.lower()
        or Path(item["path"]).suffix.lower() in {".pb", ".pbtxt"}
    ]
    return {
        "file_count": len(files),
        "total_size_bytes": sum(item["size_bytes"] for item in files),
        "proto_file_count": len(proto_files),
        "proto_files": proto_files,
        "files": files,
    }


def _run_case(
    benchmark_script: Path,
    artifact_root: Path,
    case_id: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    spec = CASES[case_id]
    case_dir = artifact_root / "cases" / case_id
    raw_root = Path(os.environ.get("NON_MXU_LLO_DUMP_ROOT", "/tmp/tpu_logs/non-mxu-llo-dumps"))
    raw_dir = raw_root / case_id
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(benchmark_script),
        "--case",
        case_id,
        "--artifact-dir",
        str(artifact_root),
    ]
    env = os.environ.copy()
    env["LIBTPU_INIT_ARGS"] = _libtpu_args(raw_dir)
    started = dt.datetime.now(dt.timezone.utc)
    timed_out = False
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        returncode = 124
        stdout = (error.stdout or "") if isinstance(error.stdout, str) else (error.stdout or b"").decode(errors="replace")
        stderr = (error.stderr or "") if isinstance(error.stderr, str) else (error.stderr or b"").decode(errors="replace")
        stderr += f"\ncase timed out after {timeout_seconds} seconds\n"
    finished = dt.datetime.now(dt.timezone.utc)
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "run.stdout.txt").write_text(stdout, encoding="utf-8")
    (case_dir / "run.stderr.txt").write_text(stderr, encoding="utf-8")

    kernel_name = f"non_mxu_{case_id}"
    candidates = _final_candidates(raw_dir, kernel_name)
    llo_dir = case_dir / "compiler" / "llo"
    selected = llo_dir / "final_bundle.llo"
    if candidates:
        llo_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], selected)
        all_dir = llo_dir / "all_final_bundles"
        all_dir.mkdir(parents=True, exist_ok=True)
        for index, candidate in enumerate(candidates):
            shutil.copy2(candidate, all_dir / f"{index:02d}-{candidate.name}")
    raw_index = _index_raw(raw_dir)
    raw_index["kernel_final_candidates"] = [
        {
            "path": str(path.relative_to(raw_dir)),
            "size_bytes": path.stat().st_size,
        }
        for path in candidates
    ]
    local_raw_archive = raw_dir.parent / f"{case_id}-raw-dump.tar.gz"
    if local_raw_archive.exists():
        local_raw_archive.unlink()
    with tarfile.open(local_raw_archive, "w:gz") as archive:
        archive.add(raw_dir, arcname=".")
    raw_dump_archive = llo_dir / "raw_dump.tar.gz"
    shutil.copy2(local_raw_archive, raw_dump_archive)
    _write_json(llo_dir / "file_index.json", raw_index)

    known_dump_abort = returncode == -6 and selected.is_file() and "vmem_report_header.tmpl" in stderr
    if timed_out:
        status, reason = "failed", "timeout"
    elif selected.is_file() and (returncode == 0 or known_dump_abort):
        status, reason = "succeeded", None
    elif selected.is_file():
        status, reason = "compiled_only", "execution_failed_after_final_bundle"
    else:
        status, reason = "failed", "kernel_final_bundle_missing"
    result = {
        "schema_version": 1,
        "case_id": case_id,
        "case": {
            "family": spec.family,
            "primitives": list(spec.primitives),
            "input_kinds": list(spec.input_kinds),
            "output_dtype": spec.output_dtype,
            "shape": list(spec.shape),
            "output_shape": list(spec.output_shape),
            "mode": spec.mode,
        },
        "status": status,
        "reason": reason,
        "returncode": returncode,
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "libtpu_init_args": env["LIBTPU_INIT_ARGS"],
        "raw_file_count": raw_index["file_count"],
        "raw_total_size_bytes": raw_index["total_size_bytes"],
        "proto_file_count": raw_index["proto_file_count"],
        "raw_dump": str(raw_dump_archive.relative_to(artifact_root)),
        "final_bundle": (
            {
                "path": str(selected.relative_to(artifact_root)),
                "size_bytes": selected.stat().st_size,
            }
            if selected.is_file()
            else None
        ),
    }
    _write_json(case_dir / "status.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def _write_rank_view(artifact_root: Path, results: list[dict[str, Any]]) -> None:
    rank_dir = artifact_root / "rank-0"
    metrics = []
    for result in results:
        case_id = result["case_id"]
        source = artifact_root / "cases" / case_id / "compiler" / "llo" / "final_bundle.llo"
        if source.is_file():
            target = rank_dir / "compiler" / "llo" / case_id
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target / source.name)
        metric = artifact_root / "cases" / case_id / "metrics.json"
        if metric.is_file():
            metrics.append(metric.read_text(encoding="utf-8").strip())
    benchmark_dir = rank_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    (benchmark_dir / "metrics.jsonl").write_text("\n".join(metrics) + ("\n" if metrics else ""), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_dir.expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    source_dir = Path(__file__).resolve().parent
    benchmark_script = source_dir / "benchmark_non_mxu_pallas.py"
    analyzer_script = source_dir / "analyze_non_mxu_final_bundles.py"
    results = [
        _run_case(
            benchmark_script,
            artifact_root,
            case_id,
            args.case_timeout_seconds,
        )
        for case_id in args.case
    ]
    _write_rank_view(artifact_root, results)
    analysis_inputs = artifact_root / "analysis_inputs"
    analysis_inputs.mkdir(parents=True, exist_ok=True)
    if analyzer_script.is_file():
        shutil.copy2(analyzer_script, analysis_inputs / analyzer_script.name)
    summary = {
        "schema_version": 1,
        "workflow": "operator-optimization",
        "operator_family": "llo_non_mxu_compute",
        "operator_name": "pallas_non_mxu_compute_sweep",
        "dimensions": {
            "requested_cases": len(results),
            "succeeded": sum(result["status"] == "succeeded" for result in results),
            "compiled_only": sum(result["status"] == "compiled_only" for result in results),
            "failed": sum(result["status"] == "failed" for result in results),
        },
        "experiment_id": os.environ.get("FALCON_EXP_ID"),
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dump_flags": [
            "--xla_jf_dump_to=<case-dir>",
            "--xla_jf_dump_llo_text=true",
            "--xla_jf_dump_llo_proto=true",
            "--xla_jf_dump_isa_program_proto=true",
            "--xla_jf_emit_annotations=true",
        ],
        "source": {
            "repository": os.environ.get("NON_MXU_SOURCE_REPOSITORY"),
            "ref": os.environ.get("NON_MXU_SOURCE_REF"),
            "commit": os.environ.get("NON_MXU_SOURCE_COMMIT"),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "cases": results,
    }
    _write_json(artifact_root / "manifest.json", summary)
    _write_json(artifact_root / "sweep_summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
    if not any(result["final_bundle"] for result in results):
        raise RuntimeError("no probe produced a named Pallas final bundle")


if __name__ == "__main__":
    main()
