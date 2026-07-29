#!/usr/bin/env python3
"""Run one Pallas ``lax.dot`` case for MXU behavior reverse engineering.

The benchmark deliberately exposes the requested matrix shapes as one Pallas
program.  It does not tile or pad M, K, or N in Python; any tiling, padding,
masking, register allocation, FIFO use, or MXU issue decomposition observed in
the LLO is therefore introduced below this source-level kernel.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import platform
import statistics
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import lax
from jax.experimental import pallas as pl


INPUT_DTYPE = jnp.bfloat16
ACCUMULATOR_DTYPE = jnp.float32
OUTPUT_DTYPE = jnp.bfloat16
KERNEL_NAME = "pallas_lax_dot_single_program"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one exact-shape Pallas lax.dot MXU case."
    )
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    for name in ("m", "k", "n"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    return args


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _device_metadata(device: Any) -> dict[str, Any]:
    fields = (
        "id",
        "platform",
        "device_kind",
        "process_index",
        "local_hardware_id",
        "slice_index",
    )
    result = {}
    for field in fields:
        value = getattr(device, field, None)
        if value is not None:
            result[field] = value
    result["repr"] = str(device)
    return result


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _build_single_program_dot(m: int, k: int, n: int):
    def kernel(lhs_ref, rhs_ref, out_ref):
        accumulator = lax.dot(
            lhs_ref[...],
            rhs_ref[...],
            precision=lax.Precision.DEFAULT,
            preferred_element_type=ACCUMULATOR_DTYPE,
        )
        out_ref[...] = accumulator.astype(OUTPUT_DTYPE)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), OUTPUT_DTYPE),
        in_specs=(
            pl.BlockSpec((m, k), lambda: (0, 0)),
            pl.BlockSpec((k, n), lambda: (0, 0)),
        ),
        out_specs=pl.BlockSpec((m, n), lambda: (0, 0)),
        grid=(),
    )
    return jax.jit(call)


def _deterministic_inputs(
    m: int, k: int, n: int, device: jax.Device
) -> tuple[jax.Array, jax.Array, np.ndarray, np.ndarray]:
    # Dyadic values make the FP32 reference accumulation exact for the planned
    # shape range, so a BF16 bit-for-bit result check is meaningful.
    lhs_host = ((np.arange(m * k, dtype=np.int64) % 17) - 8).reshape(m, k)
    rhs_host = ((np.arange(k * n, dtype=np.int64) % 13) - 6).reshape(k, n)
    lhs_host = (lhs_host.astype(np.float32) / 8.0).astype(np.float32)
    rhs_host = (rhs_host.astype(np.float32) / 8.0).astype(np.float32)

    lhs = jax.device_put(jnp.asarray(lhs_host, dtype=INPUT_DTYPE), device)
    rhs = jax.device_put(jnp.asarray(rhs_host, dtype=INPUT_DTYPE), device)
    lhs_effective = np.asarray(jax.device_get(lhs), dtype=np.float32)
    rhs_effective = np.asarray(jax.device_get(rhs), dtype=np.float32)
    return lhs, rhs, lhs_effective, rhs_effective


def _timed_call(
    compiled_call: Any, lhs: jax.Array, rhs: jax.Array
) -> tuple[jax.Array, float]:
    start_ns = time.perf_counter_ns()
    result = compiled_call(lhs, rhs)
    jax.block_until_ready(result)
    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
    return result, elapsed_ms


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_dir.expanduser().resolve()
    case_id = f"m{args.m}_k{args.k}_n{args.n}"
    case_dir = artifact_root / "cases" / case_id
    trace_dir = case_dir / "profiling" / "xprof"
    case_dir.mkdir(parents=True, exist_ok=True)

    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices")
    selected_device = devices[0]

    compiled_call = _build_single_program_dot(args.m, args.k, args.n)
    lhs, rhs, lhs_effective, rhs_effective = _deterministic_inputs(
        args.m, args.k, args.n, selected_device
    )

    # The first completed invocation forces compilation and is reported
    # separately from steady-state measurements.
    result, compile_and_first_run_ms = _timed_call(compiled_call, lhs, rhs)
    warmup_ms = []
    for _ in range(args.warmup):
        result, elapsed_ms = _timed_call(compiled_call, lhs, rhs)
        warmup_ms.append(elapsed_ms)

    xprof_enabled = os.environ.get("MXU_ENABLE_XPROF", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    if xprof_enabled:
        trace_dir.mkdir(parents=True, exist_ok=True)
        trace_context = jax.profiler.trace(
            str(trace_dir), create_perfetto_link=False
        )
    else:
        trace_context = nullcontext()

    latency_ms = []
    with trace_context:
        for step in range(args.repeat):
            with jax.profiler.StepTraceAnnotation(
                "tensorcore_mxu_lax_dot", step_num=step
            ):
                result, elapsed_ms = _timed_call(compiled_call, lhs, rhs)
            latency_ms.append(elapsed_ms)

    actual = np.asarray(jax.device_get(result), dtype=np.float32)
    reference_fp32 = lhs_effective @ rhs_effective
    expected = np.asarray(
        reference_fp32, dtype=ml_dtypes.bfloat16
    ).astype(np.float32)
    mismatch_mask = actual != expected
    mismatch_count = int(np.count_nonzero(mismatch_mask))
    max_abs_error = float(np.max(np.abs(actual - expected), initial=0.0))
    correctness_passed = mismatch_count == 0

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    packages = {
        name: _package_version(name)
        for name in ("jax", "jaxlib", "libtpu", "numpy", "ml_dtypes")
    }
    environment = {
        "schema_version": 1,
        "captured_at": now,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "jax": {
            "default_backend": jax.default_backend(),
            "process_index": jax.process_index(),
            "process_count": jax.process_count(),
            "local_device_count": jax.local_device_count(),
            "device_count": jax.device_count(),
            "devices": [_device_metadata(device) for device in devices],
            "selected_device": _device_metadata(selected_device),
        },
        "compiler_environment": {
            key: os.environ.get(key)
            for key in (
                "LIBTPU_INIT_ARGS",
                "XLA_FLAGS",
                "JAX_COMPILATION_CACHE_DIR",
            )
        },
        "falcon": {
            key: os.environ.get(key)
            for key in (
                "FALCON_EXP_ID",
                "FALCON_JOB_ID",
                "FALCON_ARTIFACT_URI",
                "FALCON_RANK",
                "FALCON_WORLD_SIZE",
            )
        },
        "source": {
            "repository": os.environ.get("MXU_SOURCE_REPOSITORY"),
            "ref": os.environ.get("MXU_SOURCE_REF"),
            "commit": os.environ.get("MXU_SOURCE_COMMIT"),
        },
    }

    manifest = {
        "schema_version": 1,
        "workflow": "operator-optimization",
        "experiment": "tensorcore_mxu_model",
        "experiment_id": os.environ.get("FALCON_EXP_ID"),
        "operator_family": "tensorcore_mxu",
        "operator_name": KERNEL_NAME,
        "case_id": case_id,
        "dimensions": {"m": args.m, "k": args.k, "n": args.n},
        "artifact_contract": "tensorcore_mxu_model.v1",
    }

    case = {
        "schema_version": 1,
        "case_id": case_id,
        "parent_case_id": None,
        "created_at": now,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "dtypes": {
            "lhs": str(INPUT_DTYPE),
            "rhs": str(INPUT_DTYPE),
            "accumulator": str(ACCUMULATOR_DTYPE),
            "output": str(OUTPUT_DTYPE),
        },
        "kernel": {
            "name": KERNEL_NAME,
            "pallas_grid": [],
            "pallas_program_count": 1,
            "block_shapes": {
                "lhs": [args.m, args.k],
                "rhs": [args.k, args.n],
                "output": [args.m, args.n],
            },
            "source_level_padding": False,
            "source_level_tiling": False,
            "multicore_parallelism_opt_in": False,
            "dot_primitive": "jax.lax.dot",
            "dot_precision": "DEFAULT",
            "preferred_element_type": "float32",
        },
        "benchmark": {
            "warmup": args.warmup,
            "repeat": args.repeat,
            "xprof_enabled": xprof_enabled,
            "xprof_directory": (
                str(trace_dir.relative_to(artifact_root))
                if xprof_enabled
                else None
            ),
        },
        "selected_device": _device_metadata(selected_device),
    }

    total_flops = 2 * args.m * args.k * args.n
    tflops = [
        total_flops / (elapsed_ms / 1_000.0) / 1_000_000_000_000.0
        for elapsed_ms in latency_ms
    ]
    metrics = {
        "schema_version": 1,
        "case_id": case_id,
        "compile_and_first_run_ms": compile_and_first_run_ms,
        "warmup_latency_ms": warmup_ms,
        "latency_ms": latency_ms,
        "latency_summary_ms": {
            "min": min(latency_ms),
            "max": max(latency_ms),
            "mean": statistics.fmean(latency_ms),
            "median": statistics.median(latency_ms),
            "p50": _percentile(latency_ms, 50),
            "p90": _percentile(latency_ms, 90),
        },
        "work": {
            "flops": total_flops,
            "tflops": tflops,
            "mean_tflops": statistics.fmean(tflops),
        },
        "correctness": {
            "passed": correctness_passed,
            "comparison": "bitwise equality after BF16 output rounding",
            "mismatch_count": mismatch_count,
            "max_abs_error": max_abs_error,
        },
    }

    _write_json(artifact_root / "manifest.json", manifest)
    _write_json(artifact_root / "environment.json", environment)
    _write_json(case_dir / "manifest.json", manifest)
    _write_json(case_dir / "environment.json", environment)
    _write_json(case_dir / "case.json", case)
    _write_json(case_dir / "metrics.json", metrics)

    print(
        json.dumps(
            {
                "artifact_root": str(artifact_root),
                "case_id": case_id,
                "correctness_passed": correctness_passed,
                "mean_latency_ms": metrics["latency_summary_ms"]["mean"],
                "mean_tflops": metrics["work"]["mean_tflops"],
            },
            sort_keys=True,
        )
    )
    if not correctness_passed:
        raise RuntimeError(
            f"correctness failed: {mismatch_count} mismatches, "
            f"max_abs_error={max_abs_error}"
        )


if __name__ == "__main__":
    main()
