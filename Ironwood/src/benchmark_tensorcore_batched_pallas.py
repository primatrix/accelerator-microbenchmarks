#!/usr/bin/env python3
"""Compile one rank-3 Pallas matmul for MXU batch scheduling analysis.

The leading ``MB`` dimension is intentionally kept inside a single Pallas
program.  The TPU backend must therefore lower ``MB`` independent
``[M, K] @ [K, N]`` products onto the physical two-MXU machine instead of
receiving MB as a Pallas grid dimension.
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
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


INPUT_DTYPE = jnp.bfloat16
ACCUMULATOR_DTYPE = jnp.float32
OUTPUT_DTYPE = jnp.bfloat16
KERNEL_NAME = "pallas_rank3_batched_matmul_single_program"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile one Pallas rank-3 batched matmul case."
    )
    parser.add_argument("--mb", type=int, required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    for name in ("mb", "m", "k", "n"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeat <= 0:
        parser.error("--repeat must be positive")
    return args


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _device_metadata(device: Any) -> dict[str, Any]:
    result = {}
    for field in (
        "id",
        "platform",
        "device_kind",
        "process_index",
        "local_hardware_id",
        "slice_index",
    ):
        value = getattr(device, field, None)
        if value is not None:
            result[field] = value
    result["repr"] = str(device)
    return result


def _build_batched_matmul(mb: int, m: int, k: int, n: int):
    def kernel(lhs_ref, rhs_ref, out_ref):
        # jnp.matmul treats MB as a batch axis and contracts only the final
        # two dimensions.  Both operands vary with MB, so this cannot be
        # represented as one ordinary dense 2-D matmul without block-diagonal
        # padding.
        accumulator = jnp.matmul(
            lhs_ref[...],
            rhs_ref[...],
            preferred_element_type=ACCUMULATOR_DTYPE,
        )
        out_ref[...] = accumulator.astype(OUTPUT_DTYPE)

    call = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((mb, m, n), OUTPUT_DTYPE),
        in_specs=(
            pl.BlockSpec((mb, m, k), lambda: (0, 0, 0)),
            pl.BlockSpec((mb, k, n), lambda: (0, 0, 0)),
        ),
        out_specs=pl.BlockSpec((mb, m, n), lambda: (0, 0, 0)),
        grid=(),
        name=KERNEL_NAME,
    )
    return jax.jit(call)


def _timed_call(compiled_call: Any, lhs: jax.Array, rhs: jax.Array):
    started_ns = time.perf_counter_ns()
    result = compiled_call(lhs, rhs)
    jax.block_until_ready(result)
    elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000.0
    return result, elapsed_ms


def main() -> None:
    args = _parse_args()
    artifact_root = args.artifact_dir.expanduser().resolve()
    case_id = f"mb{args.mb}_m{args.m}_k{args.k}_n{args.n}"
    case_dir = artifact_root / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices")
    selected_device = devices[0]

    compiled_call = _build_batched_matmul(
        args.mb, args.m, args.k, args.n
    )
    lhs = jax.device_put(
        jnp.ones((args.mb, args.m, args.k), dtype=INPUT_DTYPE),
        selected_device,
    )
    rhs = jax.device_put(
        jnp.ones((args.mb, args.k, args.n), dtype=INPUT_DTYPE),
        selected_device,
    )

    result, compile_and_first_run_ms = _timed_call(
        compiled_call, lhs, rhs
    )
    warmup_ms = []
    for _ in range(args.warmup):
        result, elapsed_ms = _timed_call(compiled_call, lhs, rhs)
        warmup_ms.append(elapsed_ms)
    latency_ms = []
    for _ in range(args.repeat):
        result, elapsed_ms = _timed_call(compiled_call, lhs, rhs)
        latency_ms.append(elapsed_ms)

    now = dt.datetime.now(dt.timezone.utc).isoformat()
    shape = {"mb": args.mb, "m": args.m, "k": args.k, "n": args.n}
    case = {
        "schema_version": 1,
        "case_id": case_id,
        "created_at": now,
        "shape": shape,
        "dtypes": {
            "lhs": str(INPUT_DTYPE),
            "rhs": str(INPUT_DTYPE),
            "accumulator": str(ACCUMULATOR_DTYPE),
            "output": str(OUTPUT_DTYPE),
        },
        "kernel": {
            "name": KERNEL_NAME,
            "operation": "batched [MB,M,K] @ [MB,K,N]",
            "pallas_grid": [],
            "pallas_program_count": 1,
            "batch_axis_inside_program": True,
            "block_shapes": {
                "lhs": [args.mb, args.m, args.k],
                "rhs": [args.mb, args.k, args.n],
                "output": [args.mb, args.m, args.n],
            },
        },
    }
    metrics = {
        "schema_version": 1,
        "case_id": case_id,
        **shape,
        "compile_and_first_run_ms": compile_and_first_run_ms,
        "warmup_latency_ms": warmup_ms,
        "latency_ms": latency_ms,
        "latency_mean_ms": statistics.fmean(latency_ms),
        "logical_matmul_count": args.mb,
        "flops": 2 * args.mb * args.m * args.k * args.n,
    }
    environment = {
        "schema_version": 1,
        "captured_at": now,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            name: _package_version(name)
            for name in ("jax", "jaxlib", "libtpu", "numpy")
        },
        "jax": {
            "default_backend": jax.default_backend(),
            "device_count": jax.device_count(),
            "selected_device": _device_metadata(selected_device),
        },
        "compiler_environment": {
            "LIBTPU_INIT_ARGS": os.environ.get("LIBTPU_INIT_ARGS"),
            "XLA_FLAGS": os.environ.get("XLA_FLAGS"),
        },
        "source": {
            "repository": os.environ.get("MXU_SOURCE_REPOSITORY"),
            "ref": os.environ.get("MXU_SOURCE_REF"),
            "commit": os.environ.get("MXU_SOURCE_COMMIT"),
        },
    }

    _write_json(case_dir / "case.json", case)
    _write_json(case_dir / "metrics.json", metrics)
    _write_json(case_dir / "environment.json", environment)
    print(
        json.dumps(
            {
                "case_id": case_id,
                "compile_and_first_run_ms": compile_and_first_run_ms,
                "latency_mean_ms": metrics["latency_mean_ms"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
