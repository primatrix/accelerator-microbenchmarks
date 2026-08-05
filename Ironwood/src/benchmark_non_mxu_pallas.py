#!/usr/bin/env python3
"""Compile and run one non-MXU Pallas compute probe.

The source operation is intentionally kept inside one named Pallas kernel so
the JF final-bundle dump can be associated with a small set of JAX primitives.
Use ``--list-cases`` to print the machine-readable probe catalog.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


VECTOR_SHAPE = (8, 128)
TRANSPOSE_SHAPE = (128, 128)
SCALAR_GRID_SHAPE = (32, 128)


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    family: str
    primitives: tuple[str, ...]
    input_kinds: tuple[str, ...]
    output_dtype: str
    shape: tuple[int, int] = VECTOR_SHAPE
    output_shape: tuple[int, int] = VECTOR_SHAPE
    mode: str = "vector"
    verify: bool = True


def _case(
    case_id: str,
    family: str,
    primitives: tuple[str, ...],
    input_kinds: tuple[str, ...],
    output_dtype: str,
    **kwargs: Any,
) -> CaseSpec:
    return CaseSpec(
        case_id, family, primitives, input_kinds, output_dtype, **kwargs
    )


CASES = {
    spec.case_id: spec
    for spec in (
        _case("add_f32", "vpu", ("lax.add",), ("f32_signed", "f32_other"), "float32"),
        _case("add_bf16", "vpu", ("lax.add",), ("bf16_signed", "bf16_other"), "bfloat16"),
        _case("add_s32", "vpu", ("lax.add",), ("s32", "s32_other"), "int32"),
        _case("sub_f32", "vpu", ("lax.sub",), ("f32_signed", "f32_other"), "float32"),
        _case("sub_bf16", "vpu", ("lax.sub",), ("bf16_signed", "bf16_other"), "bfloat16"),
        _case("sub_s32", "vpu", ("lax.sub",), ("s32", "s32_other"), "int32"),
        _case("mul_f32", "vpu", ("lax.mul",), ("f32_signed", "f32_other"), "float32"),
        _case("mul_bf16", "vpu", ("lax.mul",), ("bf16_signed", "bf16_other"), "bfloat16"),
        _case("mul_u32", "vpu", ("lax.mul",), ("u32", "u32_other"), "uint32"),
        _case("div_f32", "vpu_eup", ("lax.div",), ("f32_positive", "f32_divisor"), "float32"),
        _case("div_s32", "spu_vpu", ("lax.div",), ("s32", "s32_divisor"), "int32"),
        _case("div_u32", "spu_vpu", ("lax.div",), ("u32", "u32_divisor"), "uint32"),
        _case("rem_f32", "vpu", ("lax.rem",), ("f32_positive", "f32_divisor"), "float32"),
        _case("rem_s32", "spu_vpu", ("lax.rem",), ("s32", "s32_divisor"), "int32"),
        _case("rem_u32", "spu_vpu", ("lax.rem",), ("u32", "u32_divisor"), "uint32"),
        _case("minmax_f32", "vpu", ("lax.min", "lax.max"), ("f32_signed", "f32_other"), "float32"),
        _case("minmax_bf16", "vpu", ("lax.min", "lax.max"), ("bf16_signed", "bf16_other"), "bfloat16"),
        _case("minmax_s32", "vpu", ("lax.min", "lax.max"), ("s32", "s32_other"), "int32"),
        _case("bitwise_u32", "vpu", ("lax.and", "lax.or", "lax.xor"), ("u32", "u32_other"), "uint32"),
        _case("shifts", "vpu", ("lax.shift_left", "lax.shift_right_logical", "lax.shift_right_arithmetic"), ("u32", "shift"), "uint32"),
        _case("abs_neg_sign_f32", "vpu", ("lax.abs", "lax.neg", "lax.sign"), ("f32_signed",), "float32"),
        _case("abs_neg_sign_s32", "vpu", ("lax.abs", "lax.neg", "lax.sign"), ("s32",), "int32"),
        _case("clz_popcount", "vpu", ("lax.clz", "lax.population_count"), ("u32",), "uint32"),
        _case("sqrt_rsqrt", "eup", ("lax.sqrt", "lax.rsqrt"), ("f32_positive",), "float32"),
        _case("exp_exp2", "eup", ("lax.exp", "lax.exp2"), ("f32_small",), "float32"),
        _case("log_log1p", "eup", ("lax.log", "lax.log1p"), ("f32_positive",), "float32"),
        _case("sin_cos_tan", "eup", ("lax.sin", "lax.cos", "lax.tan"), ("f32_small",), "float32"),
        _case("tanh_logistic", "eup", ("lax.tanh", "lax.logistic"), ("f32_signed",), "float32"),
        _case("round_ceil_floor", "vpu", ("lax.round", "lax.ceil", "lax.floor"), ("f32_signed",), "float32"),
        _case("pow_nextafter", "vpu_eup", ("lax.pow", "lax.nextafter"), ("f32_positive", "f32_divisor"), "float32"),
        _case("erf_inv", "composite_eup", ("lax.erf_inv",), ("f32_unit",), "float32"),
        _case("compare_select_f32", "predicate_vpu", ("lax.eq", "lax.ne", "lax.lt", "lax.le", "lax.gt", "lax.ge", "lax.select_n"), ("f32_signed", "f32_other"), "float32"),
        _case("compare_select_s32", "predicate_vpu", ("lax.eq", "lax.ne", "lax.lt", "lax.le", "lax.gt", "lax.ge", "lax.select_n"), ("s32", "s32_other"), "int32"),
        _case("isfinite_clamp", "predicate_vpu", ("lax.is_finite", "lax.clamp"), ("f32_signed", "f32_other"), "float32"),
        _case("f32_to_s32", "convert", ("lax.convert_element_type",), ("f32_signed",), "int32"),
        _case("s32_to_f32", "convert", ("lax.convert_element_type",), ("s32",), "float32"),
        _case("f32_to_bf16", "convert", ("lax.convert_element_type",), ("f32_signed",), "bfloat16"),
        _case("bf16_to_f32", "convert", ("lax.convert_element_type",), ("bf16_signed",), "float32"),
        _case("bitcast_f32_u32", "convert", ("lax.bitcast_convert_type",), ("f32_signed",), "uint32"),
        _case("stochastic_bf16", "convert", ("pltpu.stochastic_round",), ("f32_signed", "random_bits"), "bfloat16", verify=False),
        _case("pack_f32_bf16", "pack", ("pltpu.pack_elementwise",), ("f32_signed", "f32_other"), "uint32", verify=False),
        _case("unpack_bf16_f32", "pack", ("pltpu.unpack_elementwise",), ("u32",), "float32", verify=False),
        _case("reduce_sum_f32", "xlu_reduce", ("lax.reduce_sum",), ("f32_signed",), "float32"),
        _case("reduce_minmax_f32", "xlu_reduce", ("lax.reduce_min", "lax.reduce_max"), ("f32_signed",), "float32"),
        _case("reduce_sum_bf16", "xlu_reduce", ("lax.reduce_sum",), ("bf16_signed",), "bfloat16"),
        _case("reduce_minmax_bf16", "xlu_reduce", ("lax.reduce_min", "lax.reduce_max"), ("bf16_signed",), "bfloat16"),
        _case("reduce_s32", "xlu_reduce", ("lax.reduce_sum", "lax.reduce_min", "lax.reduce_max"), ("s32",), "int32"),
        _case("argminmax_f32", "xlu_reduce", ("lax.argmin", "lax.argmax"), ("f32_signed",), "int32"),
        _case("transpose_f32", "xlu", ("lax.transpose",), ("f32_signed",), "float32", shape=TRANSPOSE_SHAPE, output_shape=TRANSPOSE_SHAPE),
        _case("iota_f32", "vpu", ("lax.iota",), ("f32_signed",), "float32"),
        _case("roll_f32", "xlu", ("pltpu.roll",), ("f32_signed",), "float32", verify=False),
        _case("scalar_arith", "spu", ("pl.program_id", "lax.add", "lax.sub", "lax.mul", "lax.div", "lax.rem", "lax.min", "lax.max"), ("f32_signed",), "float32", shape=SCALAR_GRID_SHAPE, output_shape=SCALAR_GRID_SHAPE, mode="scalar_grid"),
        _case("scalar_logic_compare", "spu_predicate", ("pl.program_id", "lax.and", "lax.or", "lax.xor", "lax.shift_left", "lax.shift_right_logical", "lax.lt", "lax.select_n"), ("f32_signed",), "float32", shape=SCALAR_GRID_SHAPE, output_shape=SCALAR_GRID_SHAPE, mode="scalar_grid"),
    )
}


def _dtype(name: str) -> Any:
    return {
        "float32": jnp.float32,
        "bfloat16": jnp.bfloat16,
        "int32": jnp.int32,
        "uint32": jnp.uint32,
    }[name]


def _input(kind: str, shape: tuple[int, int], seed: int) -> jax.Array:
    size = int(np.prod(shape))
    idx = np.arange(size, dtype=np.int64).reshape(shape)
    if kind in {"f32_signed", "bf16_signed"}:
        value = ((idx % 257) - 128).astype(np.float32) / 32.0
    elif kind in {"f32_other", "bf16_other"}:
        value = ((idx * 7 % 251) - 97).astype(np.float32) / 29.0
    elif kind == "f32_positive":
        value = (idx % 127).astype(np.float32) / 64.0 + 0.125
    elif kind == "f32_divisor":
        value = (idx % 31).astype(np.float32) / 16.0 + 0.5
    elif kind == "f32_small":
        value = ((idx % 129) - 64).astype(np.float32) / 128.0
    elif kind == "f32_unit":
        value = ((idx % 101) - 50).astype(np.float32) / 64.0
    elif kind == "s32":
        value = ((idx % 211) - 105).astype(np.int32)
    elif kind == "s32_other":
        value = ((idx * 11 % 199) - 71).astype(np.int32)
    elif kind == "s32_divisor":
        value = (idx % 31 + 1).astype(np.int32)
    elif kind == "u32":
        value = (idx.astype(np.uint64) * np.uint64(2654435761) + seed).astype(np.uint32)
    elif kind == "u32_other":
        value = (idx.astype(np.uint64) * np.uint64(2246822519) + 17 + seed).astype(np.uint32)
    elif kind == "u32_divisor":
        value = (idx % 31 + 1).astype(np.uint32)
    elif kind == "shift":
        value = (idx % 31).astype(np.uint32)
    elif kind == "random_bits":
        value = (idx.astype(np.uint64) * np.uint64(3266489917) + 101 + seed).astype(np.uint32)
    else:
        raise ValueError(f"unknown input kind: {kind}")
    if kind.startswith("bf16"):
        return jnp.asarray(value, dtype=jnp.bfloat16)
    return jnp.asarray(value)


def _broadcast_row(value: jax.Array, shape: tuple[int, int]) -> jax.Array:
    return jnp.broadcast_to(value, shape)


def _apply(case_id: str, *xs: jax.Array) -> jax.Array:
    x = xs[0]
    y = xs[1] if len(xs) > 1 else None
    if case_id.startswith("add_"): return lax.add(x, y)
    if case_id.startswith("sub_"): return lax.sub(x, y)
    if case_id.startswith("mul_"): return lax.mul(x, y)
    if case_id.startswith("div_"): return lax.div(x, y)
    if case_id.startswith("rem_"): return lax.rem(x, y)
    if case_id.startswith("minmax_"): return lax.min(x, y) + lax.max(x, y) * jnp.asarray(3, x.dtype)
    if case_id == "bitwise_u32": return lax.bitwise_and(x, y) ^ lax.bitwise_or(x, y)
    if case_id == "shifts":
        left = lax.shift_left(x, y)
        right = lax.shift_right_logical(x, y)
        arithmetic = lax.shift_right_arithmetic(lax.bitcast_convert_type(x, jnp.int32), lax.bitcast_convert_type(y, jnp.int32))
        return left ^ right ^ lax.bitcast_convert_type(arithmetic, jnp.uint32)
    if case_id.startswith("abs_neg_sign_"): return lax.abs(x) + lax.neg(x) * jnp.asarray(3, x.dtype) + lax.sign(x)
    if case_id == "clz_popcount": return lax.clz(x) + lax.population_count(x)
    if case_id == "sqrt_rsqrt": return lax.sqrt(x) + lax.rsqrt(x)
    if case_id == "exp_exp2": return lax.exp(x) + lax.exp2(x)
    if case_id == "log_log1p": return lax.log(x) + lax.log1p(x)
    if case_id == "sin_cos_tan": return lax.sin(x) + lax.cos(x) + lax.tan(x)
    if case_id == "tanh_logistic": return lax.tanh(x) + lax.logistic(x)
    if case_id == "round_ceil_floor":
        return lax.round(x, lax.RoundingMethod.AWAY_FROM_ZERO) + lax.round(x, lax.RoundingMethod.TO_NEAREST_EVEN) + lax.ceil(x) + lax.floor(x)
    if case_id == "pow_nextafter": return lax.pow(x, y) + lax.nextafter(x, y)
    if case_id == "erf_inv": return lax.erf_inv(x)
    if case_id.startswith("compare_select_"):
        predicates = (lax.eq(x, y), lax.ne(x, y), lax.lt(x, y), lax.le(x, y), lax.gt(x, y), lax.ge(x, y))
        terms = [lax.select(p, x, y) * jnp.asarray(i + 1, x.dtype) for i, p in enumerate(predicates)]
        return sum(terms[1:], terms[0])
    if case_id == "isfinite_clamp":
        finite = lax.is_finite(x)
        clipped = lax.clamp(jnp.asarray(-1.0, x.dtype), x, jnp.asarray(1.0, x.dtype))
        return lax.select(finite, clipped, y)
    if case_id == "f32_to_s32": return lax.convert_element_type(x, jnp.int32)
    if case_id == "s32_to_f32": return lax.convert_element_type(x, jnp.float32)
    if case_id == "f32_to_bf16": return lax.convert_element_type(x, jnp.bfloat16)
    if case_id == "bf16_to_f32": return lax.convert_element_type(x, jnp.float32)
    if case_id == "bitcast_f32_u32": return lax.bitcast_convert_type(x, jnp.uint32)
    if case_id == "stochastic_bf16": return pltpu.stochastic_round(x, y, target_dtype=jnp.bfloat16)
    if case_id == "pack_f32_bf16": return pltpu.pack_elementwise((x, y), packed_dtype=jnp.bfloat16)
    if case_id == "unpack_bf16_f32": return pltpu.unpack_elementwise(x, index=0, packed_dtype=jnp.bfloat16, unpacked_dtype=jnp.float32)
    if case_id == "reduce_sum_f32" or case_id == "reduce_sum_bf16":
        return _broadcast_row(jnp.sum(x, axis=1, keepdims=True), x.shape)
    if case_id.startswith("reduce_minmax_"):
        value = jnp.min(x, axis=1, keepdims=True) + jnp.max(x, axis=1, keepdims=True) * jnp.asarray(3, x.dtype)
        return _broadcast_row(value, x.shape)
    if case_id == "reduce_s32":
        value = jnp.sum(x, axis=1, keepdims=True) + jnp.min(x, axis=1, keepdims=True) + jnp.max(x, axis=1, keepdims=True)
        return _broadcast_row(value, x.shape)
    if case_id == "argminmax_f32":
        value = jnp.argmin(x, axis=1).astype(jnp.int32) + 257 * jnp.argmax(x, axis=1).astype(jnp.int32)
        return jnp.broadcast_to(value[:, None], x.shape)
    if case_id == "transpose_f32": return jnp.transpose(x, (1, 0))
    if case_id == "iota_f32":
        seq = jnp.arange(x.shape[1], dtype=x.dtype)[None, :]
        return x + jnp.broadcast_to(seq, x.shape)
    if case_id == "roll_f32":
        return pltpu.roll(x, 1, axis=1)
    raise KeyError(case_id)


def _scalar_value(case_id: str, pid: jax.Array) -> jax.Array:
    if case_id == "scalar_arith":
        a = (pid + 7) * 11 - 3
        return lax.max(lax.div(a, 3) + lax.rem(a, 5), pid + 1)
    if case_id == "scalar_logic_compare":
        a = lax.bitwise_xor((pid + 1) * 0x1F, 0x55)
        b = lax.bitwise_or(lax.bitwise_and(a, 0x3F), lax.shift_left(pid, 3))
        shifted = lax.shift_right_logical(b, 1)
        return lax.select(pid < 2, b, shifted)
    raise KeyError(case_id)


def _build_call(spec: CaseSpec, interpret: bool) -> Callable[..., jax.Array]:
    name = f"non_mxu_{spec.case_id}"
    if spec.mode == "scalar_grid":
        block_shape = VECTOR_SHAPE
        def kernel(x_ref, out_ref):
            value = _scalar_value(spec.case_id, pl.program_id(0))
            out_ref[...] = x_ref[...] + value.astype(jnp.float32)
        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(spec.output_shape, _dtype(spec.output_dtype)),
            in_specs=(pl.BlockSpec(block_shape, lambda i: (i, 0)),),
            out_specs=pl.BlockSpec(block_shape, lambda i: (i, 0)),
            grid=(4,),
            interpret=interpret,
            name=name,
        )
    else:
        def kernel(*refs):
            *input_refs, out_ref = refs
            out_ref[...] = _apply(spec.case_id, *(ref[...] for ref in input_refs))
        input_specs = tuple(pl.BlockSpec(spec.shape, lambda: (0, 0)) for _ in spec.input_kinds)
        call = pl.pallas_call(
            kernel,
            out_shape=jax.ShapeDtypeStruct(spec.output_shape, _dtype(spec.output_dtype)),
            in_specs=input_specs,
            out_specs=pl.BlockSpec(spec.output_shape, lambda: (0, 0)),
            grid=(),
            interpret=interpret,
            name=name,
        )
    return jax.jit(call)


def _reference(spec: CaseSpec, inputs: list[jax.Array]) -> np.ndarray | None:
    if not spec.verify:
        return None
    cpu_devices = jax.devices("cpu")
    if not cpu_devices:
        return None
    with jax.default_device(cpu_devices[0]):
        cpu_inputs = [jax.device_put(value, cpu_devices[0]) for value in inputs]
        if spec.mode == "scalar_grid":
            blocks = []
            for pid in range(4):
                value = _scalar_value(spec.case_id, jnp.asarray(pid, jnp.int32))
                blocks.append(cpu_inputs[0][pid * 8:(pid + 1) * 8] + value.astype(jnp.float32))
            result = jnp.concatenate(blocks, axis=0)
        else:
            result = _apply(spec.case_id, *cpu_inputs)
        return np.asarray(jax.device_get(result))


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=sorted(CASES))
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--interpret", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()
    if args.list_cases:
        print(json.dumps([asdict(CASES[name]) for name in sorted(CASES)], indent=2))
        return
    if not args.case or args.artifact_dir is None:
        parser.error("--case and --artifact-dir are required unless --list-cases is used")

    spec = CASES[args.case]
    inputs = [_input(kind, spec.shape, i + 1) for i, kind in enumerate(spec.input_kinds)]
    call = _build_call(spec, args.interpret)
    started = time.perf_counter()
    result = call(*inputs)
    jax.block_until_ready(result)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    actual = np.asarray(jax.device_get(result))
    expected = _reference(spec, inputs)
    if expected is None:
        correctness = {"checked": False, "passed": None}
    elif np.issubdtype(actual.dtype, np.integer):
        passed = bool(np.array_equal(actual, expected))
        correctness = {"checked": True, "passed": passed, "max_abs_error": 0 if passed else None}
    else:
        actual_f32 = actual.astype(np.float32)
        expected_f32 = expected.astype(np.float32)
        error = float(np.nanmax(np.abs(actual_f32 - expected_f32)))
        passed = bool(np.allclose(actual_f32, expected_f32, rtol=2e-2, atol=2e-2, equal_nan=True))
        correctness = {"checked": True, "passed": passed, "max_abs_error": error}

    case_dir = args.artifact_dir.resolve() / "cases" / spec.case_id
    payload = {
        "schema_version": 1,
        "case": asdict(spec),
        "kernel_name": f"non_mxu_{spec.case_id}",
        "elapsed_ms_compile_and_run": elapsed_ms,
        "correctness": correctness,
        "result": {
            "shape": list(actual.shape),
            "dtype": str(actual.dtype),
            "sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
            "sample": actual.reshape(-1)[:8].astype(np.float32).tolist(),
        },
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "jaxlib": _version("jaxlib"),
            "libtpu": _version("libtpu"),
            "devices": [str(device) for device in jax.devices()],
        },
    }
    _write_json(case_dir / "metrics.json", payload)
    print(json.dumps(payload, sort_keys=True))
    if correctness["checked"] and not correctness["passed"]:
        raise RuntimeError(f"correctness check failed for {spec.case_id}")


if __name__ == "__main__":
    main()
