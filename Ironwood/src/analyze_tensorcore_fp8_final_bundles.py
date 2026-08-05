#!/usr/bin/env python3
"""Compare FP8 and BF16 MXU instructions in final bundled JF LLO."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"^m(?P<m>\d+)_k(?P<k>\d+)_n(?P<n>\d+)$")
BUNDLE_RE = re.compile(
    r"^\s*(?P<index>0x[0-9a-f]+|\d+)\s+:", re.MULTILINE
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze FP8 MXU final bundles and optionally compare BF16."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--bf16-artifact-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _strip_block_comments(text: str) -> str:
    result: list[str] = []
    depth = 0
    index = 0
    while index < len(text):
        if text.startswith("/*", index):
            depth += 1
            index += 2
        elif depth and text.startswith("*/", index):
            depth -= 1
            index += 2
        else:
            char = text[index]
            if depth == 0 or char == "\n":
                result.append(char)
            index += 1
    return "".join(result)


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _by_mxu(text: str, opcode_prefix: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for opcode in re.findall(
        rf"\b{opcode_prefix}\.[^\s;}}]+", text, flags=re.IGNORECASE
    ):
        match = re.search(r"\.mxu(\d+)\b", opcode)
        if match:
            counts[f"mxu{match.group(1)}"] += 1
    return dict(sorted(counts.items()))


def _operand_registers(text: str) -> dict[int, str]:
    registers: dict[int, str] = {}
    pattern = re.compile(
        r"%(?P<register>s\d+_s\d+)\s*=\s*"
        r"inlined_call_operand\.(?:vmem|hbm)\s+"
        r"\[shape:.*?index:\s*(?P<index>\d+),",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        registers[int(match.group("index"))] = match.group("register")
    return registers


def _vld_from_register(text: str, register: str | None) -> int:
    if register is None:
        return 0
    return _count(rf"=\s+vld\b[^;}}]*%{re.escape(register)}\b", text)


def _mrb_bases(text: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for opcode in re.findall(
        r"\bvmatmul\.[^\s;}}]+", text, flags=re.IGNORECASE
    ):
        match = re.search(r"mrb\[(\d+)\]\.mxu(\d+)", opcode)
        if match:
            result.setdefault(f"mxu{match.group(2)}", []).append(
                int(match.group(1))
            )
    return dict(sorted(result.items()))


def _base_step_candidates(bases: dict[str, list[int]]) -> list[int]:
    return sorted(
        {
            current - previous
            for values in bases.values()
            for previous, current in zip(values, values[1:])
            if current > previous
        }
    )


def _input_dtype(text: str) -> str | None:
    match = re.search(
        r"inlined_call_operand\.(?:vmem|hbm)\s+"
        r"\[shape:\s*([^\[]+)\[[^\]]+\],\s*index:\s*0,",
        text,
    )
    return match.group(1) if match else None


def _analyze_bundle(path: Path) -> dict[str, Any]:
    case_id = path.parents[2].name
    match = CASE_RE.fullmatch(case_id)
    if not match:
        raise ValueError(f"unexpected case directory: {case_id}")
    text = _strip_block_comments(
        path.read_text(encoding="utf-8", errors="replace")
    )
    operands = _operand_registers(text)
    vld = _count(r"=\s+vld\b", text)
    lhs_vld = _vld_from_register(text, operands.get(0))
    rhs_vld = _vld_from_register(text, operands.get(1))
    push_variants = Counter(
        re.findall(r"\bvmatpush(\d+)\.", text, flags=re.IGNORECASE)
    )
    bases = _mrb_bases(text)
    bundle_indices = [
        int(item, 0) for item in BUNDLE_RE.findall(text)
    ]
    vector_registers = {
        int(item) for item in re.findall(r"_v(\d+)\b", text)
    }
    return {
        "case_id": case_id,
        "shape": {
            key: int(match.group(key)) for key in ("m", "k", "n")
        },
        "input_dtype": _input_dtype(text),
        "vld": {
            "total": vld,
            "lhs": lhs_vld,
            "rhs": rhs_vld,
            "other": vld - lhs_vld - rhs_vld,
        },
        "mxu": {
            "vmatprep_subr": _count(r"\bvmatprep\.subr\.", text),
            "vmatprep_subr_by_mxu": _by_mxu(
                text, r"vmatprep\.subr"
            ),
            "vmatpush": sum(push_variants.values()),
            "vmatpush_variants": dict(sorted(push_variants.items())),
            "vmatpush_by_mxu": _by_mxu(text, r"vmatpush\d+"),
            "vmatprep_mubr": _count(r"\bvmatprep\.mubr\.", text),
            "vmatprep_mubr_by_mxu": _by_mxu(
                text, r"vmatprep\.mubr"
            ),
            "vmatmul": _count(r"\bvmatmul\.", text),
            "vmatmul_by_mxu": _by_mxu(text, "vmatmul"),
            "vmatmul_masked": _count(
                r"\bvmatmul\.[^\s;}}]*\.msk\.", text
            ),
            "mrb_bases": bases,
            "mrb_positive_step_candidates": _base_step_candidates(
                bases
            ),
            "vpop": _count(r"\bvpop\.f32\.", text),
        },
        "vector": {
            "vadd_f32": _count(r"=\s+vadd\.f32\b", text),
            "vpack_bf16": _count(r"=\s+vpack\.c\.bf16\b", text),
            "vst": _count(r"=\s+vst(?:\.|\s)", text),
            "mask_ops": _count(
                r"=\s+(?:vcmask|vsmask|vmand|vmor|vsel|vcombine\.)",
                text,
            ),
        },
        "schedule": {
            "bundle_count": len(bundle_indices),
            "first_bundle": min(bundle_indices) if bundle_indices else None,
            "last_bundle": max(bundle_indices) if bundle_indices else None,
        },
        "registers": {
            "vector_count": len(vector_registers),
            "first_vector": min(vector_registers)
            if vector_registers
            else None,
            "last_vector": max(vector_registers)
            if vector_registers
            else None,
        },
        "source": str(path),
        "size_bytes": path.stat().st_size,
    }


def _find_cases(root: Path) -> dict[str, dict[str, Any]]:
    paths = sorted(root.glob("cases/*/compiler/llo/final_bundle.llo"))
    return {
        case["case_id"]: case for case in map(_analyze_bundle, paths)
    }


def _pair(before: int, after: int) -> str:
    return f"{before}→{after}"


def _mxu_pair(values: dict[str, int]) -> str:
    return f"{values.get('mxu0', 0)}/{values.get('mxu1', 0)}"


def _write_tsv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "input_dtype",
        "vld_total",
        "vld_lhs",
        "vld_rhs",
        "vld_other",
        "vmatprep_subr",
        "vmatprep_subr_mxu0",
        "vmatprep_subr_mxu1",
        "vmatpush1",
        "vmatpush3",
        "vmatprep_mubr",
        "vmatmul",
        "vmatmul_mxu0",
        "vmatmul_mxu1",
        "mrb_positive_step_candidates",
        "vpop",
        "vadd_f32",
        "vpack_bf16",
        "mask_ops",
        "bundle_count",
        "vector_register_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for case in cases:
            mxu = case["mxu"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "input_dtype": case["input_dtype"],
                    "vld_total": case["vld"]["total"],
                    "vld_lhs": case["vld"]["lhs"],
                    "vld_rhs": case["vld"]["rhs"],
                    "vld_other": case["vld"]["other"],
                    "vmatprep_subr": mxu["vmatprep_subr"],
                    "vmatprep_subr_mxu0": mxu[
                        "vmatprep_subr_by_mxu"
                    ].get("mxu0", 0),
                    "vmatprep_subr_mxu1": mxu[
                        "vmatprep_subr_by_mxu"
                    ].get("mxu1", 0),
                    "vmatpush1": mxu["vmatpush_variants"].get("1", 0),
                    "vmatpush3": mxu["vmatpush_variants"].get("3", 0),
                    "vmatprep_mubr": mxu["vmatprep_mubr"],
                    "vmatmul": mxu["vmatmul"],
                    "vmatmul_mxu0": mxu["vmatmul_by_mxu"].get(
                        "mxu0", 0
                    ),
                    "vmatmul_mxu1": mxu["vmatmul_by_mxu"].get(
                        "mxu1", 0
                    ),
                    "mrb_positive_step_candidates": ",".join(
                        str(item)
                        for item in mxu["mrb_positive_step_candidates"]
                    ),
                    "vpop": mxu["vpop"],
                    "vadd_f32": case["vector"]["vadd_f32"],
                    "vpack_bf16": case["vector"]["vpack_bf16"],
                    "mask_ops": case["vector"]["mask_ops"],
                    "bundle_count": case["schedule"]["bundle_count"],
                    "vector_register_count": case["registers"][
                        "vector_count"
                    ],
                }
            )


def _write_report(
    path: Path,
    fp8: dict[str, dict[str, Any]],
    bf16: dict[str, dict[str, Any]],
) -> None:
    cases = [fp8[key] for key in sorted(fp8)]
    baseline = fp8.get("m256_k256_n256")
    bf16_baseline = bf16.get("m256_k256_n256")
    lines = [
        "# Ironwood FP8 MXU final bundle 对照\n",
        "\n",
        "输入为 float8_e4m3fn，累加为 FP32，输出仍为 BF16。"
        "以下只统计最终 JF bundle 中真实出现的指令。\n",
        "\n",
        "## 直接结论\n",
        "\n",
    ]
    if baseline and bf16_baseline:
        f8m = baseline["mxu"]
        bfm = bf16_baseline["mxu"]
        lines.extend(
            [
                "- FP8 使用 f8e4m3/f8e4m3fn 版的 subr、push、mubr 和 "
                "vmatmul。MRB 仍为 FP32，输出仍走 vpop.f32 → "
                "vpack.c.bf16 → vst。\n",
                "- 256³ 中 A/B 的 vld 从 "
                f"{bf16_baseline['vld']['lhs']}/"
                f"{bf16_baseline['vld']['rhs']} 降到 "
                f"{baseline['vld']['lhs']}/{baseline['vld']['rhs']}；"
                "同一个 4 KiB vreg 能放 4096 个 FP8，但只能放 "
                "2048 个 BF16。\n",
                "- 256³ 的 B staging 从 "
                f"{bfm['vmatprep_subr']} 次 subr + "
                f"{bfm['vmatpush']} 次 push 降为 "
                f"{f8m['vmatprep_subr']} + {f8m['vmatpush']}；"
                "vmatmul 从 "
                f"{bfm['vmatmul']} 降为 {f8m['vmatmul']}。\n",
                "- FP8 的一次 A 发射由一个 vmatprep.mubr vreg 与"
                "紧随其后的 vmatmul operand vreg 配成 32 行；所以"
                "一条 FP8 vmatmul 覆盖 A[32,256] × B[256,256]，"
                "而 BF16 对应 16 行。\n",
                "- FP8 基线每个 MXU 的 MRB base 为 "
                f"{f8m['mrb_bases'].get('mxu0', [])}，步长 8；"
                "BF16 步长为 4。一次 FP8 发射产生 8 个 FP32 MRB "
                "fragment，因此最终 vpop 总数没有减半。\n",
            ]
        )
    k256 = fp8.get("m256_k256_n256")
    k257 = fp8.get("m256_k257_n256")
    k512 = fp8.get("m256_k512_n256")
    k513 = fp8.get("m256_k513_n256")
    if k256 and k257 and k512 and k513:
        lines.extend(
            [
                "- K panel 仍为 256，不是 512：K=256/257/512/513 的 "
                "vmatmul 分别为 "
                f"{k256['mxu']['vmatmul']}/"
                f"{k257['mxu']['vmatmul']}/"
                f"{k512['mxu']['vmatmul']}/"
                f"{k513['mxu']['vmatmul']}。FP8 双路体现在一次发射"
                "并行处理两个 16-row M 条带，而不是把 K 深度翻倍。\n",
                f"- K=512 仍有 {k512['vector']['vadd_f32']} 条 "
                "vadd.f32：两个 K panel 分到两个 MXU 后，两个部分和"
                "仍需在向量侧合并；FP8 不改变这条调度规则。\n",
            ]
        )
    lines.extend(
        [
            "\n",
            "## FP8 case 统计\n",
            "\n",
            "| Case | vld A/B/其他 | subr（mxu0/1） | push1/push3 | "
            "mubr | vmatmul（mxu0/1） | MRB 正步长 | vpop | vadd | "
            "mask ops | bundles |\n",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for case in cases:
        mxu = case["mxu"]
        lines.append(
            f"| {case['case_id']} "
            f"| {case['vld']['lhs']}/{case['vld']['rhs']}/"
            f"{case['vld']['other']} "
            f"| {mxu['vmatprep_subr']} "
            f"({_mxu_pair(mxu['vmatprep_subr_by_mxu'])}) "
            f"| {mxu['vmatpush_variants'].get('1', 0)}/"
            f"{mxu['vmatpush_variants'].get('3', 0)} "
            f"| {mxu['vmatprep_mubr']} "
            f"| {mxu['vmatmul']} "
            f"({_mxu_pair(mxu['vmatmul_by_mxu'])}) "
            f"| {mxu['mrb_positive_step_candidates']} "
            f"| {mxu['vpop']} "
            f"| {case['vector']['vadd_f32']} "
            f"| {case['vector']['mask_ops']} "
            f"| {case['schedule']['bundle_count']} |\n"
        )
    comparable = sorted(set(fp8) & set(bf16))
    if comparable:
        lines.extend(
            [
                "\n",
                "## 同 shape：BF16 → FP8\n",
                "\n",
                "| Case | vld | subr | push | mubr | vmatmul | vpop | "
                "vadd |\n",
                "|---|---:|---:|---:|---:|---:|---:|---:|\n",
            ]
        )
        for case_id in comparable:
            old, new = bf16[case_id], fp8[case_id]
            old_mxu, new_mxu = old["mxu"], new["mxu"]
            lines.append(
                f"| {case_id} "
                f"| {_pair(old['vld']['total'], new['vld']['total'])} "
                f"| {_pair(old_mxu['vmatprep_subr'], new_mxu['vmatprep_subr'])} "
                f"| {_pair(old_mxu['vmatpush'], new_mxu['vmatpush'])} "
                f"| {_pair(old_mxu['vmatprep_mubr'], new_mxu['vmatprep_mubr'])} "
                f"| {_pair(old_mxu['vmatmul'], new_mxu['vmatmul'])} "
                f"| {_pair(old_mxu['vpop'], new_mxu['vpop'])} "
                f"| {_pair(old['vector']['vadd_f32'], new['vector']['vadd_f32'])} "
                "|\n"
            )
    lines.extend(
        [
            "\n",
            "## 边界\n",
            "\n",
            "- final bundle 能证明静态发射数、物理 vreg、MRB 地址和"
            "数据路径；不能单独证明每条指令的动态 cycle。\n",
            "- Falcon 的 pallas-llo-analysis 能发现本实验的 8 份 "
            "LLO，但把 MXU instruction family 统计成 0；当前插件"
            "没有识别 f8e4m3/f8e4m3fn opcode。因此本报告直接解析"
            "最终 bundle，不采用插件的零计数。\n",
            "- 本实验没有加入缩放/dequantize，只比较纯 "
            "FP8×FP8 → FP32 accumulate → BF16 dot。实际量化 kernel "
            "还会多出 scale 的 VPU 指令。\n",
        ]
    )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    artifact_dir = args.artifact_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else artifact_dir / "analysis"
    )
    fp8 = _find_cases(artifact_dir)
    if not fp8:
        raise FileNotFoundError(f"no final bundles below {artifact_dir}")
    bf16 = (
        _find_cases(args.bf16_artifact_dir.expanduser().resolve())
        if args.bf16_artifact_dir
        else {}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "experiment_id": artifact_dir.name,
        "case_count": len(fp8),
        "cases": [fp8[key] for key in sorted(fp8)],
        "bf16_comparison_experiment": (
            args.bf16_artifact_dir.name
            if args.bf16_artifact_dir
            else None
        ),
        "bf16_cases": [bf16[key] for key in sorted(bf16)],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_tsv(
        output_dir / "instruction_counts.tsv",
        [fp8[key] for key in sorted(fp8)],
    )
    _write_report(output_dir / "analysis.md", fp8, bf16)
    print(
        json.dumps(
            {
                "artifact_dir": str(artifact_dir),
                "case_count": len(fp8),
                "output_dir": str(output_dir),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
