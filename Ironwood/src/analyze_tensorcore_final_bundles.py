#!/usr/bin/env python3
"""Analyze final bundled JF LLO from the exact-shape MXU sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


CASE_RE = re.compile(r"^m(?P<m>\d+)_k(?P<k>\d+)_n(?P<n>\d+)$")
BUNDLE_RE = re.compile(r"^\s*(?P<index>0x[0-9a-f]+|\d+)\s+:", re.MULTILINE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze final_bundle.llo files from an MXU shape sweep."
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        required=True,
        help="experiment artifact root containing cases/",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to <artifact-dir>/analysis",
    )
    return parser.parse_args()


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _strip_block_comments(text: str) -> str:
    """Remove nested LLO block comments while preserving line boundaries."""
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


def _register_numbers(pattern: str, text: str) -> list[int]:
    return sorted({int(value) for value in re.findall(pattern, text)})


def _mrb_occurrences(text: str, opcode_prefix: str) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for opcode in re.findall(rf"{opcode_prefix}\.[^\s;}}]+", text):
        match = re.search(r"mrb\[(\d+)\]\.mxu(\d+)", opcode)
        if match:
            index, mxu = int(match.group(1)), match.group(2)
            result.setdefault(mxu, []).append(index)
    return {
        f"mxu{mxu}": indices
        for mxu, indices in sorted(result.items())
    }


def _mrb_indices(text: str, opcode_prefix: str) -> dict[str, list[int]]:
    return {
        mxu: sorted(set(indices))
        for mxu, indices in _mrb_occurrences(text, opcode_prefix).items()
    }


def _mrb_lifetime(text: str) -> dict[str, dict[str, int]]:
    """Model architecturally named MRB entries between vmatmul and vpop."""
    state: dict[str, dict[str, Any]] = {}
    for opcode in re.findall(r"\b(?:vmatmul|vpop)\.[^\s;}}]+", text):
        match = re.search(r"mrb\[(\d+)\]\.mxu(\d+)", opcode)
        if not match:
            continue
        index, mxu_number = int(match.group(1)), match.group(2)
        mxu = f"mxu{mxu_number}"
        mxu_state = state.setdefault(
            mxu,
            {
                "live": set(),
                "seen": set(),
                "peak_live": 0,
                "accumulated_entry_writes": 0,
                "recycled_entry_writes": 0,
                "pop_without_live_write": 0,
            },
        )
        live: set[int] = mxu_state["live"]
        seen: set[int] = mxu_state["seen"]
        if opcode.startswith("vmatmul."):
            for entry in range(index, index + 4):
                if entry in live:
                    mxu_state["accumulated_entry_writes"] += 1
                elif entry in seen:
                    mxu_state["recycled_entry_writes"] += 1
                live.add(entry)
                seen.add(entry)
            mxu_state["peak_live"] = max(mxu_state["peak_live"], len(live))
        else:
            if index in live:
                live.remove(index)
            else:
                mxu_state["pop_without_live_write"] += 1

    return {
        mxu: {
            "peak_live": values["peak_live"],
            "final_live": len(values["live"]),
            "accumulated_entry_writes": values[
                "accumulated_entry_writes"
            ],
            "recycled_entry_writes": values["recycled_entry_writes"],
            "pop_without_live_write": values["pop_without_live_write"],
        }
        for mxu, values in sorted(state.items())
    }


def _mrb_sequence_stats(
    occurrences: dict[str, list[int]],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for mxu, indices in occurrences.items():
        unique = set(indices)
        result[mxu] = {
            "count": len(indices),
            "unique_count": len(unique),
            "address_span": max(unique) + 1 if unique else 0,
            "reuses": len(indices) - len(unique),
            "address_descents": sum(
                current < previous
                for previous, current in zip(indices, indices[1:])
            ),
            "wrap_to_zero": sum(
                current == 0 and previous > 0
                for previous, current in zip(indices, indices[1:])
            ),
        }
    return result


def _opcode_count_by_mxu(text: str, opcode_prefix: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for opcode in re.findall(rf"\b{opcode_prefix}\.[^\s;}}]+", text):
        match = re.search(r"\.mxu(\d+)", opcode)
        if match:
            counts[f"mxu{match.group(1)}"] += 1
    return dict(sorted(counts.items()))


def _output_kind(text: str) -> str | None:
    match = re.search(
        r"inlined_call_operand\.(vmem|hbm)"
        r"\s+\[shape:[^\]]+\],\s+index:\s*2,\s+kind:\s*output",
        text,
    )
    return match.group(1) if match else None


def _parallel_axis(k_panels: int, n_panels: int) -> str:
    if n_panels > 1:
        if n_panels % 2:
            return "N→K" if k_panels > 1 else "N→M"
        return "N"
    if k_panels > 1:
        return "K"
    return "M"


def _analyze_case(bundle_path: Path) -> dict[str, Any]:
    case_id = bundle_path.parents[2].name
    case_match = CASE_RE.fullmatch(case_id)
    if not case_match:
        raise ValueError(f"unexpected case directory: {case_id}")
    m, k, n = (
        int(case_match.group("m")),
        int(case_match.group("k")),
        int(case_match.group("n")),
    )
    text = bundle_path.read_text(encoding="utf-8", errors="replace")
    executable_text = _strip_block_comments(text)
    bundle_indices = [
        int(match.group("index"), 0) for match in BUNDLE_RE.finditer(text)
    ]
    m_tiles = math.ceil(m / 16)
    k_panels = math.ceil(k / 256)
    n_panels = math.ceil(n / 256)
    expected_vmatmul = m_tiles * k_panels * n_panels
    result_groups = n_panels + int(n_panels % 2 == 1 and k_panels > 1)
    expected_vpop = 4 * m_tiles * result_groups

    push_variants = Counter(
        int(variant)
        for variant in re.findall(r"\bvmatpush(\d+)\.bf16", text)
    )
    vmatmul = _count(r"\bvmatmul\.mubr(?:\.msk)?\.bf16", text)
    vpop = _count(r"\bvpop\.f32", text)
    vector_registers = _register_numbers(r"_v(\d+)\b", text)
    scalar_registers = _register_numbers(r"_s(\d+)\b", text)
    predicate_registers = _register_numbers(r"_p(\d+)\b", text)
    mask_registers = _register_numbers(r"_vm(\d+)\b", text)
    mrb_pop_occurrences = _mrb_occurrences(text, "vpop")
    mrb_pop_indices = {
        mxu: sorted(set(indices))
        for mxu, indices in mrb_pop_occurrences.items()
    }
    mrb_pop_stats = _mrb_sequence_stats(mrb_pop_occurrences)
    mrb_address_span = {
        mxu: stats["address_span"]
        for mxu, stats in mrb_pop_stats.items()
    }

    result = {
        "case_id": case_id,
        "shape": {"m": m, "k": k, "n": n},
        "decomposition": {
            "m_tiles_16": m_tiles,
            "k_panels_256": k_panels,
            "n_panels_256": n_panels,
            "parallel_axis": _parallel_axis(k_panels, n_panels),
            "mrb_result_groups": result_groups,
            "m_tail": m % 16,
            "k_tail": k % 256,
            "n_tail": n % 256,
        },
        "schedule": {
            "bundle_count": len(bundle_indices),
            "first_bundle": min(bundle_indices) if bundle_indices else None,
            "last_bundle": max(bundle_indices) if bundle_indices else None,
        },
        "mxu": {
            "vmatprep_subr": _count(
                r"\bvmatprep\.subr(?:\.msk)?\.bf16", text
            ),
            "vmatprep_subr_by_mxu": _opcode_count_by_mxu(
                text, r"vmatprep\.subr"
            ),
            "vmatpush": sum(push_variants.values()),
            "vmatpush_by_mxu": _opcode_count_by_mxu(
                text, r"vmatpush\d+"
            ),
            "vmatpush_variants": {
                str(variant): count
                for variant, count in sorted(push_variants.items())
            },
            "vmatprep_mubr": _count(
                r"\bvmatprep\.mubr(?:\.msk)?\.bf16", text
            ),
            "vmatprep_mubr_masked": _count(
                r"\bvmatprep\.mubr\.msk\.bf16", text
            ),
            "vmatmul": vmatmul,
            "vmatmul_by_mxu": _opcode_count_by_mxu(text, "vmatmul"),
            "vmatmul_masked": _count(
                r"\bvmatmul\.mubr\.msk\.bf16", text
            ),
            "expected_vmatmul": expected_vmatmul,
            "vmatmul_matches_model": vmatmul == expected_vmatmul,
            "vpop": vpop,
            "expected_vpop": expected_vpop,
            "vpop_matches_model": vpop == expected_vpop,
            "mrb_base_indices": _mrb_indices(text, "vmatmul"),
            "mrb_base_occurrences": _mrb_occurrences(text, "vmatmul"),
            "mrb_pop_indices": mrb_pop_indices,
            "mrb_pop_occurrences": mrb_pop_occurrences,
            "mrb_pop_stats": mrb_pop_stats,
            "mrb_lifetime": _mrb_lifetime(text),
            "mrb_address_span": mrb_address_span,
        },
        "vector_path": {
            "vld": _count(r"=\s+vld\b", text),
            "vadd_f32": _count(r"=\s+vadd\.f32\b", text),
            "vpack_bf16": _count(r"=\s+vpack\.c\.bf16\b", text),
            "vst": _count(r"=\s+vst\s", text),
            "vst_masked": _count(r"=\s+vst\.msk\b", text),
            "mask_ops": _count(
                r"=\s+(?:vcmask|vsmask|vmand|vmor|vsel|vcombine\.)", text
            ),
        },
        "registers": {
            "vector": vector_registers,
            "vector_count": len(vector_registers),
            "scalar": scalar_registers,
            "scalar_count": len(scalar_registers),
            "predicate": predicate_registers,
            "predicate_count": len(predicate_registers),
            "vector_mask": mask_registers,
            "vector_mask_count": len(mask_registers),
        },
        "memory": {
            "output_kind": _output_kind(text),
            "dma_vmem_to_hbm": _count(
                r"=\s+dma\.vmem_to_hbm\b", executable_text
            ),
            "dma_wait": _count(
                r"=\s+dma\.done\.wait\b", executable_text
            ),
        },
        "spill_fill": {
            "mentions": _count(r"\b(?:spill|fill)\b", text),
        },
        "fifo": {
            "explicit_mentions": _count(r"\bfifo\b", text),
        },
        "source": str(bundle_path),
        "size_bytes": bundle_path.stat().st_size,
    }
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_matrix(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "m_tiles_16",
        "k_panels_256",
        "n_panels_256",
        "parallel_axis",
        "mrb_result_groups",
        "bundle_count",
        "vmatprep_subr",
        "vmatprep_subr_mxu0",
        "vmatprep_subr_mxu1",
        "vmatpush1",
        "vmatpush3",
        "vmatprep_mubr",
        "vmatmul",
        "vmatmul_mxu0",
        "vmatmul_mxu1",
        "vpop",
        "vadd_f32",
        "vmatmul_masked",
        "mask_ops",
        "mrb_pop_count_mxu0",
        "mrb_pop_count_mxu1",
        "mrb_address_span_mxu0",
        "mrb_address_span_mxu1",
        "mrb_peak_live_mxu0",
        "mrb_peak_live_mxu1",
        "mrb_wrap_to_zero_mxu0",
        "mrb_wrap_to_zero_mxu1",
        "vector_register_count",
        "output_kind",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for case in cases:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "m_tiles_16": case["decomposition"]["m_tiles_16"],
                    "k_panels_256": case["decomposition"]["k_panels_256"],
                    "n_panels_256": case["decomposition"]["n_panels_256"],
                    "parallel_axis": case["decomposition"]["parallel_axis"],
                    "mrb_result_groups": case["decomposition"][
                        "mrb_result_groups"
                    ],
                    "bundle_count": case["schedule"]["bundle_count"],
                    "vmatprep_subr": case["mxu"]["vmatprep_subr"],
                    "vmatprep_subr_mxu0": case["mxu"][
                        "vmatprep_subr_by_mxu"
                    ].get("mxu0", 0),
                    "vmatprep_subr_mxu1": case["mxu"][
                        "vmatprep_subr_by_mxu"
                    ].get("mxu1", 0),
                    "vmatpush1": case["mxu"][
                        "vmatpush_variants"
                    ].get("1", 0),
                    "vmatpush3": case["mxu"][
                        "vmatpush_variants"
                    ].get("3", 0),
                    "vmatprep_mubr": case["mxu"]["vmatprep_mubr"],
                    "vmatmul": case["mxu"]["vmatmul"],
                    "vmatmul_mxu0": case["mxu"]["vmatmul_by_mxu"].get(
                        "mxu0", 0
                    ),
                    "vmatmul_mxu1": case["mxu"]["vmatmul_by_mxu"].get(
                        "mxu1", 0
                    ),
                    "vpop": case["mxu"]["vpop"],
                    "vadd_f32": case["vector_path"]["vadd_f32"],
                    "vmatmul_masked": case["mxu"]["vmatmul_masked"],
                    "mask_ops": case["vector_path"]["mask_ops"],
                    "mrb_pop_count_mxu0": case["mxu"][
                        "mrb_pop_stats"
                    ].get("mxu0", {}).get("count", 0),
                    "mrb_pop_count_mxu1": case["mxu"][
                        "mrb_pop_stats"
                    ].get("mxu1", {}).get("count", 0),
                    "mrb_address_span_mxu0": case["mxu"][
                        "mrb_address_span"
                    ].get("mxu0", 0),
                    "mrb_address_span_mxu1": case["mxu"][
                        "mrb_address_span"
                    ].get("mxu1", 0),
                    "mrb_peak_live_mxu0": case["mxu"][
                        "mrb_lifetime"
                    ].get("mxu0", {}).get("peak_live", 0),
                    "mrb_peak_live_mxu1": case["mxu"][
                        "mrb_lifetime"
                    ].get("mxu1", {}).get("peak_live", 0),
                    "mrb_wrap_to_zero_mxu0": case["mxu"][
                        "mrb_pop_stats"
                    ].get("mxu0", {}).get("wrap_to_zero", 0),
                    "mrb_wrap_to_zero_mxu1": case["mxu"][
                        "mrb_pop_stats"
                    ].get("mxu1", {}).get("wrap_to_zero", 0),
                    "vector_register_count": case["registers"][
                        "vector_count"
                    ],
                    "output_kind": case["memory"]["output_kind"],
                }
            )


def _case_row(case: dict[str, Any]) -> str:
    decomposition = case["decomposition"]
    mxu = case["mxu"]
    vector_path = case["vector_path"]
    mrb_stats = mxu["mrb_pop_stats"]
    mrb_span = mxu["mrb_address_span"]
    mrb_lifetime = mxu["mrb_lifetime"]
    pop_count = (
        f"{mrb_stats.get('mxu0', {}).get('count', 0)}/"
        f"{mrb_stats.get('mxu1', {}).get('count', 0)}"
    )
    address_span = (
        f"{mrb_span.get('mxu0', 0)}/{mrb_span.get('mxu1', 0)}"
    )
    peak_live = (
        f"{mrb_lifetime.get('mxu0', {}).get('peak_live', 0)}/"
        f"{mrb_lifetime.get('mxu1', {}).get('peak_live', 0)}"
    )
    wrap_to_zero = (
        f"{mrb_stats.get('mxu0', {}).get('wrap_to_zero', 0)}/"
        f"{mrb_stats.get('mxu1', {}).get('wrap_to_zero', 0)}"
    )
    prep_by_mxu = mxu["vmatprep_subr_by_mxu"]
    push_variants = mxu["vmatpush_variants"]
    matmul_by_mxu = mxu["vmatmul_by_mxu"]
    prep_count = (
        f"{prep_by_mxu.get('mxu0', 0)}/"
        f"{prep_by_mxu.get('mxu1', 0)}"
    )
    matmul_count = (
        f"{matmul_by_mxu.get('mxu0', 0)}/"
        f"{matmul_by_mxu.get('mxu1', 0)}"
    )
    return (
        f"| `{case['case_id']}` "
        f"| {decomposition['m_tiles_16']}×"
        f"{decomposition['k_panels_256']}×"
        f"{decomposition['n_panels_256']} "
        f"| {decomposition['parallel_axis']} "
        f"| {mxu['vmatprep_subr']} ({prep_count}) "
        f"| {push_variants.get('1', 0)} "
        f"| {push_variants.get('3', 0)} "
        f"| {mxu['vmatprep_mubr']} "
        f"| {mxu['vmatmul']} ({matmul_count}) "
        f"| {mxu['vpop']} ({pop_count}) "
        f"| {vector_path['vadd_f32']} "
        f"| {mxu['vmatmul_masked']}/{vector_path['mask_ops']} "
        f"| {address_span} "
        f"| {peak_live} "
        f"| {wrap_to_zero} "
        f"| {case['schedule']['bundle_count']} |\n"
    )


def _write_report(path: Path, cases: list[dict[str, Any]]) -> None:
    by_id = {case["case_id"]: case for case in cases}
    baseline = by_id["m256_k256_n256"]
    matched_case_count = sum(
        case["mxu"]["vmatmul_matches_model"]
        and case["mxu"]["vpop_matches_model"]
        for case in cases
    )
    model_ok = matched_case_count == len(cases)
    no_spill = all(case["spill_fill"]["mentions"] == 0 for case in cases)
    no_fifo = all(case["fifo"]["explicit_mentions"] == 0 for case in cases)
    all_vregs = sorted(
        {
            register
            for case in cases
            for register in case["registers"]["vector"]
        }
    )

    lines = [
        f"# Ironwood MXU 行为模型：{len(cases)} 个 shape 的 final bundle 结论\n",
        "\n",
        "## 一句话结论\n",
        "\n",
        f"这 {len(cases)} 个 case 的最终 JF bundle 给出一个稳定分解公式：\n",
        "\n",
        "`MXU vmatmul 数 = ceil(M/16) × ceil(K/256) × ceil(N/256)`。\n",
        "\n",
        "令 `Mt=ceil(M/16)`、`Kp=ceil(K/256)`、"
        "`Np=ceil(N/256)`，则本组 case 的结果组数为 "
        "`R=Np + [Np 为奇数且 Kp>1]`，"
        "`vpop 数=4×Mt×R`。\n",
        "\n",
        "每条 `vmatmul` 固定指向连续 4 个 MRB entry；多个 K panel "
        "可以累加到同一组 entry，最终再 `vpop.f32` 取回。两个 MXU "
        "按 N→K→M 的优先级递归配对：先给两个 MXU 各一个 N panel，"
        "剩余奇数 N panel 再按 K 配对，剩余奇数 K panel 最后按 M "
        "条带平分。\n",
        "\n",
        f"模型在 {matched_case_count}/{len(cases)} case 上成立"
        f"{'。' if model_ok else '，其余 case 存在例外。'}\n",
        "\n",
        "## 从输入到输出的真实指令路径\n",
        "\n",
        "1. A、B 以 `inlined_call_operand.vmem` 进入，`vld` 把 4 KiB "
        "向量装入物理向量寄存器。\n",
        "2. B 侧先走 `vmatprep.subr`，再由 `vmatpush1` 或 "
        "`vmatpush3` 送入 MXU 的 MSR/MSRA 路径；同一个 N/K panel "
        "在多个 M 条带之间复用。\n",
        "3. A 侧通过 `vmatprep.mubr` 准备 MUBR，再由 "
        "`vmatmul.mubr...gmra.mrb[i].mxu{0,1}` 发射。\n",
        "4. 结果落在每个 MXU 独立的 MRB 中；每次发射写/累加 "
        "`mrb[i:i+4]`，最后 `vpop.f32` 回到向量寄存器。\n",
        "5. 必要时用 `vadd.f32` 合并 K 分块或窄 N 的片段，随后 "
        "`vpack.c.bf16`、`vst`/`vst.msk` 写入输出 VMEM；多数 case "
        "最后再 DMA 到 HBM。\n",
        "\n",
        "## 双 MXU 的分工规则\n",
        "\n",
        "- 先成对取 N panel：一对 N panel 分给 mxu0/mxu1；每个 MXU "
        "在自己的 panel 内遍历全部 K、M，多个 K panel 直接在同一 "
        "MRB entry 上硬件累加。\n",
        "- 如果剩下一个 N panel，再成对取 K panel：两个 MXU 各算"
        "一个 K panel 的完整 M，最后用 `vadd.f32` 合并两边部分和。\n",
        "- 如果又剩下一个 K panel，则沿 M 的 16 行条带平分给两个 "
        "MXU，并累加进各自已有的部分和。\n",
        "- 如果 N、K 都只有一个 panel，则直接沿 M 条带平分；"
        "一对并行发射覆盖 32 行。\n",
        "\n",
        "## 各 shape 的直接统计\n",
        "\n",
        "| Case | M×K×N panel | 双 MXU 轴 | vmatprep.subr（mxu0/1） "
        "| vmatpush1 | vmatpush3 | vmatprep.mubr | vmatmul（mxu0/1） "
        "| vpop（mxu0/1） | vadd "
        "| masked matmul/mask ops | MRB 地址跨度 mxu0/1 "
        "| MRB 峰值 live mxu0/1 | 回绕到 0 mxu0/1 | bundles |\n",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    lines.extend(_case_row(case) for case in cases)
    large_case_ids = {
        "m768_k256_n256",
        "m256_k768_n256",
        "m256_k256_n768",
        "m768_k768_n768",
        "m512_k256_n512",
        "m513_k256_n512",
    }
    if large_case_ids.issubset(by_id):
        m_triple = by_id["m768_k256_n256"]
        k_triple = by_id["m256_k768_n256"]
        n_triple = by_id["m256_k256_n768"]
        all_triple = by_id["m768_k768_n768"]
        at_128 = by_id["m512_k256_n512"]
        past_128 = by_id["m513_k256_n512"]
        lines.extend(
            [
                "\n",
                "## 3×256 case 与 MRB 容量证据\n",
                "\n",
                f"- `M=768,K=N=256`：48 个 M 条带按 24/24 分给"
                f"两个 MXU；`vmatmul` 为 "
                f"{m_triple['mxu']['vmatmul_by_mxu']['mxu0']}/"
                f"{m_triple['mxu']['vmatmul_by_mxu']['mxu1']}，"
                f"MRB 地址跨度为 "
                f"{m_triple['mxu']['mrb_address_span']['mxu0']}/"
                f"{m_triple['mxu']['mrb_address_span']['mxu1']}。\n",
                f"- `M=N=256,K=768`：两个完整 K panel 先各给一个 "
                f"MXU，第 3 个 K panel 再沿 M 平分；所以两边都是 "
                f"{k_triple['mxu']['vmatmul_by_mxu']['mxu0']} 条 "
                f"`vmatmul`、{k_triple['mxu']['mrb_pop_stats']['mxu0']['count']} "
                f"条 `vpop`，并用 "
                f"{k_triple['vector_path']['vadd_f32']} 条 `vadd.f32` "
                "合并部分和。\n",
                f"- `M=K=256,N=768`：前两个 N panel 各给一个 MXU，"
                f"第 3 个 N panel 沿 M 平分；两边各 "
                f"{n_triple['mxu']['vmatmul_by_mxu']['mxu0']} 条 "
                f"`vmatmul`，不需要 `vadd.f32`。\n",
                f"- `M=K=N=768`：两边各 "
                f"{all_triple['mxu']['vmatmul_by_mxu']['mxu0']} 条 "
                f"`vmatmul`、"
                f"{all_triple['mxu']['mrb_pop_stats']['mxu0']['count']} 条 "
                f"`vpop`；奇数 N/K 的尾组产生 "
                f"{all_triple['vector_path']['vadd_f32']} 条 `vadd.f32`。\n",
                "\n",
                f"- `M=512,K=256,N=512` 已使用每边 0.."
                f"{at_128['mxu']['mrb_address_span']['mxu0'] - 1}；"
                f"`M=513` 继续使用到 "
                f"{past_128['mxu']['mrb_address_span']['mxu0'] - 1}，"
                "因此 **MRB 不是 128-entry 容量**。\n",
                f"- `768³` 在每个 MXU 上观察到 0.."
                f"{all_triple['mxu']['mrb_address_span']['mxu0'] - 1}，"
                f"随后回到 0（回绕 "
                f"{all_triple['mxu']['mrb_pop_stats']['mxu0']['wrap_to_zero']} "
                "次）。这直接支持“编译器暴露的每-MXU MRB 环形地址"
                "窗口为 256 entry”。\n",
                f"- 按 bundle 顺序重放 `vmatmul` 分配与 `vpop` 释放，"
                f"`768³` 的静态峰值 live 是每边 "
                f"{all_triple['mxu']['mrb_lifetime']['mxu0']['peak_live']}，"
                "不是 256。也就是说：**256 个地址都用到并发生回绕，"
                "但没有看到 256 个 entry 同时 live 的“占用打满”**。"
                "同 bundle 内并行发射会让这个静态值存在少量次序误差，"
                "但不影响“低于 256”的结论。\n",
            ]
        )
    lines.extend(
        [
            "\n",
            "其中 bundle 数是静态 bundle index 数，只是发射周期代理，"
            "不包含运行时 stall。\n",
            "\n",
            "## M、K、N 分别改变什么\n",
            "\n",
            "### M\n",
            "\n",
            "- M 的基本条带是 16 行；128/256/512 分别产生 "
            "8/16/32 条 `vmatmul`（K=N=256）。\n",
            "- M=255 仍按 16 个条带计算，尾部通过 mask 保证越界行"
            "不写出。\n",
            "- M=257 产生第 17 个条带；在 M 并行模式下它落到 mxu0，"
            "所以 MRB 地址跨度为 mxu0=36、mxu1=32 entry。\n",
            "- M=768 产生 48 个条带，两个 MXU 各处理 24 个，"
            "对应每边 96 个 MRB 结果 entry。\n",
            "- M 改变时 B 侧的 preload 数不变，证明 B 的 MXU 状态会"
            "跨 M 条带复用。\n",
            "\n",
            "### K\n",
            "\n",
            "- K 的 panel 是 256。K=128 显式构造零向量，并反复用它"
            "填充缺失的 MUBR 半块。\n",
            "- K=255 使用 `vmatprep.mubr.msk`，同时把 B 的尾部无效"
            "元素 `vsel` 为 0。\n",
            "- K=257/512 都分成两个 K panel；当 N=256 时两个 MXU "
            "各算一个 panel，64 个 FP32 片段由 64 条 `vadd.f32` "
            "相加。\n",
            "- K=768 时，前两个 K panel 各给一个 MXU，第 3 个 panel "
            "沿 M 条带平分并在硬件侧累加；最终仍只 pop 两组部分和，"
            "所以 `vpop=128` 而不是 192。\n",
            "\n",
            "### N\n",
            "\n",
            "- N=128 使用 `vmatpush3`，每次发射仍产生 4 个 MRB "
            "片段，但成对相加后只形成一个 128 列输出 tile。\n",
            "- N=255 仍是一个 panel，输出使用 `vst.msk` 屏蔽最后"
            "一列；N=256 不需要 mask。\n",
            "- N=257/512 都产生两个 N panel。N=257 的 mxu1 只处理"
            "尾部 1 列，使用 `vmatpush3`、片段相加和 masked store；"
            "N=512 则两个 MXU 各处理完整 256 列，无需相加。\n",
            "- N=768 时，前两个 N panel 各给一个 MXU，第 3 个 panel "
            "沿 M 条带平分；因此两边仍完全对称且不需要跨 MXU 相加。\n",
            "\n",
            "### M=K=N=257\n",
            "\n",
            "- 精确产生 `17 × 2 × 2 = 68` 条 `vmatmul`，对应 "
            "`68 × 4 = 272` 次 MRB fragment 更新；由于 K panel 在"
            "同一 MRB 地址上累加，最终只需 pop 136 次。\n",
            "- final bundle 中实际是 68 条 `vmatmul`、136 条 "
            "`vpop`；M/K/N 三个尾部都出现 mask。\n",
            "\n",
            "## 寄存器和 buffer\n",
            "\n",
            f"- final bundle 已经过 RA，能看到物理向量寄存器编号；"
            f"本组实验覆盖 `v{all_vregs[0]}..v{all_vregs[-1]}`，"
            f"共 {len(all_vregs)} 个编号。\n",
            f"- 基线 `(256,256,256)` 使用 "
            f"{baseline['registers']['vector_count']} 个向量寄存器编号，"
            f"MRB 在 mxu0/mxu1 上观察到的地址跨度各为 "
            f"{baseline['mxu']['mrb_address_span'].get('mxu0', 0)} entry，"
            f"峰值 live 各为 "
            f"{baseline['mxu']['mrb_lifetime'].get('mxu0', {}).get('peak_live', 0)}。\n",
            "- `subr`、`msra`、`mubr`、`gmra`、`mrb` 是 bundle 中"
            "可直接观察到的 MXU staging/accumulator/result 接口。\n",
            f"- {len(cases)} 个 final bundle 中"
            f"{'没有' if no_spill else '存在'} "
            "spill/fill；物理 vreg 压力没有触发 VMEM spill。\n",
            "\n",
            "## FIFO 能确认到什么\n",
            "\n",
            f"- final bundle 中{'没有' if no_fifo else '存在'}名为 "
            "`fifo` 的指令，也没有 FIFO 深度或 occupancy 字段。\n",
            "- 因此可以确认数据顺序是 "
            "`vld → subr/push → mubr/matmul → MRB → vpop`，但不能仅凭 "
            "LLO 声称隐藏 FIFO 的 entry 数、bank 映射或 backpressure。\n",
            "- XProf 可验证运行时 stall/utilization，但同样通常不能直接"
            "给出隐藏 FIFO 深度；当前静态行为模型不需要先跑 XProf。\n",
            "\n",
            "## 证据与边界\n",
            "\n",
            f"- 数据来自 `{path.parents[1].name}` 的 {len(cases)} 份 "
            "最终 bundle；本地每个 case 仅保存重命名后的 "
            "`compiler/llo/final_bundle.llo`。\n",
            "- dump flags：`--xla_jf_dump_to`、"
            "`--xla_jf_dump_llo_text=true`、"
            "`--xla_jf_emit_annotations=true`。\n",
            "- 同次运行还按文档启用了 "
            "`--xla_enable_custom_call_region_trace=true` 和 "
            "`--xla_xprof_register_llo_debug_info=true`；它们用于 trace "
            "映射，不是生成 final bundle 的必要选择条件。\n",
            "- 这是编译器对当前 Pallas 单程序 BF16×BF16→BF16、FP32 "
            "累加 kernel 的行为，不应直接外推到其他 dtype、layout 或"
            "显式软件 tiling kernel。\n",
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
    bundle_paths = sorted(
        artifact_dir.glob("cases/*/compiler/llo/final_bundle.llo")
    )
    if not bundle_paths:
        raise FileNotFoundError(f"no final bundles below {artifact_dir}")

    cases = [_analyze_case(bundle_path) for bundle_path in bundle_paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1,
        "experiment_id": artifact_dir.name,
        "case_count": len(cases),
        "model": {
            "vmatmul": "ceil(M/16)*ceil(K/256)*ceil(N/256)",
            "vpop": (
                "4*ceil(M/16)*result_groups; result_groups=ceil(N/256)"
                "+1 when ceil(N/256) is odd and ceil(K/256)>1"
            ),
            "all_cases_match": all(
                case["mxu"]["vmatmul_matches_model"]
                and case["mxu"]["vpop_matches_model"]
                for case in cases
            ),
        },
        "cases": cases,
    }
    _write_json(output_dir / "summary.json", summary)
    _write_matrix(output_dir / "shape_matrix.tsv", cases)
    _write_report(output_dir / "analysis.md", cases)


if __name__ == "__main__":
    main()
