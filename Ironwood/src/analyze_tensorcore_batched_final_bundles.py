#!/usr/bin/env python3
"""Analyze JF final bundles for rank-3 Pallas batched matmul cases."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CASE_RE = re.compile(
    r"^mb(?P<mb>\d+)_m(?P<m>\d+)_k(?P<k>\d+)_n(?P<n>\d+)$"
)
BUNDLE_RE = re.compile(r"^\s*(?P<index>0x[0-9a-f]+|\d+)\s+:", re.I)
INPUT_RE = re.compile(
    r"(?P<base>%s\d+_s\d+)\s*=\s*inlined_call_operand\.vmem "
    r"\[shape:\s*bf16\[[^]]+\],\s*index:\s*(?P<index>[01]),"
)
VLD_RE = re.compile(
    r"(?P<reg>%v\d+_v\d+)\s*=\s*vld(?:\.[^\s]+)?\s+"
    r"\[vmem:\[(?P<base>%s\d+_s\d+)"
    r"(?:\s+\+\s+\$0x(?P<offset>[0-9a-f]+))?\]",
    re.I,
)
VMATMUL_RE = re.compile(
    r"(?P<opcode>vmatmul\.mubr\.[^\s;}]+)\s+"
    r"(?P<operand>%v\d+_v\d+)",
    re.I,
)
SUBR_RE = re.compile(
    r"vmatprep\.subr(?:\.msk)?\.bf16\.mxu(?P<mxu>[01])\s+"
    r"(?P<operand>%v\d+_v\d+)",
    re.I,
)
MUBR_RE = re.compile(
    r"vmatprep\.mubr(?:\.msk)?\.bf16\.mxu(?P<mxu>[01])\s+"
    r"(?P<operand>%v\d+_v\d+)",
    re.I,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze batched Pallas final_bundle.llo files."
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _count(pattern: str, text: str) -> int:
    return len(re.findall(pattern, text, flags=re.I))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _head_from_vreg(
    vreg: str,
    expected_base: str,
    head_stride_units: int,
    definitions: dict[str, tuple[str, int]],
) -> tuple[int | None, int | None]:
    definition = definitions.get(vreg)
    if definition is None:
        return None, None
    base, offset = definition
    if base != expected_base or head_stride_units <= 0:
        return None, offset
    return offset // head_stride_units, offset


def _head_schedule_text(records: list[dict[str, Any]], mb: int) -> str:
    counts: dict[int, Counter[int]] = defaultdict(Counter)
    for record in records:
        head = record["head"]
        if head is not None:
            counts[head][record["mxu"]] += 1
    fields = []
    for head in range(mb):
        placement = counts.get(head, Counter())
        values = "/".join(
            f"mxu{mxu}×{count}" for mxu, count in sorted(placement.items())
        )
        fields.append(f"h{head}→{values or '?'}")
    return "; ".join(fields)


def _weight_schedule_text(records: list[dict[str, Any]], mb: int) -> str:
    placements: dict[int, set[int]] = defaultdict(set)
    for record in records:
        head = record["head"]
        if head is not None:
            placements[head].add(record["mxu"])
    return "; ".join(
        f"h{head}→"
        + "/".join(f"mxu{mxu}" for mxu in sorted(placements.get(head, set())))
        for head in range(mb)
    )


def _analyze_case(path: Path) -> dict[str, Any]:
    case_id = path.parents[2].name
    match = CASE_RE.fullmatch(case_id)
    if match is None:
        raise ValueError(f"unexpected case directory: {case_id}")
    mb, m, k, n = (
        int(match.group("mb")),
        int(match.group("m")),
        int(match.group("k")),
        int(match.group("n")),
    )
    text = path.read_text(encoding="utf-8", errors="replace")
    inputs = {
        int(input_match.group("index")): input_match.group("base")
        for input_match in INPUT_RE.finditer(text)
    }
    if set(inputs) != {0, 1}:
        raise ValueError(f"could not identify lhs/rhs VMEM bases in {path}")
    definitions = {
        load.group("reg"): (
            load.group("base"),
            int(load.group("offset") or "0", 16),
        )
        for load in VLD_RE.finditer(text)
    }
    # LLO VMEM addresses use 512-byte units in these combined-load operands.
    lhs_stride_units = m * k * 2 // 512
    rhs_stride_units = k * n * 2 // 512

    matmuls = []
    subrs = []
    mubrs = []
    bundle_op_counts: Counter[int] = Counter()
    bundle_indices = []
    for line in text.splitlines():
        bundle_match = BUNDLE_RE.match(line)
        if bundle_match is None:
            continue
        bundle = int(bundle_match.group("index"), 0)
        bundle_indices.append(bundle)
        for op_match in VMATMUL_RE.finditer(line):
            opcode = op_match.group("opcode")
            mxu_match = re.search(r"\.mxu([01])", opcode, flags=re.I)
            mrb_match = re.search(r"mrb\[(\d+)\]", opcode, flags=re.I)
            if mxu_match is None or mrb_match is None:
                continue
            mxu = int(mxu_match.group(1))
            head, offset = _head_from_vreg(
                op_match.group("operand"),
                inputs[0],
                lhs_stride_units,
                definitions,
            )
            matmuls.append(
                {
                    "bundle": bundle,
                    "mxu": mxu,
                    "mrb": int(mrb_match.group(1)),
                    "head": head,
                    "lhs_offset_units": offset,
                    "operand": op_match.group("operand"),
                    "initializes_gmra": ".vlgmr." in opcode.lower(),
                }
            )
            bundle_op_counts[bundle] += 1
        for op_match in SUBR_RE.finditer(line):
            head, offset = _head_from_vreg(
                op_match.group("operand"),
                inputs[1],
                rhs_stride_units,
                definitions,
            )
            subrs.append(
                {
                    "bundle": bundle,
                    "mxu": int(op_match.group("mxu")),
                    "head": head,
                    "rhs_offset_units": offset,
                    "operand": op_match.group("operand"),
                }
            )
        for op_match in MUBR_RE.finditer(line):
            head, offset = _head_from_vreg(
                op_match.group("operand"),
                inputs[0],
                lhs_stride_units,
                definitions,
            )
            mubrs.append(
                {
                    "bundle": bundle,
                    "mxu": int(op_match.group("mxu")),
                    "head": head,
                    "lhs_offset_units": offset,
                    "operand": op_match.group("operand"),
                }
            )

    matmul_by_mxu = Counter(record["mxu"] for record in matmuls)
    subr_by_mxu = Counter(record["mxu"] for record in subrs)
    dual_issue_bundles = sorted(
        bundle for bundle, count in bundle_op_counts.items() if count > 1
    )
    entry_text = "\n".join(
        line for line in text.splitlines() if BUNDLE_RE.match(line)
    )
    known_heads = {record["head"] for record in matmuls}
    mapping_complete = known_heads == set(range(mb))
    logical_rows_per_matmul = (
        mb * m / len(matmuls) if matmuls else None
    )
    expected_full_panel = mb * math.ceil(m / 16)

    return {
        "case_id": case_id,
        "shape": {"mb": mb, "m": m, "k": k, "n": n},
        "addressing": {
            "unit_bytes": 512,
            "lhs_head_stride_units": lhs_stride_units,
            "rhs_head_stride_units": rhs_stride_units,
        },
        "schedule": {
            "bundle_count": len(set(bundle_indices)),
            "first_bundle": min(bundle_indices) if bundle_indices else None,
            "last_bundle": max(bundle_indices) if bundle_indices else None,
            "loop_markers": _count(r"\b(?:LH|LB|LE)\b", entry_text),
            "vmatmul_dual_issue_bundles": dual_issue_bundles,
            "vmatmul_dual_issue_count": len(dual_issue_bundles),
        },
        "mxu": {
            "vmatprep_subr": len(subrs),
            "vmatprep_subr_by_mxu": {
                f"mxu{mxu}": subr_by_mxu.get(mxu, 0) for mxu in (0, 1)
            },
            "vmatpush1": _count(r"\bvmatpush1\.bf16", text),
            "vmatpush3": _count(r"\bvmatpush3\.bf16", text),
            "vmatprep_mubr": len(mubrs),
            "vmatmul": len(matmuls),
            "vmatmul_by_mxu": {
                f"mxu{mxu}": matmul_by_mxu.get(mxu, 0) for mxu in (0, 1)
            },
            "vpop": _count(r"\bvpop\.f32", text),
            "vadd_f32": _count(r"=\s+vadd\.f32", text),
            "vld": _count(r"=\s+vld(?:\.|\s)", text),
            "full_panel_expected_vmatmul": expected_full_panel,
            "matches_full_panel_model": len(matmuls) == expected_full_panel,
            "logical_m_rows_per_vmatmul": logical_rows_per_matmul,
        },
        "batch_mapping": {
            "complete": mapping_complete,
            "matmul_text": _head_schedule_text(matmuls, mb),
            "weight_text": _weight_schedule_text(subrs, mb),
            "matmuls": matmuls,
            "weight_prepares": subrs,
            "mubr_prepares": mubrs,
        },
        "source": str(path),
        "size_bytes": path.stat().st_size,
    }


def _write_tsv(path: Path, cases: list[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "mb",
        "m",
        "k",
        "n",
        "vmatprep_subr",
        "vmatprep_subr_mxu0",
        "vmatprep_subr_mxu1",
        "vmatmul",
        "vmatmul_mxu0",
        "vmatmul_mxu1",
        "logical_m_rows_per_vmatmul",
        "vpop",
        "vadd_f32",
        "vld",
        "dual_issue_bundles",
        "bundle_count",
        "matmul_mapping",
        "weight_mapping",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for case in cases:
            shape = case["shape"]
            mxu = case["mxu"]
            schedule = case["schedule"]
            mapping = case["batch_mapping"]
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    **shape,
                    "vmatprep_subr": mxu["vmatprep_subr"],
                    "vmatprep_subr_mxu0": mxu[
                        "vmatprep_subr_by_mxu"
                    ]["mxu0"],
                    "vmatprep_subr_mxu1": mxu[
                        "vmatprep_subr_by_mxu"
                    ]["mxu1"],
                    "vmatmul": mxu["vmatmul"],
                    "vmatmul_mxu0": mxu["vmatmul_by_mxu"]["mxu0"],
                    "vmatmul_mxu1": mxu["vmatmul_by_mxu"]["mxu1"],
                    "logical_m_rows_per_vmatmul": mxu[
                        "logical_m_rows_per_vmatmul"
                    ],
                    "vpop": mxu["vpop"],
                    "vadd_f32": mxu["vadd_f32"],
                    "vld": mxu["vld"],
                    "dual_issue_bundles": ",".join(
                        hex(bundle)
                        for bundle in schedule["vmatmul_dual_issue_bundles"]
                    ),
                    "bundle_count": schedule["bundle_count"],
                    "matmul_mapping": mapping["matmul_text"],
                    "weight_mapping": mapping["weight_text"],
                }
            )


def _table_row(case: dict[str, Any]) -> str:
    mxu = case["mxu"]
    schedule = case["schedule"]
    mapping = case["batch_mapping"]
    prep = mxu["vmatprep_subr_by_mxu"]
    matmul = mxu["vmatmul_by_mxu"]
    dual = ",".join(
        hex(bundle) for bundle in schedule["vmatmul_dual_issue_bundles"]
    )
    return (
        f"| `{case['case_id']}` | {mxu['vmatprep_subr']} "
        f"({prep['mxu0']}/{prep['mxu1']}) | {mxu['vmatmul']} "
        f"({matmul['mxu0']}/{matmul['mxu1']}) | "
        f"{mxu['logical_m_rows_per_vmatmul']:g} | "
        f"{mapping['matmul_text']} | {mapping['weight_text']} | "
        f"{dual or '-'} | {mxu['vpop']} | {schedule['bundle_count']} |\n"
    )


def _write_report(path: Path, cases: list[dict[str, Any]]) -> None:
    by_id = {case["case_id"]: case for case in cases}
    mb2_m16 = by_id["mb2_m16_k256_n256"]
    mb1_m32 = by_id["mb1_m32_k256_n256"]
    mb2_m32 = by_id["mb2_m32_k256_n256"]
    mb4_m32 = by_id["mb4_m32_k256_n256"]
    narrow = by_id["mb4_m64_k128_n128"]
    all_unrolled = all(case["schedule"]["loop_markers"] == 0 for case in cases)
    all_mapped = all(case["batch_mapping"]["complete"] for case in cases)

    lines = [
        "# 带 MB 维的 Pallas matmul：final bundle 调度结论\n",
        "\n",
        "## 结论\n",
        "\n",
        "`jnp.matmul([MB,M,K], [MB,K,N])` 没有变成一个带 batch "
        "能力的 MXU 指令。MB 在编译期被展开成独立二维 matmul，"
        "每个 head 有自己的 B 权重 staging 和 MRB 结果。\n",
        "\n",
        "当 `MB>=2` 时，编译器优先用 batch/head 填满两个 MXU："
        "`h0→mxu0, h1→mxu1, h2→mxu0, h3→mxu1, ...`。同一 head "
        "的所有 M tile 留在同一个 MXU；下一对 head 形成下一波。"
        "奇数 MB 的尾 head 落到 mxu0。\n",
        "\n",
        "当 `MB=1` 时没有 head 级并行可用，编译器才退回沿 M 切分："
        "M 只有一个 tile 时只用 mxu0；M 有多个 tile 时把同一份 B "
        "放入两个 MXU，并让 mxu0/mxu1 合作处理同一个 head。\n",
        "\n",
        f"本组 {len(cases)} 个 bundle "
        f"{'都没有' if all_unrolled else '存在'}运行时 batch loop，"
        "head→MXU 映射"
        f"{'都可由 A/B 的 VMEM offset 还原' if all_mapped else '有缺失'}。\n",
        "\n",
        "## 关键证据\n",
        "\n",
        "- `MB=2,M=16,K=N=256`：B 的 head0 地址范围从 offset "
        "`0x0` 开始，只进入 mxu0；head1 从 `0x100` 开始，只进入 "
        "mxu1。A 的两个 head 分别从 `0x0`、`0x10` 取数，对应 "
        f"`{mb2_m16['batch_mapping']['matmul_text']}`。\n",
        "- `MB=2,M=32,K=N=256`：head0 的 A offset `0x0/0x10` "
        "全部由 mxu0 计算；head1 的 `0x20/0x30` 全部由 mxu1 "
        f"计算，即 `{mb2_m32['batch_mapping']['matmul_text']}`。"
        "这证明 batch 分工优先于在单个 head 内沿 M 分工。\n",
        "- `MB=4,M=32`：前一波 h0/h1 使用 MRB base 0、4，"
        "后一波 h2/h3 使用 base 8、12；映射为 "
        f"`{mb4_m32['batch_mapping']['matmul_text']}`。\n",
        "- `MB=1,M=32`：两条 vmatmul 同在 bundle "
        f"`{hex(mb1_m32['schedule']['vmatmul_dual_issue_bundles'][0])}`，"
        "分别发给 mxu0/mxu1；而不同 head 的 `MB=2,M=16` 两条 "
        "vmatmul 位于 bundle "
        f"`{hex(mb2_m16['batch_mapping']['matmuls'][0]['bundle'])}` "
        f"和 `{hex(mb2_m16['batch_mapping']['matmuls'][1]['bundle'])}`。"
        "因此不同 head 的发射有一个周期的错位，不是同 bundle "
        "双发；"
        "后续阵列执行仍可流水重叠。\n",
        "- `K=N=128,M=64` 时每个 head 只有 2 条 vmatmul，"
        "即每条平均覆盖 "
        f"{narrow['mxu']['logical_m_rows_per_vmatmul']:g} 行 M。"
        "这是 K、N 同时为半 panel 后的额外 packing；不能继续使用"
        "固定 `ceil(M/16)` 公式。\n",
        "\n",
        "## 指令统计\n",
        "\n",
        "`mxu0/mxu1` 为括号中的左右两项；“每条覆盖 M 行”是"
        "`MB×M/vmatmul数`。\n",
        "\n",
        "| Case | vmatprep.subr (0/1) | vmatmul (0/1) | 每条覆盖 M 行 "
        "| head→MXU | B staging | 同 bundle 双发 | vpop | bundles |\n",
        "|---|---:|---:|---:|---|---|---:|---:|---:|\n",
    ]
    lines.extend(_table_row(case) for case in cases)
    lines.extend(
        [
            "\n",
            "## 对原始三次 batched matmul 的含义\n",
            "\n",
            "源码里的三次 rank-3 matmul 仍分别拆成 MB 个二维任务："
            "`[BT,K]×[K,V]`、`[K,BT]×[BT,V]`、"
            "`[K,BT]×[BT,V]`。对于常见的偶数 MB，两个 MXU "
            "优先各拿一个 head，按 head 对推进；不会让一条 "
            "`vmatmul` 同时跨多个 head 做乘加。\n",
            "\n",
            "这里的实验只隔离了一次 batched matmul。完整 kernel 还会受"
            "三次 matmul 之间的数据依赖、scratch VMEM 占用和 VPU 加减法"
            "影响，但 batch/head 到两个 MXU 的基本分配证据已经"
            "明确。\n",
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
    paths = sorted(
        artifact_dir.glob("cases/*/compiler/llo/final_bundle.llo")
    )
    if not paths:
        raise FileNotFoundError(f"no final bundles below {artifact_dir}")
    cases = [_analyze_case(path) for path in paths]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        output_dir / "summary.json",
        {
            "schema_version": 1,
            "experiment_id": artifact_dir.name,
            "case_count": len(cases),
            "cases": cases,
        },
    )
    _write_tsv(output_dir / "schedule.tsv", cases)
    _write_report(output_dir / "analysis.md", cases)


if __name__ == "__main__":
    main()
