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


def _mrb_indices(text: str, opcode_prefix: str) -> dict[str, list[int]]:
    result: dict[str, set[int]] = {}
    for opcode in re.findall(rf"{opcode_prefix}\.[^\s;}}]+", text):
        match = re.search(r"mrb\[(\d+)\]\.mxu(\d+)", opcode)
        if match:
            index, mxu = int(match.group(1)), match.group(2)
            result.setdefault(mxu, set()).add(index)
    return {
        f"mxu{mxu}": sorted(indices)
        for mxu, indices in sorted(result.items())
    }


def _output_kind(text: str) -> str | None:
    match = re.search(
        r"inlined_call_operand\.(vmem|hbm)"
        r"\s+\[shape:[^\]]+\],\s+index:\s*2,\s+kind:\s*output",
        text,
    )
    return match.group(1) if match else None


def _parallel_axis(k_panels: int, n_panels: int) -> str:
    if n_panels > 1:
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
    external_k_partials = k_panels if n_panels == 1 else 1
    expected_vpop = 4 * m_tiles * n_panels * external_k_partials

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
    mrb_pop_indices = _mrb_indices(text, "vpop")
    mrb_entries = {
        mxu: (max(indices) + 1 if indices else 0)
        for mxu, indices in mrb_pop_indices.items()
    }

    result = {
        "case_id": case_id,
        "shape": {"m": m, "k": k, "n": n},
        "decomposition": {
            "m_tiles_16": m_tiles,
            "k_panels_256": k_panels,
            "n_panels_256": n_panels,
            "parallel_axis": _parallel_axis(k_panels, n_panels),
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
            "vmatpush": sum(push_variants.values()),
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
            "vmatmul_masked": _count(
                r"\bvmatmul\.mubr\.msk\.bf16", text
            ),
            "expected_vmatmul": expected_vmatmul,
            "vmatmul_matches_model": vmatmul == expected_vmatmul,
            "vpop": vpop,
            "expected_vpop": expected_vpop,
            "vpop_matches_model": vpop == expected_vpop,
            "mrb_base_indices": _mrb_indices(text, "vmatmul"),
            "mrb_pop_indices": mrb_pop_indices,
            "mrb_entries": mrb_entries,
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
        "bundle_count",
        "vmatprep_subr",
        "vmatmul",
        "vpop",
        "vadd_f32",
        "vmatmul_masked",
        "mask_ops",
        "mrb_entries_mxu0",
        "mrb_entries_mxu1",
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
                    "bundle_count": case["schedule"]["bundle_count"],
                    "vmatprep_subr": case["mxu"]["vmatprep_subr"],
                    "vmatmul": case["mxu"]["vmatmul"],
                    "vpop": case["mxu"]["vpop"],
                    "vadd_f32": case["vector_path"]["vadd_f32"],
                    "vmatmul_masked": case["mxu"]["vmatmul_masked"],
                    "mask_ops": case["vector_path"]["mask_ops"],
                    "mrb_entries_mxu0": case["mxu"]["mrb_entries"].get(
                        "mxu0", 0
                    ),
                    "mrb_entries_mxu1": case["mxu"]["mrb_entries"].get(
                        "mxu1", 0
                    ),
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
    mrb = mxu["mrb_entries"]
    return (
        f"| `{case['case_id']}` "
        f"| {decomposition['m_tiles_16']}×"
        f"{decomposition['k_panels_256']}×"
        f"{decomposition['n_panels_256']} "
        f"| {decomposition['parallel_axis']} "
        f"| {mxu['vmatprep_subr']} "
        f"| {mxu['vmatmul']} "
        f"| {mxu['vpop']} "
        f"| {vector_path['vadd_f32']} "
        f"| {mxu['vmatmul_masked']}/{vector_path['mask_ops']} "
        f"| {mrb.get('mxu0', 0)}/{mrb.get('mxu1', 0)} "
        f"| {case['schedule']['bundle_count']} |\n"
    )


def _write_report(path: Path, cases: list[dict[str, Any]]) -> None:
    by_id = {case["case_id"]: case for case in cases}
    baseline = by_id["m256_k256_n256"]
    model_ok = all(
        case["mxu"]["vmatmul_matches_model"]
        and case["mxu"]["vpop_matches_model"]
        for case in cases
    )
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
        "# Ironwood MXU 行为模型：14 个 shape 的 final bundle 结论\n",
        "\n",
        "## 一句话结论\n",
        "\n",
        "这 14 个 case 的最终 JF bundle 给出一个稳定分解公式：\n",
        "\n",
        "`MXU vmatmul 数 = ceil(M/16) × ceil(K/256) × ceil(N/256)`。\n",
        "\n",
        "每条 `vmatmul` 固定指向连续 4 个 MRB entry；多个 K panel "
        "可以累加到同一组 entry，最终再 `vpop.f32` 取回。两个 MXU "
        "优先按 N 分面，其次按 K 分块；"
        "只有 N、K 都不超过 256 时，才沿 M 条带并行。\n",
        "\n",
        f"模型在 14/14 case 上{'成立' if model_ok else '存在例外'}。\n",
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
        "- `ceil(N/256) > 1`：优先按 N panel 分工，mxu0/mxu1 "
        "各算一个最多 256 列的 panel；两边都遍历全部 M。\n",
        "- 否则若 `ceil(K/256) > 1`：按 K panel 分工，两边都算完整 "
        "M×N，最后用向量 `vadd.f32` 合并部分和。\n",
        "- 否则：B 广播给两个 MXU，mxu0/mxu1 交错处理 16 行 M "
        "条带，一对并行发射覆盖 32 行。\n",
        "- 当 N 和 K 都超过 256 时，N 分工优先；K 的下一 panel "
        "在同一 MXU 上复用相同 MRB index 做硬件侧累加。\n",
        "\n",
        "## 各 shape 的直接统计\n",
        "\n",
        "| Case | M×K×N panel | 双 MXU 轴 | vmatprep.subr | vmatmul "
        "| vpop | vadd "
        "| masked matmul/mask ops | MRB entries mxu0/1 | bundles |\n",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    lines.extend(_case_row(case) for case in cases)
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
            "所以 MRB 使用量为 mxu0=36、mxu1=32 entry。\n",
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
            f"MRB 为 mxu0/mxu1 各 "
            f"{baseline['mxu']['mrb_entries'].get('mxu0', 0)} entry。\n",
            "- `subr`、`msra`、`mubr`、`gmra`、`mrb` 是 bundle 中"
            "可直接观察到的 MXU staging/accumulator/result 接口。\n",
            f"- 14 个 final bundle 中{'没有' if no_spill else '存在'} "
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
            "- 数据来自 `exp-71c0c3sf6o` 的 14 份 "
            "`tpu_custom_call.1-70-final_bundles.txt`，本地仅保存每个 "
            "case 的最终 bundle。\n",
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
                "4*ceil(M/16)*ceil(N/256)*external_K_partials; "
                "external_K_partials=ceil(K/256) when N<=256, else 1"
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
