#!/usr/bin/env python3
"""Normalize non-MXU final-bundle evidence into primitive-to-LLO tables."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


INSTRUCTION_RE = re.compile(r"(?:%[A-Za-z0-9_]+\s*=\s*)?([a-z][a-z0-9_]*(?:\.[a-z0-9_]+)*)")
NON_INSTRUCTIONS = {
    "entry", "target", "region", "hlo", "shape", "index", "kind",
    "true", "false", "resolvable", "thread", "vmem", "smem", "hbm",
}


def _strip_comments(text: str) -> str:
    result = []
    depth = 0
    i = 0
    while i < len(text):
        if text.startswith("/*", i):
            depth += 1
            i += 2
        elif depth and text.startswith("*/", i):
            depth -= 1
            i += 2
        else:
            result.append("\n" if depth and text[i] == "\n" else (text[i] if not depth else " "))
            i += 1
    return "".join(result)


def _instructions(path: Path) -> Counter[str]:
    text = _strip_comments(path.read_text(encoding="utf-8", errors="replace"))
    counts: Counter[str] = Counter()
    for line in text.splitlines():
        if ":" not in line or "{" not in line or "}" not in line:
            continue
        body = line.split("{", 1)[1].rsplit("}", 1)[0]
        for instruction in body.split(";;"):
            match = INSTRUCTION_RE.search(instruction.strip())
            if match and match.group(1) not in NON_INSTRUCTIONS:
                counts[match.group(1)] += 1
    return counts


def _family(opcode: str) -> str:
    if opcode.startswith("vmat") or opcode.startswith("vlatch"):
        return "mxu"
    if opcode.startswith(("vpop.eup", "vtanh", "vpow2", "vrecip", "vlog", "vrsqrt", "vsig", "vsin", "vcos", "verf")):
        return "eup"
    if ".xlane" in opcode or "reduce" in opcode or opcode.startswith("vpop.xlane"):
        return "xlu_reduce"
    if opcode.startswith(("vperm", "vrot", "vxpose", "vtranspose", "vrotate", "vbroadcast", "vpop.permute", "vpop.trf")):
        return "xlu_permute"
    if opcode.startswith(("vcvt", "vpack", "vunpack")):
        return "convert_pack"
    if opcode.startswith(("vcmp", "vsel", "vcmask", "vmand", "vmmov", "vmneg", "vmor", "vmxor", "vmask", "vcreate_mask")):
        return "vector_mask"
    if opcode.startswith(("scmp", "pneg", "pnand", "por", "pmov")):
        return "predicate"
    if opcode.startswith("v") and not opcode.startswith(("vld", "vst", "vsync", "vtrace", "vset")):
        return "vpu"
    if opcode.startswith("s") and not opcode.startswith(("sld", "sst", "sbr", "shalt", "sfence", "scalar_")):
        return "spu"
    if opcode.startswith(("vld", "vst", "sld", "sst", "scalar_lea", "inlined_call_operand")):
        return "memory"
    if opcode.startswith("dma"):
        return "dma"
    return "control_other"


VECTOR_COMPUTE_FAMILIES = {
    "vpu", "eup", "xlu_reduce", "xlu_permute", "convert_pack", "vector_mask",
}
COMPUTE_FAMILIES = VECTOR_COMPUTE_FAMILIES | {"spu", "predicate", "mxu"}


def _source_compute_counts(
    counts: Counter[str], mode: str
) -> dict[str, int]:
    """Remove Pallas call scaffolding from source-primitive attribution.

    Vector cases should not attribute scalar bounds-check and DMA-loop setup to
    the JAX primitive under test. Scalar-grid probes intentionally exercise the
    SPU, so their scalar/predicate instructions remain compound correlations.
    """
    families = COMPUTE_FAMILIES if mode == "scalar_grid" else VECTOR_COMPUTE_FAMILIES
    return {
        opcode: count
        for opcode, count in counts.items()
        if _family(opcode) in families
    }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def analyze(artifact_root: Path) -> dict[str, Any]:
    cases = []
    global_counts: Counter[str] = Counter()
    mapping_rows = []
    for status_path in sorted((artifact_root / "cases").glob("*/status.json")):
        status = _load_json(status_path)
        case_id = status["case_id"]
        bundle = status_path.parent / "compiler" / "llo" / "final_bundle.llo"
        counts = _instructions(bundle) if bundle.is_file() else Counter()
        global_counts.update(counts)
        compute_counts = {
            opcode: count
            for opcode, count in counts.items()
            if _family(opcode) in COMPUTE_FAMILIES
        }
        source_compute_counts = _source_compute_counts(
            counts, status["case"].get("mode", "vector")
        )
        families = Counter()
        for opcode, count in counts.items():
            families[_family(opcode)] += count
        primitives = status["case"]["primitives"]
        for primitive in primitives:
            for opcode, count in sorted(source_compute_counts.items()):
                mapping_rows.append(
                    {
                        "case_id": case_id,
                        "source_primitive": primitive,
                        "llo_opcode": opcode,
                        "family": _family(opcode),
                        "count": count,
                        "attribution": "case-correlated; not necessarily 1:1" if len(primitives) > 1 else "isolated case",
                    }
                )
        cases.append(
            {
                "case_id": case_id,
                "status": status["status"],
                "primitives": primitives,
                "final_bundle": str(bundle.relative_to(artifact_root)) if bundle.is_file() else None,
                "bundle_sha256": status.get("final_bundle", {}).get("sha256") if status.get("final_bundle") else None,
                "instruction_count": sum(counts.values()),
                "unique_opcodes": len(counts),
                "family_counts": dict(sorted(families.items())),
                "compute_opcodes": dict(sorted(compute_counts.items())),
                "source_compute_opcodes": dict(sorted(source_compute_counts.items())),
            }
        )
    mxu = {opcode: count for opcode, count in global_counts.items() if _family(opcode) == "mxu"}
    compute = {
        opcode: count
        for opcode, count in global_counts.items()
        if _family(opcode) in COMPUTE_FAMILIES - {"mxu"}
    }
    return {
        "schema_version": 1,
        "artifact_root": str(artifact_root),
        "case_count": len(cases),
        "cases_with_final_bundle": sum(case["final_bundle"] is not None for case in cases),
        "unique_all_opcodes": len(global_counts),
        "unique_compute_opcodes": len(compute),
        "mxu_violation": mxu,
        "compute_opcode_counts": dict(sorted(compute.items())),
        "all_opcode_counts": dict(sorted(global_counts.items())),
        "cases": cases,
        "mapping_rows": mapping_rows,
    }


def _report(summary: dict[str, Any]) -> str:
    lines = [
        "# Non-MXU Pallas LLO Final-Bundle Report\n",
        f"- Cases: {summary['case_count']}\n",
        f"- Cases with named final bundle: {summary['cases_with_final_bundle']}\n",
        f"- Unique compute mnemonics: {summary['unique_compute_opcodes']}\n",
        f"- MXU mnemonics (must be empty): `{summary['mxu_violation']}`\n",
        "\n## Opcode counts\n\n",
        "| LLO mnemonic | Family | Count |\n",
        "|---|---|---:|\n",
    ]
    for opcode, count in summary["compute_opcode_counts"].items():
        lines.append(f"| `{opcode}` | {_family(opcode)} | {count} |\n")
    lines.extend(["\n## Primitive-correlated observations\n\n", "| Case | JAX/Pallas primitives | Observed compute LLO | Status |\n", "|---|---|---|---|\n"])
    for case in summary["cases"]:
        primitives = ", ".join(f"`{p}`" for p in case["primitives"])
        opcodes = ", ".join(f"`{op}`×{count}" for op, count in case["source_compute_opcodes"].items()) or "—"
        lines.append(f"| `{case['case_id']}` | {primitives} | {opcodes} | {case['status']} |\n")
    lines.extend([
        "\n## Attribution rule\n\n",
        "Vector cases exclude scalar bounds-check, address, and DMA-loop scaffolding from source attribution. A one-primitive case supports an isolated source→LLO observation. A compound case only establishes that the listed LLO mnemonics occur in that source primitive set; it does not prove a 1:1 mapping. Scalar-grid probes remain compound correlations because loop control and the tested SPU operations share the scalar slot. Final scheduled bundles are authoritative for emitted instructions, while unsupported/optimized-away enum members remain unobserved.\n",
    ])
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    summary = analyze(args.artifact_dir.resolve())
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps({k: v for k, v in summary.items() if k != "mapping_rows"}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    opcode_rows = [
        {"llo_opcode": opcode, "family": _family(opcode), "count": count}
        for opcode, count in summary["all_opcode_counts"].items()
    ]
    _write_csv(output / "opcode_counts.csv", ["llo_opcode", "family", "count"], opcode_rows)
    _write_csv(output / "primitive_to_llo.csv", ["case_id", "source_primitive", "llo_opcode", "family", "count", "attribution"], summary["mapping_rows"])
    print(json.dumps({"case_count": summary["case_count"], "unique_compute_opcodes": summary["unique_compute_opcodes"], "mxu_violation": summary["mxu_violation"]}, sort_keys=True))


if __name__ == "__main__":
    main()
