#!/usr/bin/env python3
"""Residual rule refinement workflow for Transformer GEMM scheduling.

This script focuses on residual error mining for the deterministic rule-based
schedule by comparing it against historical MetaSchedule best traces.
"""

import argparse
import ast
import json
import importlib
import math
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tvm

from research.workloads.bert.bert_shapes import (
    M_LIST,
    mlp_compressed_shape,
    mlp_expanded_shape,
    qkv_shape,
)
from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.common.rule_based_schedule import (
    apply_rule_based_schedule,
    describe_tile_sizes,
)

try:
    from research.workloads.bert.metaschedule.metaschedule_8020_tuner import (
        _summarize_trace_text as summarize_trace_text,
    )
except Exception:
    summarize_trace_text = None


KERNELS = {
    "qkv": qkv_shape,
    "mlp_expand": mlp_expanded_shape,
    "mlp_reduce": mlp_compressed_shape,
}

DEFAULT_RESULTS_FILE = "research/results/bert_matmul_results.json"
DEFAULT_BEST_SCHEDULES_FILE = "research/results/metaschedule/best_schedules.json"
DEFAULT_COMPARE_FILES = [
    "research/results/metaschedule/comparison_results.json",
    "research/archive/metaschedule_8020/results/metaschedule/comparison_results.json",
]
DEFAULT_CASES_FILE = "research/results/residual_rule_cases.json"
DEFAULT_SUGGESTIONS_FILE = "research/results/residual_rule_suggestions.json"
DEFAULT_PATCH_FILE = "research/results/rule_patch_candidate.diff"
DEFAULT_SWEEP_FILE = "research/results/residual_rule_sweep.json"
DEFAULT_REFINED_SCHEDULER = "research/workloads/common/rule_based_residual_refined.py"
REFINED_VARIANT_NAME = "rule_based_residual_refined"

TARGET = "llvm"
DEV = tvm.cpu(0)

ANSI_RED = "\033[31m"
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"


def _format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    # Support multiline cell contents (\n separated). Compute column widths
    # from the longest line in each column and render rows that may span
    # multiple visual lines by emitting sub-rows per logical row.
    rows_str = [[str(v) for v in row] for row in rows]
    # Split cells into lines
    split_rows: List[List[List[str]]] = []
    for row in rows_str:
        split_rows.append([cell.split("\n") for cell in row])

    num_cols = len(headers)
    widths = [len(str(h)) for h in headers]
    for row in split_rows:
        for i, cell_lines in enumerate(row):
            for line in cell_lines:
                widths[i] = max(widths[i], len(line))

    def _line(ch: str = "-") -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def _format_row_lines(row_lines: List[List[str]]) -> List[str]:
        # row_lines is a list of columns, each a list of lines
        max_lines = max((len(col) for col in row_lines), default=0)
        out_lines: List[str] = []
        for li in range(max_lines):
            parts: List[str] = []
            for ci in range(num_cols):
                col_lines = row_lines[ci]
                text = col_lines[li] if li < len(col_lines) else ""
                parts.append(text.ljust(widths[ci]))
            out_lines.append("| " + " | ".join(parts) + " |")
        return out_lines

    out: List[str] = []
    out.append(_line("="))
    out.append("| " + " | ".join(headers[i].ljust(widths[i]) for i in range(num_cols)) + " |")
    out.append(_line("="))

    for row_lines in split_rows:
        out.extend(_format_row_lines(row_lines))

    out.append(_line("-"))
    return "\n".join(out)


def _print_section(title: str) -> None:
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _colorize(text: str, color: str) -> str:
    if not _supports_color():
        return text
    return f"{color}{text}{ANSI_RESET}"


def _format_shapes_column(shapes: Sequence[Sequence[int]], max_items: int = 4) -> str:
    formatted = [f"({int(s[0])},{int(s[1])},{int(s[2])})" for s in shapes if len(s) >= 3]
    if not formatted:
        return ""
    if len(formatted) <= max_items:
        return "\n".join(formatted)
    shown = "\n".join(formatted[:max_items])
    remaining = len(formatted) - max_items
    return f"{shown}\n... (+{remaining})"


def _print_residual_report(cases: Sequence[Dict[str, Any]], threshold: float) -> None:
    _print_section(f"Residual Gap Report (threshold > {threshold:.4f})")
    if not cases:
        print("No residual cases found above threshold.")
        return

    sorted_cases = sorted(
        cases,
        key=lambda x: (
            str(x.get("kernel", "")),
            int(x.get("M", 0)),
            int(x.get("K", 0)),
            int(x.get("N", 0)),
        ),
    )
    rows = []
    for idx, case in enumerate(sorted_cases, 1):
        rows.append(
            [
                idx,
                case.get("kernel"),
                case.get("M"),
                case.get("K"),
                case.get("N"),
                f"{float(case.get('rule_latency_us', 0.0)):.3f}",
                f"{float(case.get('metaschedule_latency_us', 0.0)):.3f}",
                f"{float(case.get('residual_gap', 0.0)):.4f}",
            ]
        )
    print(
        _format_table(
            ["#", "kernel", "M", "K", "N", "rule_us", "metasched_us", "gap"],
            rows,
        )
    )

    gaps = [float(x.get("residual_gap", 1.0)) for x in cases]
    stats_rows = [
        ["count", str(len(cases))],
        ["avg_gap", f"{statistics.fmean(gaps):.4f}"],
        ["max_gap", f"{max(gaps):.4f}"],
        ["min_gap", f"{min(gaps):.4f}"],
    ]
    print(_format_table(["metric", "value"], stats_rows))


def _print_pattern_report(patterns: Sequence[Dict[str, Any]]) -> None:
    _print_section("Recurring Residual Patterns")
    if not patterns:
        print("No recurring decision-difference patterns found.")
        return
    rows = []
    for p in patterns[:20]:
        rows.append(
            [
                p.get("parameter"),
                f"{p.get('rule_value')} -> {p.get('metaschedule_value')}",
                p.get("occurrences"),
                f"{float(p.get('avg_residual_gap', 0.0)):.4f}",
                f"{float(p.get('max_residual_gap', 0.0)):.4f}",
                _format_shapes_column(p.get("shapes", [])),
                ",".join(str(k) for k in p.get("kernels", [])),
            ]
        )
    print(
        _format_table(
            ["parameter", "rule_to_meta", "count", "avg_gap", "max_gap", "(M,K,N)", "kernels"],
            rows,
        )
    )


def _print_suggestion_report(suggestions: Sequence[Dict[str, Any]]) -> None:
    _print_section("Heuristic Refinement Suggestions")
    if not suggestions:
        print("No suggestions produced.")
        return
    rows = []
    for idx, s in enumerate(suggestions[:20], 1):
        param = str(s.get("parameter", ""))
        curr = s.get("current_heuristic", {}).get(param)
        sugg = s.get("suggested_heuristic", {}).get(param)
        rows.append(
            [
                idx,
                param,
                str(s.get("condition", "")),
                str(curr),
                str(sugg),
                f"{float(s.get('expected_gain_pct', 0.0)):.2f}",
                f"{float(s.get('confidence', 0.0)):.3f}",
                int(s.get("sample_count", 0)),
            ]
        )
    print(
        _format_table(
            ["#", "param", "condition", "current", "suggested", "gain_%", "confidence", "samples"],
            rows,
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Residual rule refinement pipeline for deterministic Transformer GEMM scheduling"
        )
    )
    parser.add_argument("--kernel", choices=sorted(KERNELS.keys()), help="Run a single kernel")
    parser.add_argument("--all", action="store_true", help="Run all kernels")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.05,
        help="Residual gap threshold (rule_latency / metaschedule_latency)",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Skip fresh benchmarking and build residuals from existing JSON artifacts",
    )
    parser.add_argument(
        "--suggestion-only",
        action="store_true",
        help="Skip sweep/filter and generate suggestions from residual_rule_cases.json",
    )
    parser.add_argument(
        "--best-schedules",
        default=DEFAULT_BEST_SCHEDULES_FILE,
        help="Path to best_schedules.json",
    )
    parser.add_argument(
        "--results-file",
        default=DEFAULT_RESULTS_FILE,
        help="Path to benchmark results JSON",
    )
    parser.add_argument(
        "--comparison-json",
        action="append",
        default=None,
        help="Path to comparison JSON file (repeat flag for multiple files)",
    )
    parser.add_argument("--residual-cases", default=DEFAULT_CASES_FILE)
    parser.add_argument("--suggestions", default=DEFAULT_SUGGESTIONS_FILE)
    parser.add_argument("--sweep-log", default=DEFAULT_SWEEP_FILE)
    parser.add_argument(
        "--refined-scheduler",
        default=DEFAULT_REFINED_SCHEDULER,
        help="Path to iterative refined scheduler copy",
    )
    parser.add_argument(
        "--max-auto-rules",
        type=int,
        default=2,
        help="Maximum top suggestions converted into refined scheduler rules",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=15,
        help="Maximum iterative refinement rounds to run",
    )
    parser.add_argument(
        "--target-residuals",
        type=int,
        default=0,
        help="Stop once residual case count reaches this value",
    )
    parser.add_argument(
        "--force-iterations",
        action="store_true",
        help=(
            "Run all --max-iterations rounds even when early-stop conditions are met "
            "(for debugging/diagnostics)"
        ),
    )
    parser.add_argument(
        "--auto-patch",
        action="store_true",
        help="Generate patch candidate diff (does not modify source)",
    )
    parser.add_argument("--patch-file", default=DEFAULT_PATCH_FILE)
    parser.add_argument("--number", type=int, default=50, help="time_evaluator number")
    parser.add_argument("--repeat", type=int, default=3, help="time_evaluator repeat")
    parser.add_argument("--min-repeat-ms", type=int, default=50)

    args = parser.parse_args()
    if not args.kernel and not args.all:
        parser.error("Specify either --kernel <name> or --all")
    if args.kernel and args.all:
        parser.error("Use either --kernel or --all, not both")
    if args.threshold <= 0:
        parser.error("--threshold must be > 0")
    return args


def _load_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _refined_scheduler_template() -> str:
    return '''"""Iteratively refined rule-based CPU schedule for Transformer MatMul kernels.

Auto-generated by `research.analysis.residual_rule_refiner` when missing.
"""

import tvm


_VEC_WIDTH = 8
_UNROLL_LIMIT = 16
_AUTO_UNROLL_STEP = 64
_CACHE_WRITE_SCOPE = "global"
_J_PACK_MULT = 4


# BEGIN_RESIDUAL_REFINEMENTS
RESIDUAL_REFINEMENT_RULES = []
# END_RESIDUAL_REFINEMENTS


def _matches_condition(condition, M, K, N, kernel):
    if not isinstance(condition, dict):
        return False
    if "kernel" in condition and str(condition["kernel"]) != str(kernel):
        return False
    if "M_eq" in condition and int(M) != int(condition["M_eq"]):
        return False
    if "M_min" in condition and int(M) < int(condition["M_min"]):
        return False
    if "M_max" in condition and int(M) > int(condition["M_max"]):
        return False
    if "M_mod" in condition:
        mod = int(condition["M_mod"])
        if mod <= 0 or int(M) % mod != 0:
            return False
    if "K_eq" in condition and int(K) != int(condition["K_eq"]):
        return False
    if "N_eq" in condition and int(N) != int(condition["N_eq"]):
        return False
    return True


def _collect_overrides(M, K, N, kernel):
    merged = {}
    matched = 0
    for rule in RESIDUAL_REFINEMENT_RULES:
        if not isinstance(rule, dict):
            continue
        condition = rule.get("condition", {})
        if not _matches_condition(condition, M, K, N, kernel):
            continue
        overrides = rule.get("overrides", {})
        if not isinstance(overrides, dict):
            continue
        matched += 1
        for key in ("TM", "TN", "TK", "j_pack_width"):
            if key not in overrides:
                continue
            try:
                merged[key] = int(overrides[key])
            except (TypeError, ValueError):
                continue
    return merged, matched


def _select_tile_sizes(M, K, N, kernel="qkv"):
    TK = 8

    if N >= 512:
        TN = 64
    else:
        TN = max(_VEC_WIDTH, (N // _VEC_WIDTH) * _VEC_WIDTH)

    if M <= 32:
        TM = M
    elif M % 64 == 0:
        TM = 64
    else:
        TM = 32

    TM = min(TM, M)
    TN = min(TN, N)
    TK = min(TK, K)

    if TN < _VEC_WIDTH:
        TN = min(_VEC_WIDTH, N)

    return TM, TN, TK


def _apply_tile_overrides(TM, TN, TK, M, K, N, overrides):
    if "TM" in overrides and int(overrides["TM"]) > 0:
        TM = int(overrides["TM"])
    if "TN" in overrides and int(overrides["TN"]) > 0:
        TN = int(overrides["TN"])
    if "TK" in overrides and int(overrides["TK"]) > 0:
        TK = int(overrides["TK"])

    TM = min(max(1, TM), M)
    TN = min(max(1, TN), N)
    TK = min(max(1, TK), K)

    if TN < _VEC_WIDTH:
        TN = min(_VEC_WIDTH, N)

    return TM, TN, TK


def _resolve_j_pack(TN, overrides):
    j_pack = min(_VEC_WIDTH * _J_PACK_MULT, TN)
    if "j_pack_width" in overrides:
        try:
            j_pack = int(overrides["j_pack_width"])
        except (TypeError, ValueError):
            pass

    j_pack = min(max(1, j_pack), TN)
    if j_pack < _VEC_WIDTH:
        j_pack = min(_VEC_WIDTH, TN)
    if j_pack > _VEC_WIDTH and (j_pack % _VEC_WIDTH) != 0:
        j_pack = (j_pack // _VEC_WIDTH) * _VEC_WIDTH
        if j_pack <= 0:
            j_pack = min(_VEC_WIDTH, TN)
    return j_pack


def _should_unroll_k(TK):
    return TK <= _UNROLL_LIMIT


def apply_rule_based_schedule(mod, M, K, N, kernel="qkv"):
    base_tm, base_tn, base_tk = _select_tile_sizes(M, K, N, kernel)
    overrides, matched_rules = _collect_overrides(M, K, N, kernel)
    TM, TN, TK = _apply_tile_overrides(base_tm, base_tn, base_tk, M, K, N, overrides)
    j_pack = _resolve_j_pack(TN, overrides)

    print(
        f"    [rule_based_residual_refined] M={M} K={K} N={N}  kernel={kernel}  "
        f"-> TM={TM} TN={TN} TK={TK}  VEC={_VEC_WIDTH}  "
        f"j_pack={j_pack}  "
        f"unroll_k={'yes' if _should_unroll_k(TK) else 'no'}  "
        f"cache_write={_CACHE_WRITE_SCOPE}  auto_unroll={_AUTO_UNROLL_STEP}  "
        f"matched_rules={matched_rules}"
    )

    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)

    i_outer, i_inner = sch.split(i, factors=[None, TM])
    j_outer, j_inner = sch.split(j, factors=[None, TN])
    k_outer, k_inner = sch.split(k, factors=[None, TK])

    j_inner_outer, j_vec = sch.split(j_inner, factors=[None, j_pack])

    sch.reorder(i_outer, j_outer, k_outer, i_inner, j_inner_outer, k_inner, j_vec)

    c_write = sch.cache_write(block, 0, _CACHE_WRITE_SCOPE)
    sch.reverse_compute_at(c_write, j_outer)

    fused = sch.fuse(i_outer, j_outer)
    sch.parallel(fused)

    sch.vectorize(j_vec)

    write_loops = sch.get_loops(c_write)
    if write_loops:
        _, write_inner = sch.split(write_loops[-1], factors=[None, j_pack])
        sch.vectorize(write_inner)

    if _should_unroll_k(TK):
        sch.unroll(k_inner)

    sch.annotate(fused, "pragma_auto_unroll_max_step", _AUTO_UNROLL_STEP)
    sch.annotate(fused, "pragma_unroll_explicit", 1)

    sch.decompose_reduction(block, k_outer)
    return sch.mod


def describe_tile_sizes(M, K, N, kernel="qkv"):
    base_tm, base_tn, base_tk = _select_tile_sizes(M, K, N, kernel)
    overrides, matched_rules = _collect_overrides(M, K, N, kernel)
    TM, TN, TK = _apply_tile_overrides(base_tm, base_tn, base_tk, M, K, N, overrides)
    j_pack = _resolve_j_pack(TN, overrides)

    par_tasks = (M // TM) * (N // TN) if TM <= M and TN <= N else "N/A"
    ws_ab_bytes = 4 * (TM * TK + TK * TN)
    ws_c_bytes = 4 * (TM * TN)
    ws_total_bytes = ws_ab_bytes + ws_c_bytes
    return {
        "TM": TM,
        "TN": TN,
        "TK": TK,
        "VEC": _VEC_WIDTH,
        "j_pack_width": j_pack,
        "unroll_k": _should_unroll_k(TK),
        "auto_unroll_step": _AUTO_UNROLL_STEP,
        "cache_write": True,
        "cache_write_scope": _CACHE_WRITE_SCOPE,
        "parallel_tasks": par_tasks,
        "working_set_ab_bytes": ws_ab_bytes,
        "working_set_c_local_bytes": ws_c_bytes,
        "working_set_total_bytes": ws_total_bytes,
        "working_set_pct_l1": round(100 * ws_ab_bytes / 32768, 1),
        "matched_refinement_rules": matched_rules,
    }
'''


def _bootstrap_refined_scheduler_if_missing(refined_scheduler_path: str) -> bool:
    if os.path.exists(refined_scheduler_path):
        return False
    os.makedirs(os.path.dirname(refined_scheduler_path), exist_ok=True)
    with open(refined_scheduler_path, "w", encoding="utf-8") as f:
        f.write(_refined_scheduler_template())
    return True


def _import_scheduler_module(refined_scheduler_path: str):
    refined_path = os.path.abspath(refined_scheduler_path)
    root = os.path.abspath(".")
    refined_module = "research.workloads.common.rule_based_residual_refined"
    base_module = "research.workloads.common.rule_based_schedule"

    if refined_path.startswith(root) and os.path.exists(refined_path) and os.path.getsize(refined_path) > 0:
        mod = importlib.import_module(refined_module)
        mod = importlib.reload(mod)
        return {
            "module": mod,
            "variant": REFINED_VARIANT_NAME,
            "path": refined_scheduler_path,
        }

    mod = importlib.import_module(base_module)
    mod = importlib.reload(mod)
    return {
        "module": mod,
        "variant": "rule_based",
        "path": "research/workloads/common/rule_based_schedule.py",
    }


def _condition_to_rule(condition: str) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}

    m_kernel = re.search(r"kernel\s*==\s*'([^']+)'", condition)
    if m_kernel:
        parsed["kernel"] = m_kernel.group(1)

    m_eq = re.search(r"M\s*==\s*([0-9]+)", condition)
    if m_eq:
        parsed["M_eq"] = int(m_eq.group(1))

    m_min = re.search(r"M\s*>?=\s*([0-9]+)", condition)
    if m_min:
        parsed["M_min"] = int(m_min.group(1))

    m_max = re.search(r"M\s*<=\s*([0-9]+)", condition)
    if m_max:
        parsed["M_max"] = int(m_max.group(1))

    m_mod = re.search(r"M\s*%\s*([0-9]+)\s*==\s*0", condition)
    if m_mod:
        parsed["M_mod"] = int(m_mod.group(1))

    return parsed


def _suggestion_specificity(condition: str) -> float:
    parsed = _condition_to_rule(condition)
    score = 0.0

    if "kernel" in parsed:
        score += 2.0
    if "M_eq" in parsed:
        score += 2.0
    if "M_min" in parsed or "M_max" in parsed:
        try:
            m_min = int(parsed.get("M_min", parsed.get("M_eq", 0)))
            m_max = int(parsed.get("M_max", parsed.get("M_eq", 0)))
            width = max(0, m_max - m_min)
        except (TypeError, ValueError):
            width = 384
        score += max(0.0, 1.5 - min(float(width), 384.0) / 384.0)
    if "M_mod" in parsed:
        score += 0.25
    if not parsed:
        score -= 1.0
    return score


def _suggestions_to_rules(suggestions: Sequence[Dict[str, Any]], max_rules: int) -> List[Dict[str, Any]]:
    rules: List[Dict[str, Any]] = []
    used_rule_keys = set()
    ranked = sorted(
        suggestions,
        key=lambda s: (
            -_suggestion_specificity(str(s.get("condition", ""))),
            -float(s.get("confidence", 0.0)),
            -float(s.get("expected_gain_pct", 0.0)),
            -int(s.get("sample_count", 0)),
        ),
    )

    for s in ranked:
        param = str(s.get("parameter", ""))

        # Only safe tile-level refinements
        if param not in ("TM", "TN", "TK", "j_pack_width"):
            continue

        sample_count = int(s.get("sample_count", 0))
        confidence = float(s.get("confidence", 0.0))
        expected_gain_pct = float(s.get("expected_gain_pct", 0.0))

        condition_text = str(s.get("condition", "")).strip()
        condition = _condition_to_rule(condition_text)
        if not condition:
            continue

        is_exact_rule = "kernel" in condition and "M_eq" in condition

        # Allow exact, shape-specific rules to be tried with less evidence.
        if is_exact_rule:
            if sample_count < 1:
                continue
            if confidence < 0.70:
                continue
            if expected_gain_pct < 10.0:
                continue
        else:
            # Require repeated evidence for broad rules
            if sample_count < 4:
                continue
            if confidence < 0.74:
                continue
            if expected_gain_pct < 20.0:
                continue

        value = s.get("suggested_heuristic", {}).get(param)
        if value is None:
            continue
        try:
            value = int(str(value))
        except ValueError:
            continue

        current_value = s.get("current_heuristic", {}).get(param)
        try:
            current_value = int(str(current_value))
        except (TypeError, ValueError):
            current_value = None

        # Guardrail: avoid auto-applying smaller tile sizes; these repeatedly
        # regressed in iterative sweeps (e.g., TK 8 -> 6/4, TN 64 -> 32).
        if current_value is not None and param in ("TM", "TN", "TK") and value < current_value:
            continue

        # Hard safety caps to prevent pathological tiles.
        # Exact-shape rules are allowed a wider range because they are
        # validated against the exact matching shape before persistence.
        if is_exact_rule:
            if param == "TM" and value > 256:
                continue
            if param == "TN" and value > 256:
                continue
            if param == "TK" and value > 32:
                continue
        else:
            if param == "TM" and value > 128:
                continue
            if param == "TN" and value > 256:
                continue
            if param == "TK" and value > 16:
                continue

        rule_key = json.dumps({"c": condition, "o": {param: value}}, sort_keys=True)
        if rule_key in used_rule_keys:
            continue

        rules.append(
            {
                "condition": condition,
                "overrides": {param: value},
                "source": {
                    "from_condition": condition_text,
                    "confidence": confidence,
                    "expected_gain_pct": expected_gain_pct,
                    "sample_count": sample_count,
                },
            }
        )
        used_rule_keys.add(rule_key)

        if len(rules) >= int(max_rules):
            break
    return rules


def _read_refined_rules(refined_scheduler_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(refined_scheduler_path):
        return []
    with open(refined_scheduler_path, "r", encoding="utf-8") as f:
        src = f.read()
    m = re.search(
        r"# BEGIN_RESIDUAL_REFINEMENTS\s*RESIDUAL_REFINEMENT_RULES\s*=\s*(\[[\s\S]*?\])\s*# END_RESIDUAL_REFINEMENTS",
        src,
    )
    if not m:
        return []
    try:
        parsed = ast.literal_eval(m.group(1))
        if isinstance(parsed, list):
            return parsed
    except Exception:
        return []
    return []


def _merge_rules(existing: Sequence[Dict[str, Any]], incoming: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = list(existing)
    seen = set()
    for r in merged:
        seen.add(json.dumps({"c": r.get("condition", {}), "o": r.get("overrides", {})}, sort_keys=True))
    for r in incoming:
        key = json.dumps({"c": r.get("condition", {}), "o": r.get("overrides", {})}, sort_keys=True)
        if key in seen:
            continue
        merged.append(r)
        seen.add(key)
    return merged


def _patch_refined_scheduler_rules(refined_scheduler_path: str, merged_rules: Sequence[Dict[str, Any]]) -> None:
    if not os.path.exists(refined_scheduler_path):
        raise RuntimeError(
            f"Refined scheduler file is missing at {refined_scheduler_path}. "
            "Expected research/workloads/common/rule_based_residual_refined.py"
        )
    with open(refined_scheduler_path, "r", encoding="utf-8") as f:
        src = f.read()

    replacement = (
        "# BEGIN_RESIDUAL_REFINEMENTS\n"
        f"RESIDUAL_REFINEMENT_RULES = {json.dumps(list(merged_rules), indent=2)}\n"
        "# END_RESIDUAL_REFINEMENTS"
    )
    new_src, count = re.subn(
        r"# BEGIN_RESIDUAL_REFINEMENTS\s*RESIDUAL_REFINEMENT_RULES\s*=\s*\[[\s\S]*?\]\s*# END_RESIDUAL_REFINEMENTS",
        replacement,
        src,
    )
    if count != 1:
        raise RuntimeError("Failed to patch residual refinement rules block in refined scheduler.")

    with open(refined_scheduler_path, "w", encoding="utf-8") as f:
        f.write(new_src)


def _upsert_variant_results(
    results_file: str,
    variant: str,
    run_rows: Sequence[Dict[str, Any]],
    target: str,
) -> None:
    payload = _load_json(results_file, default=[])
    existing = payload if isinstance(payload, list) else []

    keys = {
        (str(r.get("kernel")), int(r.get("M", 0)), int(r.get("K", 0)), int(r.get("N", 0)))
        for r in run_rows
    }
    kept = []
    for row in existing:
        if not isinstance(row, dict):
            continue
        if str(row.get("variant", "")) != variant:
            kept.append(row)
            continue
        k = (
            str(row.get("kernel", "")),
            int(row.get("M", 0)),
            int(row.get("K", 0)),
            int(row.get("N", 0)),
        )
        if k not in keys:
            kept.append(row)

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    for r in run_rows:
        kept.append(
            {
                "kernel": r.get("kernel"),
                "variant": variant,
                "M": int(r.get("M", 0)),
                "K": int(r.get("K", 0)),
                "N": int(r.get("N", 0)),
                "latency_us": float(r.get("rule_latency_us", 0.0)),
                "std_us": float(r.get("rule_std_us", 0.0)),
                "number": int(r.get("number", 0)),
                "repeat": int(r.get("repeat", 0)),
                "min_repeat_ms": int(r.get("min_repeat_ms", 0)),
                "target": target,
                "source": "residual_rule_refiner",
                "timestamp": timestamp,
            }
        )

    _write_json(results_file, kept)


def _select_kernels(args: argparse.Namespace) -> List[str]:
    if args.kernel:
        return [args.kernel]
    return list(KERNELS.keys())


def _shape_key(kernel: str, m: int, k: int, n: int) -> Tuple[str, int, int, int]:
    return (kernel, int(m), int(k), int(n))


def _parse_ints(text: str) -> List[int]:
    return [int(v) for v in re.findall(r"-?[0-9]+", text)]


def _fallback_trace_summary(trace_text: str) -> Dict[str, Any]:
    split_factors: List[List[int]] = []
    for match in re.finditer(r"sample_perfect_tile\([^\)]*decision=\[([^\]]+)\]", trace_text):
        factors = _parse_ints(match.group(1))
        if factors:
            split_factors.append(factors)
    vector_vals = [int(v) for v in re.findall(r"ann_key=\"meta_schedule\\.vectorize\", ann_val=([0-9]+)", trace_text)]
    unroll_vals = [int(v) for v in re.findall(r"ann_key=\"meta_schedule\\.unroll_explicit\", ann_val=([0-9]+)", trace_text)]
    return {
        "sample_perfect_tile_count": len(split_factors),
        "split_factor_signatures": split_factors,
        "vector_widths": vector_vals,
        "unroll_values": unroll_vals,
        "reduction_decompose_count": trace_text.count("decompose_reduction("),
        "cache_write_count": trace_text.count("cache_write("),
        "reverse_compute_at_count": trace_text.count("reverse_compute_at("),
    }


def _extract_parallel_pattern(trace_text: str) -> Optional[str]:
    fuse_defs: Dict[str, int] = {}
    for line in trace_text.splitlines():
        m_fuse = re.search(r"(l\d+)\s*=\s*sch\.fuse\(([^\)]*)\)", line)
        if m_fuse:
            loops = [x.strip() for x in m_fuse.group(2).split(",") if x.strip().startswith("l")]
            fuse_defs[m_fuse.group(1)] = len(loops)

    for line in trace_text.splitlines():
        m_par = re.search(r"sch\.parallel\(loop=(l\d+)\)", line)
        if not m_par:
            m_par = re.search(r"sch\.parallel\((l\d+)\)", line)
        if m_par:
            loop_var = m_par.group(1)
            fused_rank = fuse_defs.get(loop_var)
            if fused_rank is not None:
                return f"parallel_on_fused_{fused_rank}_loops"
            return "parallel_on_single_loop"
    return None


def _extract_metaschedule_decisions(trace_text: str, summary: Dict[str, Any]) -> Dict[str, Any]:
    tile_signatures = re.findall(r"sample_perfect_tile\([^\)]*decision=\[([^\]]+)\]", trace_text)
    factors = [_parse_ints(sig) for sig in tile_signatures]

    tm = None
    tn = None
    tk = None
    if len(factors) >= 1 and factors[0]:
        tail = factors[0][-2:] if len(factors[0]) >= 2 else factors[0]
        tm = int(math.prod(tail))
    if len(factors) >= 2 and factors[1]:
        tail = factors[1][-2:] if len(factors[1]) >= 2 else factors[1]
        tn = int(math.prod(tail))
    if len(factors) >= 3 and factors[2]:
        tk = int(factors[2][-1])

    vector_widths = summary.get("vector_widths", []) or []
    unroll_values = summary.get("unroll_values", []) or []

    reverse_compute = re.search(r"reverse_compute_at\([^\)]*loop=(l\d+)", trace_text)
    decompose = re.search(r"decompose_reduction\([^\)]*loop=(l\d+)", trace_text)

    return {
        "TM": tm,
        "TN": tn,
        "TK": tk,
        "j_pack_width": max(vector_widths) if vector_widths else None,
        "vector_width": max(vector_widths) if vector_widths else None,
        "unroll": max(unroll_values) if unroll_values else None,
        "cache_write_position": reverse_compute.group(1) if reverse_compute else None,
        "decompose_reduction_location": decompose.group(1) if decompose else None,
        "parallel_fusion_pattern": _extract_parallel_pattern(trace_text),
    }


def _build_trace_map(best_schedules_file: str) -> Dict[Tuple[str, int, int, int], Dict[str, Any]]:
    payload = _load_json(best_schedules_file, default=[])
    trace_map: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
    if not isinstance(payload, list):
        return trace_map

    for row in payload:
        if not isinstance(row, dict):
            continue
        kernel = row.get("kernel")
        if kernel not in KERNELS:
            continue
        try:
            m = int(row["M"])
            k = int(row["K"])
            n = int(row["N"])
            latency = float(row["latency_us"])
        except (KeyError, TypeError, ValueError):
            continue

        trace_text = str(row.get("trace", ""))
        if summarize_trace_text is not None:
            summary = summarize_trace_text(trace_text)
        else:
            summary = _fallback_trace_summary(trace_text)

        decisions = _extract_metaschedule_decisions(trace_text, summary)
        trace_map[_shape_key(kernel, m, k, n)] = {
            "latency_us": latency,
            "trace": trace_text,
            "trace_summary": summary,
            "metaschedule_decisions": decisions,
        }

    return trace_map


def _build_variant_latency_map(results_file: str, variant: str) -> Dict[Tuple[str, int, int, int], float]:
    payload = _load_json(results_file, default=[])
    best: Dict[Tuple[str, int, int, int], float] = {}
    if not isinstance(payload, list):
        return best

    for row in payload:
        if not isinstance(row, dict):
            continue
        if str(row.get("variant", "")) != variant:
            continue
        kernel = row.get("kernel")
        if kernel not in KERNELS:
            continue
        try:
            m = int(row["M"])
            k = int(row["K"])
            n = int(row["N"])
            latency = float(row["latency_us"])
        except (KeyError, TypeError, ValueError):
            continue

        key = _shape_key(kernel, m, k, n)
        prev = best.get(key)
        if prev is None or latency < prev:
            best[key] = latency
    return best


def _extract_comparison_rows(payload: Any) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not isinstance(payload, dict):
        return rows

    comparisons = payload.get("comparisons", [])
    if not isinstance(comparisons, list):
        return rows

    for comp in comparisons:
        if not isinstance(comp, dict):
            continue
        table = comp.get("shape_comparison_table", [])
        if not isinstance(table, list):
            continue
        base_name = str(comp.get("baseline", {}).get("config", {}).get("name", "baseline"))
        cand_name = str(comp.get("candidate", {}).get("config", {}).get("name", "candidate"))
        for row in table:
            if not isinstance(row, dict):
                continue
            merged = dict(row)
            merged["baseline_name"] = base_name
            merged["candidate_name"] = cand_name
            rows.append(merged)
    return rows


def _build_latency_maps_from_comparisons(
    comparison_files: Sequence[str],
) -> Tuple[Dict[Tuple[str, int, int, int], float], Dict[Tuple[str, int, int, int], float]]:
    meta: Dict[Tuple[str, int, int, int], float] = {}
    rule: Dict[Tuple[str, int, int, int], float] = {}

    for path in comparison_files:
        payload = _load_json(path, default={})
        rows = _extract_comparison_rows(payload)
        for row in rows:
            kernel = row.get("kernel")
            if kernel not in KERNELS:
                continue
            try:
                m = int(row["M"])
                k = int(row["K"])
                n = int(row["N"])
            except (KeyError, TypeError, ValueError):
                continue
            key = _shape_key(kernel, m, k, n)

            base_name = str(row.get("baseline_name", "")).lower()
            cand_name = str(row.get("candidate_name", "")).lower()
            base_lat = row.get("baseline_latency_us")
            cand_lat = row.get("candidate_latency_us")

            if base_lat is not None and "meta" in base_name:
                meta[key] = float(base_lat)
            if cand_lat is not None and "meta" in cand_name:
                meta[key] = float(cand_lat)

            if base_lat is not None and "rule" in base_name:
                rule[key] = float(base_lat)
            if cand_lat is not None and "rule" in cand_name:
                rule[key] = float(cand_lat)

    return meta, rule


def _seed_for_shape(kernel: str, m: int, k: int, n: int) -> int:
    return abs(hash((kernel, m, k, n))) % (2**31)


def _benchmark_rule_case(
    kernel: str,
    m: int,
    k: int,
    n: int,
    number: int,
    repeat: int,
    min_repeat_ms: int,
    apply_schedule_fn,
) -> Tuple[float, float]:
    seed = _seed_for_shape(kernel, m, k, n)
    rng = np.random.default_rng(seed)

    mod = matmul_tir(m, k, n)
    scheduled_mod = apply_schedule_fn(mod, m, k, n, kernel)
    rt_mod = tvm.build(scheduled_mod, target=TARGET)

    a_np = rng.standard_normal(size=(m, k), dtype=np.float32)
    b_np = rng.standard_normal(size=(k, n), dtype=np.float32)
    c_np = np.zeros((m, n), dtype=np.float32)

    a = tvm.nd.array(a_np, DEV)
    b = tvm.nd.array(b_np, DEV)
    c = tvm.nd.array(c_np, DEV)

    for _ in range(3):
        rt_mod(a, b, c)

    evaluator = rt_mod.time_evaluator(
        "main",
        dev=DEV,
        number=number,
        repeat=repeat,
        min_repeat_ms=min_repeat_ms,
    )
    result = evaluator(a, b, c)
    return float(result.mean * 1e6), float(result.std * 1e6)


def _rule_matches_shape(
    condition: Dict[str, Any], kernel: str, m: int, k: int, n: int
) -> bool:
    if not isinstance(condition, dict):
        return False
    if "kernel" in condition and str(condition["kernel"]) != str(kernel):
        return False
    if "M_eq" in condition and int(m) != int(condition["M_eq"]):
        return False
    if "M_min" in condition and int(m) < int(condition["M_min"]):
        return False
    if "M_max" in condition and int(m) > int(condition["M_max"]):
        return False
    if "M_mod" in condition:
        mod = int(condition["M_mod"])
        if mod <= 0 or int(m) % mod != 0:
            return False
    if "K_eq" in condition and int(k) != int(condition["K_eq"]):
        return False
    if "N_eq" in condition and int(n) != int(condition["N_eq"]):
        return False
    return True


def _select_rule_shapes(
    rule: Dict[str, Any], kernels: Sequence[str], limit: int = 4
) -> List[Tuple[str, int, int, int]]:
    condition = rule.get("condition", {})
    shapes: List[Tuple[str, int, int, int]] = []
    for kernel in kernels:
        shape_fn = KERNELS[kernel]
        for m in M_LIST:
            _, k, n = shape_fn(m)
            if _rule_matches_shape(condition, kernel, m, k, n):
                shapes.append((kernel, int(m), int(k), int(n)))
    # Keep validation cost bounded for iterative runs.
    return shapes[: max(1, int(limit))]


def _filter_rules_by_validation(
    args: argparse.Namespace,
    kernels: Sequence[str],
    refined_scheduler_path: str,
    existing_rules: Sequence[Dict[str, Any]],
    candidate_rules: Sequence[Dict[str, Any]],
    baseline_latency: Dict[Tuple[str, int, int, int], float],
) -> List[Dict[str, Any]]:
    if not candidate_rules or not baseline_latency:
        return []

    accepted: List[Dict[str, Any]] = []
    for idx, rule in enumerate(candidate_rules, 1):
        rule_set = list(existing_rules) + accepted + [rule]
        _patch_refined_scheduler_rules(refined_scheduler_path, rule_set)
        refined_info = _import_scheduler_module(refined_scheduler_path)
        refined_apply = getattr(refined_info["module"], "apply_rule_based_schedule")

        test_shapes = _select_rule_shapes(rule, kernels)
        if not test_shapes:
            continue

        ratios: List[float] = []
        for kernel, m, k, n in test_shapes:
            base_lat = baseline_latency.get(_shape_key(kernel, m, k, n))
            if base_lat is None or base_lat <= 0:
                continue
            new_lat, _ = _benchmark_rule_case(
                kernel,
                m,
                k,
                n,
                number=int(args.number),
                repeat=int(args.repeat),
                min_repeat_ms=int(args.min_repeat_ms),
                apply_schedule_fn=refined_apply,
            )
            ratios.append(float(new_lat) / float(base_lat))

        if not ratios:
            print(
                f"[residual-refiner] Candidate {idx}: skipped (no benchmarkable shapes for validation)"
            )
            continue

        mean_ratio = statistics.fmean(ratios)
        worst_ratio = max(ratios)
        # Accept only if it improves on average and avoids clear regressions.
        if mean_ratio <= 0.98 and worst_ratio <= 1.03:
            print(
                f"[residual-refiner] Candidate {idx}: accepted "
                f"(mean_ratio={mean_ratio:.4f}, worst_ratio={worst_ratio:.4f})"
            )
            accepted.append(rule)
        else:
            print(
                f"[residual-refiner] Candidate {idx}: rejected "
                f"(mean_ratio={mean_ratio:.4f}, worst_ratio={worst_ratio:.4f})"
            )

    return accepted


def _run_rule_sweep(
    args: argparse.Namespace,
    kernels: Sequence[str],
    apply_schedule_fn,
    variant_label: str,
    metaschedule_latency: Optional[Dict[Tuple[str, int, int, int], float]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int, int, int], float]]:
    rows: List[Dict[str, Any]] = []
    lat_map: Dict[Tuple[str, int, int, int], float] = {}

    print(f"[residual-refiner] Running {variant_label} sweep for kernels={list(kernels)}")
    for kernel in kernels:
        shape_fn = KERNELS[kernel]
        for m in M_LIST:
            _, k, n = shape_fn(m)
            mean_us, std_us = _benchmark_rule_case(
                kernel,
                m,
                k,
                n,
                number=int(args.number),
                repeat=int(args.repeat),
                min_repeat_ms=int(args.min_repeat_ms),
                apply_schedule_fn=apply_schedule_fn,
            )

            key = _shape_key(kernel, m, k, n)
            lat_map[key] = mean_us
            row = {
                "kernel": kernel,
                "M": m,
                "K": k,
                "N": n,
                "rule_latency_us": mean_us,
                "rule_std_us": std_us,
                "number": int(args.number),
                "repeat": int(args.repeat),
                "min_repeat_ms": int(args.min_repeat_ms),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            rows.append(row)
            line = (
                f"  kernel={kernel:10s} M={m:4d} K={k:4d} N={n:4d} "
                f"rule_latency={mean_us:.4f} us (+/- {std_us:.4f})"
            )
            meta_us = None
            if metaschedule_latency is not None:
                meta_us = metaschedule_latency.get(key)

            if meta_us is not None and meta_us > 0:
                gap = mean_us / meta_us
                line += f"  meta_latency={meta_us:.4f} us gap={gap:.4f}"
                if mean_us < meta_us:
                    line = _colorize(line, ANSI_GREEN)
                elif mean_us > meta_us:
                    line = _colorize(line, ANSI_RED)
            print(line)

    return rows, lat_map


def _full_sweep_is_better(
    baseline_latency: Dict[Tuple[str, int, int, int], float],
    refined_rows: Sequence[Dict[str, Any]],
) -> Tuple[bool, Dict[str, float]]:
    refined_latency: Dict[Tuple[str, int, int, int], float] = {}
    for row in refined_rows:
        try:
            key = _shape_key(
                str(row.get("kernel", "")),
                int(row.get("M", 0)),
                int(row.get("K", 0)),
                int(row.get("N", 0)),
            )
            refined_latency[key] = float(row.get("rule_latency_us", 0.0))
        except (TypeError, ValueError):
            continue

    ratios: List[float] = []
    for key, base_lat in baseline_latency.items():
        refined_lat = refined_latency.get(key)
        if base_lat is None or refined_lat is None or base_lat <= 0 or refined_lat <= 0:
            continue
        ratios.append(float(refined_lat) / float(base_lat))

    if not ratios:
        return False, {"mean_ratio": float("inf"), "worst_ratio": float("inf")}

    mean_ratio = statistics.fmean(ratios)
    worst_ratio = max(ratios)
    keep = mean_ratio <= 0.995 and worst_ratio <= 1.05
    return keep, {"mean_ratio": float(mean_ratio), "worst_ratio": float(worst_ratio)}


def _residual_metrics(cases: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    gaps = [float(case.get("residual_gap", 1.0)) for case in cases]
    if not gaps:
        return {"count": 0.0, "mean_gap": 0.0, "worst_gap": 0.0}
    return {
        "count": float(len(gaps)),
        "mean_gap": float(statistics.fmean(gaps)),
        "worst_gap": float(max(gaps)),
    }


def _rule_decision_dict(m: int, k: int, n: int, kernel: str) -> Dict[str, Any]:
    tiles = describe_tile_sizes(m, k, n, kernel)
    return {
        "TM": tiles.get("TM"),
        "TN": tiles.get("TN"),
        "TK": tiles.get("TK"),
        "j_pack_width": tiles.get("j_pack_width"),
        "vector_width": tiles.get("VEC"),
        "unroll": tiles.get("TK") if bool(tiles.get("unroll_k")) else 0,
        "cache_write_position": "j_outer",
        "decompose_reduction_location": "k_outer",
        "parallel_fusion_pattern": "parallel_on_fused_2_loops",
    }


def _build_residual_cases(
    kernels: Sequence[str],
    threshold: float,
    rule_latency: Dict[Tuple[str, int, int, int], float],
    metaschedule_latency: Dict[Tuple[str, int, int, int], float],
    trace_map: Dict[Tuple[str, int, int, int], Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_rows: List[Dict[str, Any]] = []
    residual_rows: List[Dict[str, Any]] = []

    for kernel in kernels:
        shape_fn = KERNELS[kernel]
        for m in M_LIST:
            _, k, n = shape_fn(m)
            key = _shape_key(kernel, m, k, n)
            rule_lat = rule_latency.get(key)
            meta_lat = metaschedule_latency.get(key)
            if rule_lat is None or meta_lat is None or meta_lat <= 0:
                continue

            gap = float(rule_lat / meta_lat)
            trace_entry = trace_map.get(key, {})
            row = {
                "kernel": kernel,
                "M": m,
                "K": k,
                "N": n,
                "rule_latency_us": float(rule_lat),
                "metaschedule_latency_us": float(meta_lat),
                "residual_gap": gap,
                "trace_summary": trace_entry.get("trace_summary"),
                "rule_decisions": _rule_decision_dict(m, k, n, kernel),
                "metaschedule_decisions": trace_entry.get("metaschedule_decisions", {}),
            }
            all_rows.append(row)
            if gap > threshold:
                residual_rows.append(row)

    return all_rows, residual_rows


DIFF_KEYS = [
    "TM",
    "TN",
    "TK",
    "j_pack_width",
    "vector_width",
    "unroll",
    "cache_write_position",
    "decompose_reduction_location",
    "parallel_fusion_pattern",
]


def _diff_one_case(case: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    diffs: Dict[str, Dict[str, Any]] = {}
    rule_dec = case.get("rule_decisions", {})
    meta_dec = case.get("metaschedule_decisions", {})
    for key in DIFF_KEYS:
        r = rule_dec.get(key)
        m = meta_dec.get(key)
        if m is None:
            continue
        if r != m:
            diffs[key] = {"rule": r, "metaschedule": m}
    return diffs


def _mine_patterns(cases: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    bucket: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for case in cases:
        diffs = _diff_one_case(case)
        case["decision_differences"] = diffs
        for key, values in diffs.items():
            sig = (key, str(values.get("rule")), str(values.get("metaschedule")))
            bucket[sig].append(case)

    patterns: List[Dict[str, Any]] = []
    for (param, rule_val, meta_val), rows in bucket.items():
        gaps = [float(r.get("residual_gap", 1.0)) for r in rows]
        m_vals = [int(r["M"]) for r in rows]
        shapes = sorted({(int(r["M"]), int(r["K"]), int(r["N"])) for r in rows})
        kernels = sorted({str(r["kernel"]) for r in rows})
        patterns.append(
            {
                "parameter": param,
                "rule_value": rule_val,
                "metaschedule_value": meta_val,
                "occurrences": len(rows),
                "kernels": kernels,
                "M_values": sorted(set(m_vals)),
                "shapes": [[int(a), int(b), int(c)] for a, b, c in shapes],
                "avg_residual_gap": round(float(statistics.fmean(gaps)), 6),
                "max_residual_gap": round(float(max(gaps)), 6),
            }
        )

    patterns.sort(
        key=lambda p: (
            -int(p.get("occurrences", 0)),
            -float(p.get("avg_residual_gap", 0.0)),
            str(p.get("parameter", "")),
        )
    )
    return patterns


def _condition_from_rows(rows: Sequence[Dict[str, Any]], parameter: str, suggested_val: str) -> str:
    kernels = sorted({str(r["kernel"]) for r in rows})
    m_vals = sorted({int(r["M"]) for r in rows})
    cond_parts: List[str] = []

    if len(kernels) == 1:
        cond_parts.append(f"kernel == '{kernels[0]}'")
    if m_vals:
        m_min = min(m_vals)
        m_max = max(m_vals)
        if m_min == m_max:
            cond_parts.append(f"M == {m_min}")
        else:
            cond_parts.append(f"M >= {m_min} and M <= {m_max}")

    if parameter in ("TM", "TN", "TK"):
        try:
            val = int(suggested_val)
            if val > 0 and m_vals and all((m % val) == 0 for m in m_vals):
                cond_parts.append(f"M % {val} == 0")
        except ValueError:
            pass

    if not cond_parts:
        return "true"
    return " and ".join(cond_parts)


def _confidence_score(rows: Sequence[Dict[str, Any]], total_cases: int) -> float:
    if not rows or total_cases <= 0:
        return 0.0
    gaps = [float(r.get("residual_gap", 1.0)) for r in rows]
    avg_gain = max(0.0, statistics.fmean(gaps) - 1.0)
    support = min(1.0, len(rows) / float(total_cases))
    consistency = 1.0
    if len(gaps) > 1:
        try:
            std = statistics.pstdev(gaps)
            consistency = max(0.0, 1.0 - min(std, 0.5) / 0.5)
        except statistics.StatisticsError:
            consistency = 1.0
    score = 0.45 + 0.30 * support + 0.20 * min(avg_gain / 0.20, 1.0) + 0.05 * consistency
    return round(min(score, 0.99), 3)


def _build_suggestions(
    cases: Sequence[Dict[str, Any]],
    patterns: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not cases:
        return []

    # Recreate mapping pattern -> rows for richer shape lists in final output.
    grouped_rows: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in cases:
        for key, vals in row.get("decision_differences", {}).items():
            sig = (key, str(vals.get("rule")), str(vals.get("metaschedule")))
            grouped_rows[sig].append(row)

    suggestions: List[Dict[str, Any]] = []
    total_cases = len(cases)

    for pattern in patterns:
        param = str(pattern["parameter"])
        rule_val = str(pattern["rule_value"])
        meta_val = str(pattern["metaschedule_value"])
        sig = (param, rule_val, meta_val)
        rows = grouped_rows.get(sig, [])
        if not rows:
            continue

        gaps = [float(r.get("residual_gap", 1.0)) for r in rows]
        expected_gain = max(0.0, statistics.fmean(gaps) - 1.0) * 100.0
        # suppress unrealistic gains from single noisy outliers
        if len(rows) < 3:
            expected_gain *= 0.5
        condition = _condition_from_rows(rows, param, meta_val)

        affected_shapes = [
            {
                "kernel": r["kernel"],
                "M": r["M"],
                "K": r["K"],
                "N": r["N"],
                "residual_gap": round(float(r["residual_gap"]), 6),
            }
            for r in rows
        ]

        suggestion = {
            "parameter": param,
            "condition": condition,
            "current_heuristic": {param: rule_val},
            "suggested_heuristic": {param: meta_val},
            "affected_shapes": affected_shapes,
            "expected_gain_pct": round(float(expected_gain), 4),
            "confidence": _confidence_score(rows, total_cases),
            "sample_count": len(rows),
        }
        suggestions.append(suggestion)

    suggestions.sort(
        key=lambda s: (
            -float(s.get("confidence", 0.0)),
            -float(s.get("expected_gain_pct", 0.0)),
            -int(s.get("sample_count", 0)),
        )
    )
    return suggestions


def _build_patch_candidate(suggestions: Sequence[Dict[str, Any]]) -> str:
    tm_suggestion = None
    tn_suggestion = None
    tk_suggestion = None

    for s in suggestions:
        param = s.get("parameter")
        if param == "TM" and tm_suggestion is None:
            tm_suggestion = s
        elif param == "TN" and tn_suggestion is None:
            tn_suggestion = s
        elif param == "TK" and tk_suggestion is None:
            tk_suggestion = s

    lines = [
        "diff --git a/research/workloads/common/rule_based_schedule.py b/research/workloads/common/rule_based_schedule.py",
        "--- a/research/workloads/common/rule_based_schedule.py",
        "+++ b/research/workloads/common/rule_based_schedule.py",
        "@@ def _select_tile_sizes(M, K, N, kernel=\"qkv\"):",
        "-    # F7: TM selection (use full M when small; prefer 64 if divisible)",
        "+    # F7: TM selection (residual-refiner candidate clauses)",
    ]

    if tm_suggestion:
        sval = tm_suggestion.get("suggested_heuristic", {}).get("TM")
        cond = str(tm_suggestion.get("condition", "true")).replace("kernel == ", "")
        try:
            sval_int = int(str(sval))
            lines.append(f"+    # Candidate: if {cond}, prefer TM={sval_int}")
            lines.append("+    # if <translated condition>: TM = <suggested value>")
        except ValueError:
            lines.append(f"+    # Candidate TM suggestion: {tm_suggestion}")

    lines.extend(
        [
            "     if M <= 32:",
            "         TM = M                     # Full M - minimal outer-loop overhead",
            "     elif M % 64 == 0:",
            "         TM = 64                    # Clean division, larger tiles",
            "@@ def _select_tile_sizes(M, K, N, kernel=\"qkv\"):",
        ]
    )

    if tn_suggestion:
        lines.append(
            "+    # Candidate TN rule: "
            f"if {tn_suggestion.get('condition', 'true')} then TN={tn_suggestion.get('suggested_heuristic', {}).get('TN')}"
        )
    if tk_suggestion:
        lines.append(
            "+    # Candidate TK rule: "
            f"if {tk_suggestion.get('condition', 'true')} then TK={tk_suggestion.get('suggested_heuristic', {}).get('TK')}"
        )

    lines.append("+    # Note: This diff is advisory and must be translated into concrete conditions.")
    return "\n".join(lines) + "\n"


def _comparison_candidates(args: argparse.Namespace) -> List[str]:
    if args.comparison_json:
        return [p for p in args.comparison_json if os.path.exists(p)]
    return [p for p in DEFAULT_COMPARE_FILES if os.path.exists(p)]


def main() -> None:
    random.seed(0)
    np.random.seed(0)

    args = _parse_args()
    kernels = _select_kernels(args)
    comparison_files = _comparison_candidates(args)

    created_refined_scheduler = _bootstrap_refined_scheduler_if_missing(args.refined_scheduler)
    if created_refined_scheduler:
        print(
            f"[residual-refiner] Created missing refined scheduler scaffold at {args.refined_scheduler}"
        )

    scheduler_info = _import_scheduler_module(args.refined_scheduler)
    scheduler_mod = scheduler_info["module"]
    apply_fn = getattr(scheduler_mod, "apply_rule_based_schedule")
    describe_fn = getattr(scheduler_mod, "describe_tile_sizes")
    active_variant = str(scheduler_info["variant"])

    global describe_tile_sizes
    describe_tile_sizes = describe_fn

    trace_map = _build_trace_map(args.best_schedules)
    meta_from_comp, rule_from_comp = _build_latency_maps_from_comparisons(comparison_files)
    meta_from_results = _build_variant_latency_map(args.results_file, "metaschedule")
    rule_from_results = _build_variant_latency_map(args.results_file, "rule_based")

    metaschedule_latency: Dict[Tuple[str, int, int, int], float] = {}
    rule_latency: Dict[Tuple[str, int, int, int], float] = {}
    metaschedule_latency.update(meta_from_results)
    metaschedule_latency.update(meta_from_comp)
    for key, row in trace_map.items():
        if "latency_us" in row:
            metaschedule_latency[key] = float(row["latency_us"])

    if args.suggestion_only:
        residual_payload = _load_json(args.residual_cases, default={})
        residual_cases = residual_payload.get("residual_cases", []) if isinstance(residual_payload, dict) else []
        residual_cases = [r for r in residual_cases if str(r.get("kernel")) in set(kernels)]
        if not residual_cases:
            raise RuntimeError(
                f"No residual cases found in {args.residual_cases}. Run without --suggestion-only first."
            )
        print(f"[residual-refiner] Loaded {len(residual_cases)} residual cases from {args.residual_cases}")
    else:
        if args.compare_only:
            rule_latency = {}
            if active_variant == REFINED_VARIANT_NAME:
                rule_latency.update(_build_variant_latency_map(args.results_file, REFINED_VARIANT_NAME))
            else:
                rule_latency.update(rule_from_results)
            rule_latency.update(rule_from_comp)
            sweep_rows: List[Dict[str, Any]] = []
            print("[residual-refiner] Compare-only mode: using existing JSON latencies")
        else:
            sweep_rows, rule_latency = _run_rule_sweep(
                args,
                kernels,
                apply_fn,
                active_variant,
                metaschedule_latency=metaschedule_latency,
            )
            sweep_payload = {
                "metadata": {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "threshold": float(args.threshold),
                    "kernels": kernels,
                    "target": TARGET,
                    "variant": active_variant,
                    "scheduler_path": args.refined_scheduler
                    if active_variant == REFINED_VARIANT_NAME
                    else "research/workloads/common/rule_based_schedule.py",
                    "number": int(args.number),
                    "repeat": int(args.repeat),
                    "min_repeat_ms": int(args.min_repeat_ms),
                },
                "rule_sweep": sweep_rows,
            }
            _write_json(args.sweep_log, sweep_payload)
            print(f"[residual-refiner] Wrote sweep log to {args.sweep_log}")
            _upsert_variant_results(args.results_file, active_variant, sweep_rows, TARGET)
            print(
                f"[residual-refiner] Logged {active_variant} sweep rows to {args.results_file} as separate variant"
            )

        _, residual_cases = _build_residual_cases(
            kernels=kernels,
            threshold=float(args.threshold),
            rule_latency=rule_latency,
            metaschedule_latency=metaschedule_latency,
            trace_map=trace_map,
        )

        residual_payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "threshold": float(args.threshold),
                "kernels": kernels,
                "compare_only": bool(args.compare_only),
                "comparison_json_files": comparison_files,
                "best_schedules_file": args.best_schedules,
                "results_file": args.results_file,
            },
            "residual_cases": residual_cases,
        }
        _write_json(args.residual_cases, residual_payload)
        print(
            f"[residual-refiner] Residual cases (gap > {args.threshold:.4f}) "
            f"= {len(residual_cases)} -> {args.residual_cases}"
        )
        _print_residual_report(residual_cases, float(args.threshold))

        if args.compare_only:
            print("[residual-refiner] Compare-only mode complete")
            return

    if args.suggestion_only:
        patterns = _mine_patterns(residual_cases)
        suggestions = _build_suggestions(residual_cases, patterns)

        suggestions_payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "threshold": float(args.threshold),
                "kernels": kernels,
                "num_residual_cases": len(residual_cases),
                "iteration": 0,
            },
            "recurring_patterns": patterns,
            "suggestions": suggestions,
        }
        _write_json(args.suggestions, suggestions_payload)
        print(
            f"[residual-refiner] Generated {len(suggestions)} heuristic suggestions -> {args.suggestions}"
        )
        _print_pattern_report(patterns)
        _print_suggestion_report(suggestions)
        print("[residual-refiner] Suggestion-only mode complete")
        return

    current_rules = _read_refined_rules(args.refined_scheduler)
    current_rule_latency = dict(rule_latency)
    current_residual_cases = list(residual_cases)
    current_metrics = _residual_metrics(current_residual_cases)
    max_iterations = max(1, int(args.max_iterations))
    target_residuals = max(0, int(args.target_residuals))
    force_iterations = bool(args.force_iterations)

    for iteration in range(1, max_iterations + 1):
        patterns = _mine_patterns(current_residual_cases)
        suggestions = _build_suggestions(current_residual_cases, patterns)

        suggestions_payload = {
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "threshold": float(args.threshold),
                "kernels": kernels,
                "num_residual_cases": len(current_residual_cases),
                "iteration": iteration,
                "max_iterations": max_iterations,
                "target_residuals": target_residuals,
            },
            "recurring_patterns": patterns,
            "suggestions": suggestions,
        }
        _write_json(args.suggestions, suggestions_payload)
        print(
            f"[residual-refiner] Iteration {iteration}/{max_iterations}: "
            f"residual_cases={len(current_residual_cases)} "
            f"mean_gap={current_metrics['mean_gap']:.4f} worst_gap={current_metrics['worst_gap']:.4f}"
        )
        print(
            f"[residual-refiner] Generated {len(suggestions)} heuristic suggestions -> {args.suggestions}"
        )
        _print_pattern_report(patterns)
        _print_suggestion_report(suggestions)

        if int(current_metrics["count"]) <= target_residuals:
            if force_iterations:
                print(
                    f"[residual-refiner] Target residual count reached "
                    f"({int(current_metrics['count'])} <= {target_residuals}); "
                    "continuing because --force-iterations is set."
                )
                continue
            print(
                f"[residual-refiner] Target residual count reached "
                f"({int(current_metrics['count'])} <= {target_residuals}); stopping."
            )
            break

        if not suggestions:
            if force_iterations:
                print(
                    "[residual-refiner] No suggestions produced; "
                    "continuing because --force-iterations is set."
                )
                continue
            print("[residual-refiner] No suggestions produced; stopping iterative refinement.")
            break

        candidate_budget = max(8, int(args.max_auto_rules) * 4)
        incoming_rules = _suggestions_to_rules(suggestions, candidate_budget)
        print(
            f"[residual-refiner] Candidate auto-rules selected={len(incoming_rules)} "
            f"(from {len(suggestions)} suggestions, candidate_budget={candidate_budget}, "
            f"max_auto_rules={int(args.max_auto_rules)})"
        )

        if not incoming_rules:
            if force_iterations:
                print(
                    "[residual-refiner] No candidate rules survived selection; "
                    "continuing because --force-iterations is set."
                )
                continue
            print("[residual-refiner] No candidate rules survived selection; stopping.")
            break

        previous_rules = list(current_rules)
        validated_rules = _filter_rules_by_validation(
            args,
            kernels,
            args.refined_scheduler,
            current_rules,
            incoming_rules,
            current_rule_latency,
        )
        merged_rules = _merge_rules(current_rules, validated_rules)
        if len(merged_rules) == len(current_rules):
            if force_iterations:
                print(
                    "[residual-refiner] Validation produced no new rules; "
                    "continuing because --force-iterations is set."
                )
                continue
            print("[residual-refiner] Validation produced no new rules; stopping.")
            break

        _patch_refined_scheduler_rules(args.refined_scheduler, merged_rules)
        print(
            f"[residual-refiner] Updated {args.refined_scheduler} with "
            f"{len(validated_rules)} new rule(s), total={len(merged_rules)}"
        )

        refined_info = _import_scheduler_module(args.refined_scheduler)
        refined_mod = refined_info["module"]
        refined_apply = getattr(refined_mod, "apply_rule_based_schedule")
        refined_rows, refined_rule_latency = _run_rule_sweep(
            args,
            kernels,
            refined_apply,
            REFINED_VARIANT_NAME,
            metaschedule_latency=metaschedule_latency,
        )
        keep_refined, refined_gate = _full_sweep_is_better(current_rule_latency, refined_rows)
        if not keep_refined:
            _patch_refined_scheduler_rules(args.refined_scheduler, previous_rules)
            print(
                f"[residual-refiner] Rolled back {args.refined_scheduler} to the previous rules "
                f"(mean_ratio={refined_gate['mean_ratio']:.4f}, worst_ratio={refined_gate['worst_ratio']:.4f})"
            )
            if force_iterations:
                print(
                    "[residual-refiner] Full-sweep gate failed; "
                    "continuing because --force-iterations is set."
                )
                continue
            break

        _upsert_variant_results(args.results_file, REFINED_VARIANT_NAME, refined_rows, TARGET)
        print(
            f"[residual-refiner] Logged refined variant rows to {args.results_file} "
            f"with variant={REFINED_VARIANT_NAME}"
        )
        print(
            f"[residual-refiner] Refined full-sweep gate passed "
            f"(mean_ratio={refined_gate['mean_ratio']:.4f}, worst_ratio={refined_gate['worst_ratio']:.4f})"
        )

        _, refined_residual_cases = _build_residual_cases(
            kernels=kernels,
            threshold=float(args.threshold),
            rule_latency=refined_rule_latency,
            metaschedule_latency=metaschedule_latency,
            trace_map=trace_map,
        )
        refined_metrics = _residual_metrics(refined_residual_cases)
        print(
            f"[residual-refiner] Post-update residuals={int(refined_metrics['count'])} "
            f"mean_gap={refined_metrics['mean_gap']:.4f} worst_gap={refined_metrics['worst_gap']:.4f}"
        )

        if (
            int(refined_metrics["count"]) > int(current_metrics["count"]) 
            or (
                int(refined_metrics["count"]) == int(current_metrics["count"])
                and (
                    refined_metrics["mean_gap"] > current_metrics["mean_gap"]
                    or (
                        refined_metrics["mean_gap"] == current_metrics["mean_gap"]
                        and refined_metrics["worst_gap"] >= current_metrics["worst_gap"]
                    )
                )
            )
        ):
            _patch_refined_scheduler_rules(args.refined_scheduler, previous_rules)
            print(
                f"[residual-refiner] Residual metrics did not improve enough to continue; "
                f"rolled back to the previous rules (count={int(current_metrics['count'])} -> {int(refined_metrics['count'])}, "
                f"mean_gap={current_metrics['mean_gap']:.4f} -> {refined_metrics['mean_gap']:.4f})"
            )
            if force_iterations:
                print(
                    "[residual-refiner] Residual gate failed; "
                    "continuing because --force-iterations is set."
                )
                continue
            break

        current_rules = merged_rules
        current_rule_latency = refined_rule_latency
        current_residual_cases = refined_residual_cases
        current_metrics = refined_metrics

        if int(current_metrics["count"]) <= target_residuals:
            if force_iterations:
                print(
                    f"[residual-refiner] Target residual count reached "
                    f"({int(current_metrics['count'])} <= {target_residuals}); "
                    "continuing because --force-iterations is set."
                )
                continue
            print(
                f"[residual-refiner] Target residual count reached "
                f"({int(current_metrics['count'])} <= {target_residuals}); stopping."
            )
            break

    final_patterns = _mine_patterns(current_residual_cases)
    suggestions = _build_suggestions(current_residual_cases, final_patterns)
    residual_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": float(args.threshold),
            "kernels": kernels,
            "compare_only": bool(args.compare_only),
            "comparison_json_files": comparison_files,
            "best_schedules_file": args.best_schedules,
            "results_file": args.results_file,
            "iteration": int(iteration),
            "max_iterations": max_iterations,
            "target_residuals": target_residuals,
        },
        "residual_cases": current_residual_cases,
    }
    _write_json(args.residual_cases, residual_payload)

    suggestions_payload = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "threshold": float(args.threshold),
            "kernels": kernels,
            "num_residual_cases": len(current_residual_cases),
            "iteration": int(iteration),
            "max_iterations": max_iterations,
            "target_residuals": target_residuals,
        },
        "recurring_patterns": final_patterns,
        "suggestions": suggestions,
    }
    _write_json(args.suggestions, suggestions_payload)

    if args.auto_patch:
        patch_text = _build_patch_candidate(suggestions)
        os.makedirs(os.path.dirname(args.patch_file), exist_ok=True)
        with open(args.patch_file, "w", encoding="utf-8") as f:
            f.write(patch_text)
        print(f"[residual-refiner] Patch candidate written to {args.patch_file}")


if __name__ == "__main__":
    main()
