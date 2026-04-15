"""Extract ML-ready training rows from MetaSchedule best schedules.

This script reads `research/results/metaschedule/best_schedules.json`, derives
shape/runtime features and schedule-knob labels from trace text, and writes a
CSV file for downstream model training.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

LOGGER = logging.getLogger("extract_training_data")

RESEARCH_DIR = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_PATH = RESEARCH_DIR / "results" / "metaschedule" / "best_schedules.json"
DEFAULT_OUTPUT_PATH = RESEARCH_DIR / "results" / "ml_schedule_predictor" / "training_dataset.csv"
ALLOWED_VECTOR_WIDTHS = (8, 16, 32)

_SPLIT_RE = re.compile(
    r"(?P<outs>l\d+(?:\s*,\s*l\d+)*)\s*=\s*sch\.split\(loop=(?P<input>l\d+),"
    r"\s*factors=\[(?P<factors>[^\]]+)\]"
)
_VECTORIZE_RE = re.compile(r"sch\.vectorize\(loop=(?P<loop>l\d+)\)")
_PRAGMA_UNROLL_RE = re.compile(
    r"ann_key=\"pragma_auto_unroll_max_step\",\s*ann_val=(?P<value>v\d+|\d+)"
)
_META_UNROLL_RE = re.compile(
    r"ann_key=\"meta_schedule\.unroll_explicit\",\s*ann_val=(?P<value>v\d+|\d+)"
)
_META_VECTORIZE_RE = re.compile(
    r"ann_key=\"meta_schedule\.vectorize\",\s*ann_val=(?P<value>v\d+|\d+)"
)
_GET_LOOPS_RE = re.compile(
    r"(?P<loops>l\d+(?:\s*,\s*l\d+)*)\s*=\s*sch\.get_loops\(block=(?P<block>\w+)\)"
)
_SAMPLE_PERFECT_TILE_RE = re.compile(
    r"(?P<vars>v\d+(?:\s*,\s*v\d+)*)\s*=\s*sch\.sample_perfect_tile\([^\n]*decision=\[(?P<decisions>[^\]]+)\]"
)
_SAMPLE_CATEGORICAL_RE = re.compile(
    r"(?P<var>v\d+)\s*=\s*sch\.sample_categorical\(" 
    r"[^\n]*candidates=\[(?P<candidates>[^\]]+)\][^\n]*decision=(?P<decision>\d+)"
)
_FUSE_RE = re.compile(
    r"(?P<out>l\d+)\s*=\s*sch\.fuse\((?P<inputs>l\d+(?:\s*,\s*l\d+)*)"
)
_INT_RE = re.compile(r"-?\d+")


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _nearest_allowed(value: Optional[int], candidates: tuple[int, ...]) -> Optional[int]:
    if value is None:
        return None
    return min(candidates, key=lambda item: abs(item - int(value)))


def _extract_kernel_name(record: Dict[str, Any]) -> str:
    kernel = record.get("kernel_name")
    if kernel is None:
        kernel = record.get("kernel", "unknown")
    return str(kernel)


def _parse_int_list(text: str) -> List[int]:
    return [int(item) for item in _INT_RE.findall(text)]


def _resolve_value(token: str, variable_values: Dict[str, int]) -> Optional[int]:
    cleaned = token.strip()
    if not cleaned or cleaned == "None":
        return None
    if cleaned in variable_values:
        return variable_values[cleaned]
    if re.fullmatch(r"-?\d+", cleaned):
        return int(cleaned)
    cast_match = re.fullmatch(r"T\.int\d+\((-?\d+)\)", cleaned)
    if cast_match:
        return int(cast_match.group(1))
    return None


def _extract_variable_values(trace: str) -> Dict[str, int]:
    variable_values: Dict[str, int] = {}

    for match in _SAMPLE_PERFECT_TILE_RE.finditer(trace):
        vars_list = [item.strip() for item in match.group("vars").split(",") if item.strip()]
        decisions = _parse_int_list(match.group("decisions"))
        for var_name, decision in zip(vars_list, decisions):
            variable_values[var_name] = int(decision)

    for match in _SAMPLE_CATEGORICAL_RE.finditer(trace):
        var_name = match.group("var")
        candidates = _parse_int_list(match.group("candidates"))
        decision_idx = int(match.group("decision"))
        if 0 <= decision_idx < len(candidates):
            variable_values[var_name] = int(candidates[decision_idx])

    return variable_values


def _parse_innermost_factor(
    factors_text: str,
    variable_values: Dict[str, int],
) -> Optional[int]:
    parts = [part.strip() for part in factors_text.split(",") if part.strip()]
    if not parts:
        return None
    return _resolve_value(parts[-1], variable_values)


def _extract_trace_labels(trace: str) -> Dict[str, Optional[int]]:
    if not trace:
        return {
            "vector_width": None,
            "unroll_factor": None,
            "cache_write_count": 0,
            "cache_write_used": 0,
            "decompose_reduction_count": 0,
            "reduction_decompose_used": 0,
            "innermost_j_tile": None,
        }

    loop_factor_map: Dict[str, Optional[int]] = {}
    fused_loop_map: Dict[str, List[str]] = {}
    loop_order_by_block: Dict[str, List[str]] = {}
    variable_values = _extract_variable_values(trace)

    for match in _GET_LOOPS_RE.finditer(trace):
        loops = [item.strip() for item in match.group("loops").split(",")]
        block = match.group("block")
        loop_order_by_block[block] = loops

    for match in _FUSE_RE.finditer(trace):
        fused_loop_map[match.group("out")] = [
            item.strip() for item in match.group("inputs").split(",") if item.strip()
        ]

    for match in _SPLIT_RE.finditer(trace):
        out_loops = [item.strip() for item in match.group("outs").split(",")]
        innermost = _parse_innermost_factor(match.group("factors"), variable_values)
        if out_loops:
            loop_factor_map[out_loops[-1]] = innermost

    innermost_j_tile: Optional[int] = None
    for block_name, loops in loop_order_by_block.items():
        if len(loops) < 2:
            continue
        j_loop = loops[1]
        split_match = re.search(
            rf"l\d+(?:\s*,\s*l\d+)*\s*=\s*sch\.split\(loop={j_loop},\s*factors=\[(?P<factors>[^\]]+)\]",
            trace,
        )
        if split_match:
            innermost_j_tile = _parse_innermost_factor(
                split_match.group("factors"),
                variable_values,
            )
            if innermost_j_tile is not None:
                break
        if block_name == "b0":
            break

    vector_width: Optional[int] = None
    for match in _VECTORIZE_RE.finditer(trace):
        vector_loop = match.group("loop")
        factor = loop_factor_map.get(vector_loop)
        if factor is None and vector_loop in fused_loop_map:
            source_loops = fused_loop_map[vector_loop]
            known_factors = [loop_factor_map.get(item) for item in source_loops]
            if known_factors and all(item is not None for item in known_factors):
                if len(known_factors) == 1:
                    factor = int(known_factors[0])
                else:
                    factor = int(math.prod(int(item) for item in known_factors if item is not None))
        if factor is not None:
            vector_width = _nearest_allowed(int(factor), ALLOWED_VECTOR_WIDTHS)
            break

    if vector_width is None:
        meta_vector_match = _META_VECTORIZE_RE.search(trace)
        if meta_vector_match:
            meta_vector_val = _resolve_value(meta_vector_match.group("value"), variable_values)
            vector_width = _nearest_allowed(meta_vector_val, ALLOWED_VECTOR_WIDTHS)

    if vector_width is None and innermost_j_tile is not None:
        vector_width = _nearest_allowed(innermost_j_tile, ALLOWED_VECTOR_WIDTHS)

    unroll_factor: Optional[int] = None
    pragma_unroll_match = _PRAGMA_UNROLL_RE.search(trace)
    if pragma_unroll_match:
        unroll_factor = _resolve_value(pragma_unroll_match.group("value"), variable_values)
    if unroll_factor is None:
        meta_unroll_match = _META_UNROLL_RE.search(trace)
        if meta_unroll_match:
            unroll_factor = _resolve_value(meta_unroll_match.group("value"), variable_values)

    cache_write_count = len(re.findall(r"sch\.cache_write\(", trace))
    decompose_reduction_count = len(re.findall(r"sch\.decompose_reduction\(", trace))

    return {
        "vector_width": vector_width,
        "unroll_factor": unroll_factor,
        "cache_write_count": cache_write_count,
        "cache_write_used": int(cache_write_count > 0),
        "decompose_reduction_count": decompose_reduction_count,
        "reduction_decompose_used": int(decompose_reduction_count > 0),
        "innermost_j_tile": innermost_j_tile,
    }


def extract_rows(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in records:
        kernel_name = _extract_kernel_name(record)
        m_val = _as_int(record.get("M"))
        k_val = _as_int(record.get("K"))
        n_val = _as_int(record.get("N"))

        denominator = (m_val * k_val) + (k_val * n_val) + (m_val * n_val)
        arithmetic_intensity = _safe_ratio(m_val * k_val * n_val, denominator)

        row: Dict[str, Any] = {
            "kernel_type": kernel_name,
            "M": m_val,
            "K": k_val,
            "N": n_val,
            "M_div_8": _safe_ratio(m_val, 8),
            "N_div_8": _safe_ratio(n_val, 8),
            "K_div_8": _safe_ratio(k_val, 8),
            "M_div_16": _safe_ratio(m_val, 16),
            "N_div_16": _safe_ratio(n_val, 16),
            "arithmetic_intensity_proxy": arithmetic_intensity,
            "reduction_ratio": _safe_ratio(k_val, max(n_val, 1)),
            "output_size": m_val * n_val,
            "flops": 2 * m_val * k_val * n_val,
            "latency_us": float(record.get("latency_us", math.nan)),
            "std_us": float(record.get("std_us", math.nan)),
        }

        trace = str(record.get("trace", ""))
        row.update(_extract_trace_labels(trace))
        rows.append(row)

    return rows


def build_training_dataframe(best_schedules_path: Path) -> pd.DataFrame:
    with best_schedules_path.open("r", encoding="utf-8") as file_in:
        payload = json.load(file_in)

    if not isinstance(payload, list):
        raise ValueError(f"Expected list JSON format in {best_schedules_path}")

    rows = extract_rows(item for item in payload if isinstance(item, dict))
    dataframe = pd.DataFrame(rows)
    return dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ML training data from best schedules")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to best_schedules.json (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    input_path = args.input
    output_path = args.output

    if not input_path.exists():
        LOGGER.error("Input file not found: %s", input_path)
        return 1

    LOGGER.info("Loading schedules from %s", input_path)
    dataframe = build_training_dataframe(input_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False)

    LOGGER.info("Extracted %d rows", len(dataframe))
    if not dataframe.empty:
        LOGGER.info(
            "Label coverage: vector_width=%d/%d, unroll_factor=%d/%d",
            int(dataframe["vector_width"].notna().sum()),
            len(dataframe),
            int(dataframe["unroll_factor"].notna().sum()),
            len(dataframe),
        )
    LOGGER.info("Saved training dataset to %s", output_path)
    LOGGER.info("Columns: %s", list(dataframe.columns))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
