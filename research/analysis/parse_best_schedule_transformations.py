"""Parse MetaSchedule best traces and summarize applied transformations.

This script reads the best schedules artifact, extracts transformation values from
trace lines (for example annotate values, tile decisions, split factors), prints
a table, and writes a JSON summary that is overwritten on each run.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

LOGGER = logging.getLogger("parse_best_schedule_transformations")

RESEARCH_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = RESEARCH_DIR / "results" / "metaschedule" / "best_schedules.json"
DEFAULT_OUTPUT_PATH = (
    RESEARCH_DIR / "results" / "metaschedule" / "best_schedule_transformations.json"
)
DEFAULT_UPLOAD_TIMEOUT = 120


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parse best MetaSchedule traces and print/write a transformation table"
        )
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Input best schedules JSON path (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--view",
        choices=["compact", "long", "wide"],
        default="compact",
        help=(
            "Table layout for terminal output: compact (default), long (one transformation per row), "
            "or wide (all transformation columns)."
        ),
    )
    parser.add_argument(
        "--max-transform-cols",
        type=int,
        default=8,
        help="Number of transformation columns to show in compact view (default: 8)",
    )
    parser.add_argument(
        "--pager",
        action="store_true",
        help="Show table in less -SR for horizontal scrolling when supported",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Do not upload parsed transformation rows to data_aggregator",
    )
    parser.add_argument(
        "--upload-url",
        type=str,
        default=None,
        help=(
            "Override DATA_AGGREGATOR_BEST_SCHEDULE_TRANSFORMATIONS_URL for upload"
        ),
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Override DATA_AGGREGATOR_PROFILE for upload",
    )
    parser.add_argument(
        "--upload-timeout",
        type=int,
        default=DEFAULT_UPLOAD_TIMEOUT,
        help="Upload timeout in seconds (default: 120)",
    )
    return parser.parse_args()


def _safe_literal(value: str) -> Any:
    try:
        return ast.literal_eval(value)
    except Exception:  # pylint: disable=broad-except
        return value


def _strip_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        return stripped[1:-1]
    return stripped


def _split_top_level_args(arg_string: str) -> List[str]:
    args: List[str] = []
    current: List[str] = []
    depth = 0
    in_quote = False
    quote_char = ""
    escape = False

    for char in arg_string:
        if in_quote:
            current.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote_char:
                in_quote = False
            continue

        if char in {'"', "'"}:
            in_quote = True
            quote_char = char
            current.append(char)
            continue

        if char in "([{" :
            depth += 1
            current.append(char)
            continue

        if char in ")]}":
            depth = max(depth - 1, 0)
            current.append(char)
            continue

        if char == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                args.append(token)
            current = []
            continue

        current.append(char)

    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _parse_kwargs(arg_string: str) -> Dict[str, str]:
    kwargs: Dict[str, str] = {}
    for token in _split_top_level_args(arg_string):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        kwargs[key.strip()] = value.strip()
    return kwargs


def _append_value(target: Dict[str, List[Any]], key: str, value: Any) -> None:
    values = target.setdefault(key, [])
    for existing in values:
        if existing == value:
            return
    values.append(value)


def _resolve_value(raw: str, symbol_table: Dict[str, Any]) -> Any:
    token = raw.strip()

    if token.startswith("[") and token.endswith("]"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_resolve_value(item, symbol_table) for item in _split_top_level_args(inner)]

    if token.startswith("(") and token.endswith(")"):
        inner = token[1:-1].strip()
        if not inner:
            return []
        return [_resolve_value(item, symbol_table) for item in _split_top_level_args(inner)]

    if token in symbol_table:
        return symbol_table[token]
    parsed = _safe_literal(token)
    if isinstance(parsed, str):
        return _strip_quotes(parsed)
    return parsed


def _extract_symbol_table(trace: str) -> Dict[str, Any]:
    symbol_table: Dict[str, Any] = {}
    assignment_pattern = re.compile(r"^\s*([^=]+?)\s*=\s*sch\.(\w+)\((.*)\)\s*$")

    for line in trace.splitlines():
        match = assignment_pattern.match(line)
        if not match:
            continue

        lhs, op_name, args_blob = match.groups()
        kwargs = _parse_kwargs(args_blob)

        if op_name == "sample_perfect_tile":
            decision_raw = kwargs.get("decision")
            if decision_raw is None:
                continue
            decision = _safe_literal(decision_raw)
            if not isinstance(decision, list):
                continue
            var_names = [name.strip() for name in lhs.split(",") if name.strip()]
            for idx, var_name in enumerate(var_names):
                if idx < len(decision):
                    symbol_table[var_name] = decision[idx]

        elif op_name == "sample_categorical":
            var_names = [name.strip() for name in lhs.split(",") if name.strip()]
            if len(var_names) != 1:
                continue
            decision_raw = kwargs.get("decision")
            candidates_raw = kwargs.get("candidates")
            if decision_raw is None:
                continue

            decision = _safe_literal(decision_raw)
            selected_value: Any = decision
            if candidates_raw is not None:
                candidates = _safe_literal(candidates_raw)
                if isinstance(candidates, list):
                    try:
                        selected_value = candidates[int(decision)]
                    except Exception:  # pylint: disable=broad-except
                        selected_value = decision
            symbol_table[var_names[0]] = selected_value

    return symbol_table


def _collect_transformations(trace: str) -> Dict[str, Any]:
    symbol_table = _extract_symbol_table(trace)
    collected: Dict[str, List[Any]] = {}

    call_pattern = re.compile(r"^\s*(?:[^=]+?=\s*)?sch\.(\w+)\((.*)\)\s*$")
    for line in trace.splitlines():
        match = call_pattern.match(line)
        if not match:
            continue

        op_name, args_blob = match.groups()
        kwargs = _parse_kwargs(args_blob)

        if op_name == "annotate":
            ann_key = _strip_quotes(kwargs.get("ann_key", "annotate"))
            ann_val_raw = kwargs.get("ann_val", "")
            ann_val = _resolve_value(ann_val_raw, symbol_table)
            _append_value(collected, f"annotate.{ann_key}", ann_val)
            continue

        if op_name == "sample_perfect_tile":
            loop_id = kwargs.get("loop", "loop")
            decision_raw = kwargs.get("decision")
            if decision_raw is not None:
                decision = _resolve_value(decision_raw, symbol_table)
                _append_value(collected, f"sample_perfect_tile.{loop_id}", decision)
            continue

        if op_name == "sample_categorical":
            decision_raw = kwargs.get("decision")
            candidates_raw = kwargs.get("candidates")
            if decision_raw is not None:
                decision = _resolve_value(decision_raw, symbol_table)
                _append_value(collected, "sample_categorical.decision", decision)
            if candidates_raw is not None:
                candidates = _resolve_value(candidates_raw, symbol_table)
                _append_value(collected, "sample_categorical.candidates", candidates)
            continue

        if op_name == "split":
            loop_id = kwargs.get("loop", "loop")
            factors_raw = kwargs.get("factors")
            if factors_raw is not None:
                factors = _resolve_value(factors_raw, symbol_table)
                _append_value(collected, f"split.{loop_id}.factors", factors)
            continue

        if op_name == "cache_write":
            storage_scope_raw = kwargs.get("storage_scope")
            if storage_scope_raw is not None:
                storage_scope = _resolve_value(storage_scope_raw, symbol_table)
                _append_value(collected, "cache_write.storage_scope", storage_scope)
            continue

        if op_name == "reverse_compute_at":
            index_raw = kwargs.get("index")
            if index_raw is not None:
                index_val = _resolve_value(index_raw, symbol_table)
                _append_value(collected, "reverse_compute_at.index", index_val)
            continue

        if op_name in {"parallel", "vectorize", "decompose_reduction"}:
            loop_raw = kwargs.get("loop")
            if loop_raw is not None:
                _append_value(collected, f"{op_name}.loop", _resolve_value(loop_raw, symbol_table))
            continue

        if op_name == "fuse":
            loops = [token.strip() for token in _split_top_level_args(args_blob) if "=" not in token]
            if loops:
                _append_value(collected, "fuse.loops", loops)
            continue

    flattened: Dict[str, Any] = {}
    for key, values in collected.items():
        if not values:
            continue
        if len(values) == 1:
            flattened[key] = values[0]
        else:
            flattened[key] = values
    return flattened


def _to_printable(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"), ensure_ascii=True)
    return str(value)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _sanitize_json_value(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value

    if isinstance(value, dict):
        return {k: _sanitize_json_value(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_sanitize_json_value(item) for item in value]

    if isinstance(value, tuple):
        return [_sanitize_json_value(item) for item in value]

    return value


def _upload_rows(
    rows: List[Dict[str, Any]],
    profile: Optional[str],
    upload_url: Optional[str],
    timeout: Optional[int],
) -> bool:
    try:
        from research.workloads.common.data_aggregator_client import (  # pylint: disable=import-outside-toplevel
            upload_best_schedule_transformations,
        )
    except ModuleNotFoundError:
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        from research.workloads.common.data_aggregator_client import (  # type: ignore  # pylint: disable=import-outside-toplevel
            upload_best_schedule_transformations,
        )

    return upload_best_schedule_transformations(
        rows,
        url=upload_url,
        profile=profile,
        dedupe=True,
        timeout=timeout,
    )


def _format_table(dataframe: pd.DataFrame) -> str:
    try:
        from tabulate import tabulate  # pylint: disable=import-outside-toplevel

        return tabulate(dataframe, headers="keys", tablefmt="github", showindex=False)
    except Exception:  # pylint: disable=broad-except
        return dataframe.to_string(index=False)


def _select_compact_transform_cols(
    dataframe: pd.DataFrame,
    transform_cols: List[str],
    max_cols: int,
) -> List[str]:
    if max_cols <= 0:
        return []

    counts: List[Tuple[str, int]] = []
    for col in transform_cols:
        counts.append((col, int(dataframe[col].notna().sum())))

    counts.sort(key=lambda item: (-item[1], item[0]))
    return [name for name, _ in counts[:max_cols]]


def _build_output_dataframe(
    dataframe: pd.DataFrame,
    transform_cols: List[str],
    view: str,
    max_transform_cols: int,
) -> pd.DataFrame:
    base_cols = ["profile", "kernel", "M", "K", "N"]

    if view == "wide":
        return dataframe.copy()

    if view == "compact":
        selected = _select_compact_transform_cols(dataframe, transform_cols, max_transform_cols)
        remaining = [col for col in transform_cols if col not in selected]
        compact = dataframe[base_cols + selected].copy()
        if remaining:
            compact["other_transformations"] = dataframe[remaining].notna().sum(axis=1)
        return compact

    # view == "long"
    long_rows: List[Dict[str, Any]] = []
    for _, row in dataframe.iterrows():
        base = {
            "profile": row.get("profile"),
            "kernel": row.get("kernel"),
            "M": row.get("M"),
            "K": row.get("K"),
            "N": row.get("N"),
        }
        for col in transform_cols:
            value = row.get(col)
            if _is_missing(value):
                continue
            long_rows.append(
                {
                    **base,
                    "transformation": col,
                    "value": _to_printable(value),
                }
            )

    if not long_rows:
        return pd.DataFrame(columns=base_cols + ["transformation", "value"])
    return pd.DataFrame(long_rows)


def _print_with_optional_pager(text: str, use_pager: bool) -> None:
    if not use_pager:
        print(text)
        return

    less_path = shutil.which("less")
    if less_path and sys.stdout.isatty():
        try:
            subprocess.run([less_path, "-SR"], input=text, text=True, check=False)
            return
        except Exception:  # pylint: disable=broad-except
            pass

    print(text)


def _load_best_schedules(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as file_in:
        payload = json.load(file_in)
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {input_path}, got {type(payload).__name__}")
    return payload


def _build_rows(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    all_transform_cols: set[str] = set()

    for record in records:
        trace = str(record.get("trace", ""))
        transformations = _collect_transformations(trace)

        row: Dict[str, Any] = {
            "profile": record.get("profile", "unknown"),
            "kernel": record.get("kernel", "unknown"),
            "M": int(record.get("M", 0)),
            "K": int(record.get("K", 0)),
            "N": int(record.get("N", 0)),
        }

        for key, value in transformations.items():
            row[key] = value
            all_transform_cols.add(key)

        rows.append(row)

    ordered_transform_cols = sorted(all_transform_cols)
    return rows, ordered_transform_cols


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.input_json.exists():
        raise SystemExit(f"Input file does not exist: {args.input_json}")

    records = _load_best_schedules(args.input_json)
    rows, transform_cols = _build_rows(records)

    base_cols = ["profile", "kernel", "M", "K", "N"]
    ordered_cols = base_cols + transform_cols

    df = pd.DataFrame(rows)
    df = df.reindex(columns=ordered_cols)

    printable_df = df.copy()
    for col in transform_cols:
        printable_df[col] = printable_df[col].apply(
            lambda value: "" if _is_missing(value) else _to_printable(value)
        )

    output_df = _build_output_dataframe(
        dataframe=printable_df,
        transform_cols=transform_cols,
        view=args.view,
        max_transform_cols=args.max_transform_cols,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    # Overwrite output on every run to keep the artifact deterministic.
    raw_records = df.to_dict(orient="records")
    output_records = _sanitize_json_value(raw_records)
    if not isinstance(output_records, list):
        raise ValueError("Transformation output must be a JSON array")

    with args.output_json.open("w", encoding="utf-8") as file_out:
        json.dump(output_records, file_out, indent=2, allow_nan=False)

    LOGGER.info("Parsed %d schedules", len(df))
    LOGGER.info("Found %d transformation columns", len(transform_cols))
    LOGGER.info("Using '%s' view for terminal output", args.view)
    LOGGER.info("Saved transformation summary JSON to %s", args.output_json)

    if args.no_upload:
        LOGGER.info("Skipping cloud upload (--no-upload set)")
    else:
        upload_df = _build_output_dataframe(
            dataframe=printable_df,
            transform_cols=transform_cols,
            view="wide",
            max_transform_cols=args.max_transform_cols,
        )
        if args.view != "wide":
            LOGGER.info(
                "Uploading wide-view rows to keep DB schema aligned with --view wide table output"
            )

        raw_upload_records = upload_df.to_dict(orient="records")
        upload_records = _sanitize_json_value(raw_upload_records)
        if not isinstance(upload_records, list):
            raise ValueError("Upload payload must be a JSON array")

        upload_ok = _upload_rows(
            rows=upload_records,
            profile=args.profile,
            upload_url=args.upload_url,
            timeout=args.upload_timeout,
        )
        if upload_ok:
            LOGGER.info(
                "Uploaded %d transformation rows to data_aggregator best_schedule_transformations",
                len(upload_records),
            )
        else:
            LOGGER.warning(
                "Transformation upload failed; JSON file is still saved at %s",
                args.output_json,
            )

    _print_with_optional_pager(_format_table(output_df), use_pager=args.pager)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
