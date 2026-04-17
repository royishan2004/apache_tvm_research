"""
Persist and reload the best MetaSchedule traces for later analysis.

Storage format  (JSON list):
[
  {
    "kernel":      "qkv",
    "M": 16, "K": 768, "N": 768,
    "latency_us":  42.37,
    "trace":       "<human-readable trace string>",
    "decisions":   [ ... per-instruction decisions ... ]
  },
  ...
]

Each entry in ``decisions`` is a dict:
  { "instruction": "<inst kind>", "name": "<block/loop name>", "decision": <value(s)> }

This makes it straightforward to analyse tiling factors, vectorisation
widths, unroll depths, etc. across kernels and M values.
"""

import json
import os
import time
from typing import List, Optional

from research.workloads.common.data_aggregator_client import (
    upload_results,
    upload_best_schedules,
    resolve_profile,
)

SCHEDULES_FILE = "research/results/metaschedule/best_schedules.json"
RESULTS_FILE = "research/results/bert_matmul_results.json"


def _load_existing(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            text = f.read()
        # Try legacy JSON list first
        try:
            payload = json.loads(text)
            if isinstance(payload, list):
                return payload
        except json.JSONDecodeError:
            pass

        # Fallback to NDJSON (one JSON object per line)
        results = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    results.append(obj)
            except json.JSONDecodeError:
                # ignore malformed lines
                continue
        return results
    except Exception:
        return []


def _append_to_json_array(path: str, entry: dict) -> None:
    """Append one object to a JSON array file without full rewrite.

    Fast-path: in-place append by replacing the trailing `]`.
    Fallback: if the file is not a JSON array (e.g. NDJSON), load and
    rewrite once into JSON array format, then append.
    """
    entry_json = json.dumps(entry)

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w") as f:
            f.write("[\n")
            f.write(entry_json)
            f.write("\n]\n")
        return

    try:
        with open(path, "rb+") as f:
            payload = f.read()

            end = len(payload) - 1
            while end >= 0 and payload[end] in b" \t\r\n":
                end -= 1

            if end < 0 or payload[end] != ord("]"):
                raise ValueError("not a JSON array")

            prev = end - 1
            while prev >= 0 and payload[prev] in b" \t\r\n":
                prev -= 1
            is_empty = prev >= 0 and payload[prev] == ord("[")

            f.seek(end)
            f.truncate()

            separator = "\n" if is_empty else ",\n"
            suffix = (separator + entry_json + "\n]\n").encode("utf-8")
            f.write(suffix)
            return
    except Exception:
        # Compatibility fallback for older/invalid file formats (e.g. NDJSON).
        existing = _load_existing(path)
        existing.append(entry)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)


def _group_by_variant(records: List[dict]) -> List[dict]:
    """Group entries so records with the same variant stay together.

    Preserves first-seen variant order and in-variant insertion order.
    """
    buckets = {}
    variant_order = []

    for record in records:
        variant = record.get("variant") if isinstance(record, dict) else None
        key = str(variant) if variant is not None else ""
        if key not in buckets:
            buckets[key] = []
            variant_order.append(key)
        buckets[key].append(record)

    grouped = []
    for key in variant_order:
        grouped.extend(buckets[key])
    return grouped


def _extract_decisions(trace) -> List[dict]:
    """Walk every instruction in the trace and pull out its decision."""
    decisions = []
    for inst in trace.insts:
        decision = trace.get_decision(inst)
        entry = {
            "instruction": str(inst.kind),
            "attrs": [str(a) for a in inst.attrs] if inst.attrs else [],
        }
        if decision is not None:
            try:
                entry["decision"] = [int(d) for d in decision]
            except (TypeError, ValueError):
                entry["decision"] = str(decision)
        decisions.append(entry)
    return decisions


def save_best_schedule(
    kernel_name: str,
    M: int,
    K: int,
    N: int,
    best_record,
    latency_us: float,
    std_us: float = 0.0,
    profile: Optional[str] = None,
    variant: str = "metaschedule",
    runs_label: str = "MetaSchedule",
    source_label: str = "MetaSchedule-db",
    schedules_file: str = SCHEDULES_FILE,
    results_file: str = RESULTS_FILE,
) -> None:
    """Append the best schedule for a given kernel + M value."""
    profile = resolve_profile(profile)
    trace = best_record.trace

    schedule_entry = {
        "profile": profile,
        "variant": variant,
        "kernel": kernel_name,
        "M": M,
        "K": K,
        "N": N,
        "latency_us": latency_us,
        "std_us": std_us,
        "trace": str(trace),
        "decisions": _extract_decisions(trace),
    }

    os.makedirs(os.path.dirname(schedules_file), exist_ok=True)

    # Append to JSON array without rewriting the whole file in normal cases.
    try:
        _append_to_json_array(schedules_file, schedule_entry)
    except Exception as e:
        print(f"⚠ Failed to append to {schedules_file}: {e}")

    upload_best_schedules([schedule_entry], profile=profile)

    # Also update the global results summary file so we don't need to parse
    # logs separately. Append, group by variant, and rewrite with indent=2
    # to match qkv/mlp formatting and ordering conventions.
    try:
        result_entry = {
            "profile": profile,
            "kernel": kernel_name,
            "variant": variant,
            "M": M,
            "K": K,
            "N": N,
            "latency_us": latency_us,
            "std_us": std_us,
            "runs": runs_label,
            "target": "llvm",
            "source": source_label,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        results = [r for r in _load_existing(results_file) if isinstance(r, dict)]
        results.append(result_entry)
        grouped_results = _group_by_variant(results)
        with open(results_file, "w") as f:
            json.dump(grouped_results, f, indent=2)

        print(
            f"✔ Appended MetaSchedule summary entry for variant={variant} "
            f"kernel={kernel_name} M={M} to {results_file}"
        )

        upload_results([result_entry], profile=profile)
    except Exception as e:
        # Non-fatal: log error so user can debug why results file wasn't updated
        print(f"⚠ Failed to update {results_file}: {e}")
