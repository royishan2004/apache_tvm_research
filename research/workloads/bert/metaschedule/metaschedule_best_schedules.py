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
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        return []
    return payload


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
    """Append (or update) the best schedule for a given kernel + M value."""
    profile = resolve_profile(profile)
    trace = best_record.trace

    new_entry = {
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

    records = _load_existing(schedules_file)

    # Replace existing entry for this variant+kernel+M if present.
    records = [
        r for r in records
        if not (
            r.get("variant", "metaschedule") == variant
            and r.get("kernel") == kernel_name
            and r.get("M") == M
        )
    ]
    records.append(new_entry)

    # Sort for stable output
    records.sort(key=lambda r: (r.get("variant", "metaschedule"), r.get("kernel", ""), r.get("M", 0)))

    os.makedirs(os.path.dirname(schedules_file), exist_ok=True)
    with open(schedules_file, "w") as f:
        json.dump(records, f, indent=2)

    upload_best_schedules([new_entry], profile=profile)

    # Also update the global results summary file so we don't need to parse
    # logs separately. Replace any existing metaschedule entry for same
    # (kernel, M) and append a new one.
    try:
        results = []
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = []
        if not isinstance(results, list):
            results = []

        results = [
            r for r in results
            if not (
                r.get("variant") == variant
                and r.get("kernel") == kernel_name
                and r.get("M") == M
            )
        ]

        new_entry = {
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
        results.append(new_entry)

        # Keep other entries; sort for stability
        results.sort(key=lambda r: (r.get("kernel", ""), r.get("variant", ""), r.get("M", 0)))
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✔ Wrote MetaSchedule summary entry for variant={variant} kernel={kernel_name} M={M} to {results_file}")

        upload_results([new_entry], profile=profile)
    except Exception as e:
        # Non-fatal: log error so user can debug why results file wasn't updated
        print(f"⚠ Failed to update {results_file}: {e}")
