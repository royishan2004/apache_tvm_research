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

SCHEDULES_FILE = "research/results/metaschedule/best_schedules.json"
RESULTS_FILE = "research/results/bert_matmul_results.json"


def _load_existing() -> List[dict]:
    if not os.path.exists(SCHEDULES_FILE):
        return []
    with open(SCHEDULES_FILE, "r") as f:
        return json.load(f)


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
) -> None:
    """Append (or update) the best schedule for a given kernel + M value."""
    latency_us = float(sum(best_record.run_secs)) / len(best_record.run_secs) * 1e6
    trace = best_record.trace

    new_entry = {
        "kernel": kernel_name,
        "M": M,
        "K": K,
        "N": N,
        "latency_us": latency_us,
        "trace": str(trace),
        "decisions": _extract_decisions(trace),
    }

    records = _load_existing()

    # Replace existing entry for this kernel+M if present
    records = [
        r for r in records
        if not (r["kernel"] == kernel_name and r["M"] == M)
    ]
    records.append(new_entry)

    # Sort for stable output
    records.sort(key=lambda r: (r["kernel"], r["M"]))

    os.makedirs(os.path.dirname(SCHEDULES_FILE), exist_ok=True)
    with open(SCHEDULES_FILE, "w") as f:
        json.dump(records, f, indent=2)

    # Also update the global results summary file so we don't need to parse
    # logs separately. Replace any existing metaschedule entry for same
    # (kernel, M) and append a new one.
    try:
        results = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, "r") as f:
                try:
                    results = json.load(f)
                except json.JSONDecodeError:
                    results = []

        # remove old metaschedule entries for this kernel+M
        results = [
            r for r in results
            if not (r.get("variant") == "metaschedule" and r.get("kernel") == kernel_name and r.get("M") == M)
        ]

        results.append({
            "kernel": kernel_name,
            "variant": "metaschedule",
            "M": M,
            "K": K,
            "N": N,
            "latency_us": latency_us,
            "runs": "MetaSchedule",
            "target": "llvm",
            "source": "MetaSchedule-db",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        # Keep other entries; sort for stability
        results.sort(key=lambda r: (r.get("kernel", ""), r.get("M", 0)))
        os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)

        print(f"✔ Wrote MetaSchedule summary entry for {kernel_name} M={M} to {RESULTS_FILE}")
    except Exception as e:
        # Non-fatal: log error so user can debug why results file wasn't updated
        print(f"⚠ Failed to update {RESULTS_FILE}: {e}")
