import os
import re
import json
import time

# --------------------------------------------------
# Paths
# --------------------------------------------------

# Base directory where per-kernel/per-M work dirs are created by the tuner.
LOG_BASE_DIR = "research/results/metaschedule"
RESULTS_FILE = "research/results/bert_matmul_results.json"

# --------------------------------------------------
# Regex to capture latency values
# --------------------------------------------------

LATENCY_PATTERN = re.compile(r"Total latency \(us\): \s*([0-9.]+)")

from collections import defaultdict
# Nested mapping: kernel_name -> M -> [latencies]
latencies = defaultdict(lambda: defaultdict(list))

for root, _, files in os.walk(LOG_BASE_DIR):
    for fname in files:
        if fname == "tvm.meta_schedule.logging.task_scheduler.log":
            with open(os.path.join(root, fname), "r") as f:
                for line in f:
                    match = LATENCY_PATTERN.search(line)
                    if not match:
                        continue
                    val = float(match.group(1))
                    if int(val) == 0:
                        continue
                    # infer kernel and M from path like .../metaschedule/<kernel>/M_<M>/logs/...
                    kernel_match = re.search(r"metaschedule/(qkv|mlp_expand|mlp_reduce)", root)
                    m_match = re.search(r"M_(\d+)", root)
                    kernel = kernel_match.group(1) if kernel_match else None
                    m_val = int(m_match.group(1)) if m_match else None
                    latencies[kernel][m_val].append(val)

if not any(any(vals for vals in m.values()) for m in latencies.values()):
    print("⚠️  No MetaSchedule latency values found under", LOG_BASE_DIR)

from research.workloads.bert.bert_shapes import (
    qkv_shape,
    mlp_expanded_shape,
    mlp_compressed_shape,
    M_LIST,
)

# Map kernel name -> shape helper
SHAPE_MAP = {
    "qkv": qkv_shape,
    "mlp_expand": mlp_expanded_shape,
    "mlp_reduce": mlp_compressed_shape,
}

# ─── build one entry per kernel and M value in the sweep list ─────
results = []
if os.path.exists(RESULTS_FILE):
    try:
        with open(RESULTS_FILE, "r") as f:
            results = json.load(f)
    except json.JSONDecodeError:
        results = []

# Remove old metaschedule entries (avoid duplicates)
results = [r for r in results if r.get("variant") != "metaschedule"]

missing = []
for kernel_name, shape_fn in SHAPE_MAP.items():
    for M_val in M_LIST:
        M, K, N = shape_fn(M_val)
        best_latency = None
        # Prefer exact M matches for this kernel; fallback to kernel=None logs
        if kernel_name in latencies and M in latencies[kernel_name] and latencies[kernel_name][M]:
            best_latency = min(latencies[kernel_name][M])
        elif kernel_name in latencies and None in latencies[kernel_name] and latencies[kernel_name][None]:
            best_latency = min(latencies[kernel_name][None])

        if best_latency is None:
            missing.append((kernel_name, M))

        results.append({
            "kernel": kernel_name,
            "variant": "metaschedule",
            "M": M,
            "K": K,
            "N": N,
            "latency_us": best_latency,
            "runs": "MetaSchedule",
            "target": "llvm",
            "source": "MetaSchedule-log",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

if missing:
    print(f"⚠️  No metaschedule logs found for (kernel,M) pairs: {missing}. Check {LOG_BASE_DIR}/*/M_*/logs/")

# For diagnostics, print simple summary of found bests
for kernel_name, mmap in latencies.items():
    for m, vals in mmap.items():
        if vals:
            print(f"Found {len(vals)} latency entries for kernel={kernel_name} M={m}: best={min(vals):.3f} us")

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"✔ MetaSchedule results ({len(M_LIST)} M values per kernel) saved to", RESULTS_FILE)
