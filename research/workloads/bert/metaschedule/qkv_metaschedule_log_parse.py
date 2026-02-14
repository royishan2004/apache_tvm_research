import os
import re
import json
import time

# --------------------------------------------------
# Paths
# --------------------------------------------------

LOG_DIR = "research/results/metaschedule/qkv/logs"
RESULTS_FILE = "research/results/bert_matmul_results.json"

# --------------------------------------------------
# Regex to capture latency values
# --------------------------------------------------
LATENCY_PATTERN = re.compile(r"Total latency \(us\): \s*([0-9.]+)")

latencies = []

for root, _, files in os.walk(LOG_DIR):
    for fname in files:
        if fname == "tvm.meta_schedule.logging.task_scheduler.log":
            with open(os.path.join(root, fname), "r") as f:
                for line in f:
                    match = LATENCY_PATTERN.search(line)
                    if match:
                        if int(float(match.group(1))) != 0:
                            latencies.append(float(match.group(1)))

assert latencies, "No latency values found in MetaSchedule logs"

best_latency = min(latencies)

print(f"✔ Best MetaSchedule latency found: {best_latency:.3f} us")

from research.workloads.bert.bert_shapes import qkv_shape, M_LIST

# ─── build one entry per M value in the sweep list ─────────────────
results = []
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "r") as f:
        results = json.load(f)

# Remove old metaschedule entries (avoid duplicates)
results = [r for r in results if r.get("variant") != "metaschedule"]

for M_val in M_LIST:
    M, K, N = qkv_shape(M_val)
    results.append({
        "kernel": "qkv",
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

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"✔ MetaSchedule results ({len(M_LIST)} M values) saved to", RESULTS_FILE)
