import os
import re
import json
import time

# This file has been archived. Kept here for reference in case you want
# to fall back to log-parsing instead of using the database-backed results.

# Original location: research/workloads/bert/metaschedule/metaschedule_log_parse.py

LOG_BASE_DIR = "research/results/metaschedule"
RESULTS_FILE = "research/results/bert_matmul_results.json"

LATENCY_PATTERN = re.compile(r"Total latency \(us\): \s*([0-9.]+)")

from collections import defaultdict
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
                    kernel_match = re.search(r"metaschedule/(qkv|mlp_expand|mlp_reduce)", root)
                    m_match = re.search(r"M_(\d+)", root)
                    kernel = kernel_match.group(1) if kernel_match else None
                    m_val = int(m_match.group(1)) if m_match else None
                    latencies[kernel][m_val].append(val)

print("Archived metaschedule_log_parse.py — use database-backed results instead.")
