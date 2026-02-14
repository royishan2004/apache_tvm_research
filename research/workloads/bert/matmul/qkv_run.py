import sys
import time
import json
import os
import numpy as np
import tvm
from tvm.runtime import device

from research.workloads.bert.bert_shapes import (
    HIDDEN, DEFAULT_M, qkv_shape, print_config,
)

VARIANTS = {
    "baseline": "baseline",

    # K-dimension unrolling
    "k16": "research.workloads.bert.matmul.qkv_matmul_sched_k16",
    "k32": "research.workloads.bert.matmul.qkv_matmul_sched_k32",
    "k64": "research.workloads.bert.matmul.qkv_matmul_sched_k64",

    # Parallelism / vectorization
    "parallel": "research.workloads.bert.matmul.qkv_matmul_sched_parallel",
    "vec_j": "research.workloads.bert.matmul.qkv_matmul_sched_vec_j",
    "vec_k": "research.workloads.bert.matmul.qkv_matmul_sched_vec_k",

    # Mixed schedules
    "parallel_k16": "research.workloads.bert.matmul.qkv_matmul_sched_parallel_k16",
    "parallel_vec_j": "research.workloads.bert.matmul.qkv_matmul_sched_parallel_vec_j",
    "vec_j_k16": "research.workloads.bert.matmul.qkv_matmul_sched_vec_j_k16",
    "full": "research.workloads.bert.matmul.qkv_matmul_sched_full",
}

if len(sys.argv) < 2 or sys.argv[1] not in VARIANTS:
    print("Usage:")
    print("  python3 -m research.workloads.bert.matmul.qkv_run <variant> [--batch M]")
    print("\nAvailable variants:")
    for k in VARIANTS:
        print(" ", k)
    print(f"\n  --batch M   override batch dimension (default: {DEFAULT_M})")
    sys.exit(1)

variant = sys.argv[1]

# Optional --batch override
BATCH = DEFAULT_M
if "--batch" in sys.argv:
    idx = sys.argv.index("--batch")
    BATCH = int(sys.argv[idx + 1])

M, K, N = qkv_shape(BATCH)
TARGET = "llvm"

print(f"Running variant: {variant}")
print_config(BATCH)
print()


if variant == "baseline":
    from research.workloads.common.matmul_templates import matmul_tir
    mod = matmul_tir(M, K, N)
else:
    module_path = VARIANTS[variant]
    sched_mod = __import__(module_path, fromlist=["sch"])
    sch = sched_mod.sch
    mod = sch.mod


rt_mod = tvm.build(mod, target=TARGET)


dev = device("cpu", 0)

A_np = np.random.rand(M, K).astype("float32")
B_np = np.random.rand(K, N).astype("float32")
C_np = np.zeros((M, N), dtype="float32")

A = tvm.nd.array(A_np, dev)
B_mat = tvm.nd.array(B_np, dev)
C = tvm.nd.array(C_np, dev)


f = rt_mod["main"]

# Warm-up
for _ in range(5):
    f(A, B_mat, C)

# Timed runs
n = 10
start = time.time()
for _ in range(n):
    f(A, B_mat, C)
end = time.time()

latency_us = (end - start) * 1e6 / n
print(f"Latency (us): {latency_us:.3f}")


RESULTS_DIR = "research/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "bert_qkv_results.json")

os.makedirs(RESULTS_DIR, exist_ok=True)

result_entry = {
    "variant": variant,
    "M": M,
    "K": K,
    "N": N,
    "latency_us": float(latency_us),
    "runs": n,
    "target": TARGET,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

results = []
if os.path.exists(RESULTS_FILE):
    try:
        with open(RESULTS_FILE, "r") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                results = [r for r in loaded if isinstance(r, dict)]
    except json.JSONDecodeError:
        print("⚠️ Corrupted results file — starting fresh")

results.append(result_entry)

with open(RESULTS_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"Result saved to {RESULTS_FILE}")
