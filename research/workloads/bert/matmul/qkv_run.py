import sys
import time
import numpy as np
import tvm
from tvm.runtime import device

# ---------------------------------------------
# Variant registry
# ---------------------------------------------
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
}

# ---------------------------------------------
# CLI handling
# ---------------------------------------------
if len(sys.argv) != 2 or sys.argv[1] not in VARIANTS:
    print("Usage:")
    print("  python3 -m research.workloads.bert.matmul.qkv_run <variant>")
    print("\nAvailable variants:")
    for k in VARIANTS:
        print(" ", k)
    sys.exit(1)

variant = sys.argv[1]
print(f"Running variant: {variant}")

# ---------------------------------------------
# Load module
# ---------------------------------------------
B, H = 128, 768

if variant == "baseline":
    from research.workloads.common.matmul_templates import matmul_tir
    mod = matmul_tir(B, H, H)
else:
    module_path = VARIANTS[variant]
    sched_mod = __import__(module_path, fromlist=["sch"])
    sch = sched_mod.sch
    mod = sch.mod

# ---------------------------------------------
# Build
# ---------------------------------------------
rt_mod = tvm.build(mod, target="llvm")

# ---------------------------------------------
# Allocate data
# ---------------------------------------------
dev = device("cpu", 0)

A_np = np.random.rand(B, H).astype("float32")
B_np = np.random.rand(H, H).astype("float32")
C_np = np.zeros((B, H), dtype="float32")

A = tvm.nd.array(A_np, dev)
B = tvm.nd.array(B_np, dev)
C = tvm.nd.array(C_np, dev)

# ---------------------------------------------
# Execute & time
# ---------------------------------------------
f = rt_mod["main"]

# Warm-up
for _ in range(5):
    f(A, B, C)

# Timed runs
n = 10
start = time.time()
for _ in range(n):
    f(A, B, C)
end = time.time()

latency_us = (end - start) * 1e6 / n
print(f"Latency (us): {latency_us:.3f}")
