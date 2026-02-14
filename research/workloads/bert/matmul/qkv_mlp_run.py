import sys
import time
import json
import os
import numpy as np
import tvm
from tvm.runtime import device

from research.workloads.bert.bert_shapes import (
    M_LIST,
    qkv_shape, mlp_expanded_shape, mlp_compressed_shape,
    print_config,
)
from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.common.schedule_recipes import (
    apply_schedule, available_variants,
)

KERNELS = {
    "qkv":        qkv_shape,
    "mlp_expand": mlp_expanded_shape,
    "mlp_reduce": mlp_compressed_shape,
}

VARIANT_LIST = available_variants()

def _parse_args():
    if len(sys.argv) < 2 or sys.argv[1] not in VARIANT_LIST:
        print("Usage:")
        print("  python3 -m research.workloads.bert.matmul.qkv_mlp_run <variant> [--kernel K]")
        print("\nAvailable variants:")
        for v in VARIANT_LIST:
            print(f"  {v}")
        print(f"\nOptions:")
        print(f"  --kernel K   one of: {', '.join(KERNELS)}  (default: qkv)")
        print(f"\nEvery run sweeps M_LIST = {M_LIST}")
        sys.exit(1)

    variant = sys.argv[1]

    # --kernel
    kernel = "qkv"
    if "--kernel" in sys.argv:
        idx = sys.argv.index("--kernel")
        kernel = sys.argv[idx + 1]
        if kernel not in KERNELS:
            print(f"Unknown kernel '{kernel}'. Choose from: {', '.join(KERNELS)}")
            sys.exit(1)

    return variant, kernel

variant, kernel = _parse_args()
shape_fn = KERNELS[kernel]
m_values = M_LIST
TARGET = "llvm"

print(f"Kernel : {kernel}")
print(f"Variant: {variant}")
print(f"M sweep: {m_values}")
print_config()
print()

RESULTS_DIR = "research/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "bert_matmul_results.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

results = []
if os.path.exists(RESULTS_FILE):
    try:
        with open(RESULTS_FILE, "r") as f_in:
            loaded = json.load(f_in)
            if isinstance(loaded, list):
                results = [r for r in loaded if isinstance(r, dict)]
    except json.JSONDecodeError:
        print("⚠️  Corrupted results file — starting fresh")

for M_val in m_values:
    M, K, N = shape_fn(M_val)
    base_mod = matmul_tir(M, K, N)
    scheduled_mod = apply_schedule(base_mod, variant)

    rt_mod = tvm.build(scheduled_mod, target=TARGET)
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
    print(f"  M={M:4d}  K={K:4d}  N={N:4d}  latency = {latency_us:.3f} us")

    results.append({
        "kernel":     kernel,
        "variant":    variant,
        "M": M,
        "K": K,
        "N": N,
        "latency_us": float(latency_us),
        "runs":       n,
        "target":     TARGET,
        "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
    })

with open(RESULTS_FILE, "w") as f_out:
    json.dump(results, f_out, indent=2)

print(f"\nResults saved to {RESULTS_FILE}")
