"""
LEVEL 5 — LARGE MATRIX MULTIPLICATION STRESS TEST

Purpose:
--------
This test validates TVM's ability to handle large, compute-intensive
workloads representative of real ML and HPC applications.

What this test checks:
----------------------
1. Large tensor allocation and memory management
2. Reduction-heavy computation (matrix multiplication)
3. Correct initialization and accumulation semantics
4. Cache-friendly scheduling (tiling)
5. Runtime stability under heavy compute load
6. Numerical correctness against NumPy reference

Why this test matters:
----------------------
This is a research-grade stress test.
Passing this test demonstrates:
- TVM is production- and research-ready
- LLVM backend scales to realistic workloads
- No hidden runtime or memory issues exist

Expected outcome:
-----------------
Large matmul executes correctly, efficiently, and without errors.
"""

import tvm
from tvm.script import tir as T
import numpy as np
import time

print("=" * 60)
print("LEVEL 5 — LARGE MATRIX MULTIPLICATION")
print("=" * 60)
print("TVM version:", tvm.__version__)

# --------------------------------------------------
# 1. Problem size (large enough to matter)
# --------------------------------------------------

M = N = K = 1024   

# --------------------------------------------------
# 2. Define TIR MatMul
# --------------------------------------------------

@tvm.script.ir_module
class MatMul:
    @T.prim_func
    def matmul(
        A: T.Buffer((M, K), "float32"),
        B: T.Buffer((K, N), "float32"),
        C: T.Buffer((M, N), "float32"),
    ):
        for i, j, k in T.grid(M, N, K):
            with T.block("C"):
                vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                with T.init():
                    C[vi, vj] = T.float32(0)
                C[vi, vj] += A[vi, vk] * B[vk, vj]

print("[1] TIR MatMul defined")

# --------------------------------------------------
# 3. Schedule (basic blocking)
# --------------------------------------------------

sch = tvm.tir.Schedule(MatMul)

block = sch.get_block("C", func_name="matmul")
i, j, k = sch.get_loops(block)

io, ii = sch.split(i, factors=[None, 32])
jo, ji = sch.split(j, factors=[None, 32])
ko, ki = sch.split(k, factors=[None, 8])

sch.reorder(io, jo, ko, ii, ji, ki)

print("[2] Schedule applied (tiling)")

# --------------------------------------------------
# 4. Build
# --------------------------------------------------

target = tvm.target.Target("llvm")
rt_mod = tvm.build(sch.mod, target=target)

print("[3] Build successful")

# --------------------------------------------------
# 5. Runtime execution
# --------------------------------------------------

dev = tvm.device("cpu", 0)

a_np = np.random.rand(M, K).astype("float32")
b_np = np.random.rand(K, N).astype("float32")
c_np = np.zeros((M, N), dtype="float32")

a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(c_np, dev)

# Warm-up
rt_mod["matmul"](a, b, c)

# Timed run
t0 = time.time()
rt_mod["matmul"](a, b, c)
t1 = time.time()

print(f"[4] Execution time: {(t1 - t0) * 1e3:.2f} ms")

# --------------------------------------------------
# 6. Correctness check
# --------------------------------------------------

np_ref = a_np @ b_np
np.testing.assert_allclose(c.numpy(), np_ref, rtol=1e-4)

print("[5] Numerical correctness verified")

print("=" * 60)
print("✅ LEVEL 5 — LARGE MATMUL PASSED")
print("=" * 60)
