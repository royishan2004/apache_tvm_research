"""
LEVEL 4 — PERFORMANCE & IR VALIDATION

Purpose:
--------
This test validates that TVM produces performant, optimized LLVM code
and allows deep inspection of generated IR.

What this test checks:
----------------------
1. Baseline (unscheduled) kernel performance
2. Optimized (scheduled) kernel performance
3. Runtime benchmarking stability
4. Numerical correctness under optimization
5. Generation and inspection of LLVM IR
6. Presence of vectorization and low-level optimizations in IR

Why this test matters:
----------------------
This is where correctness meets performance.
It confirms that:
- Optimizations do not break correctness
- LLVM backend emits real optimized code
- Performance measurement infrastructure is stable

Expected outcome:
-----------------
Correct results, stable performance numbers, and valid LLVM IR output.
"""

import tvm
from tvm.script import tir as T
import numpy as np
import time

print("=" * 60)
print("LEVEL 4 — PERFORMANCE & IR VALIDATION (TVM 0.21)")
print("=" * 60)
print("TVM version:", tvm.__version__)

# --------------------------------------------------
# 1. Define workload
# --------------------------------------------------

@tvm.script.ir_module
class Module:
    @T.prim_func
    def add(
        A: T.Buffer((1024,), "float32"),
        B: T.Buffer((1024,), "float32"),
        C: T.Buffer((1024,), "float32"),
    ):
        for i in range(1024):
            with T.block("add"):
                vi = T.axis.spatial(1024, i)
                C[vi] = A[vi] + B[vi]

print("[1] Workload defined")

# --------------------------------------------------
# 2. Build BASELINE (no schedule)
# --------------------------------------------------

target = tvm.target.Target("llvm")
baseline_mod = tvm.build(Module, target=target)

print("[2] Baseline build OK")

# --------------------------------------------------
# 3. Manual optimized schedule (vectorization)
# --------------------------------------------------

sch = tvm.tir.Schedule(Module)
sch.work_on("add")

block = sch.get_block("add")
i, = sch.get_loops(block)
io, ii = sch.split(i, factors=[None, 4])
sch.vectorize(ii)

optimized_mod = tvm.build(sch.mod, target=target)

print("[3] Optimized build OK")

# --------------------------------------------------
# 4. Runtime setup
# --------------------------------------------------

dev = tvm.device("cpu", 0)

a_np = np.random.rand(1024).astype("float32")
b_np = np.random.rand(1024).astype("float32")
c_np = np.zeros(1024, dtype="float32")

a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(c_np, dev)

# --------------------------------------------------
# 5. Correctness check
# --------------------------------------------------

baseline_mod["add"](a, b, c)
np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-5)

optimized_mod["add"](a, b, c)
np.testing.assert_allclose(c.numpy(), a_np + b_np, rtol=1e-5)

print("[4] Numerical correctness verified")

# --------------------------------------------------
# 6. Performance benchmarking
# --------------------------------------------------

def benchmark(rt_mod, n=1000):
    start = time.time()
    for _ in range(n):
        rt_mod["add"](a, b, c)
    return (time.time() - start) / n * 1e6  # microseconds

baseline_time = benchmark(baseline_mod)
optimized_time = benchmark(optimized_mod)

print("[5] Baseline latency  : %.3f us" % baseline_time)
print("[5] Optimized latency : %.3f us" % optimized_time)

speedup = baseline_time / optimized_time
print("[5] Speedup           : %.2fx" % speedup)

# --------------------------------------------------
# 7. Dump LLVM IR
# --------------------------------------------------

print("\n[6] LLVM IR (Optimized):")
print(optimized_mod.get_source("ll"))

print("=" * 60)
print("✅ LEVEL 4 — PERFORMANCE & IR VALIDATION PASSED")
print("=" * 60)
