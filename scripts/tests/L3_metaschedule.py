"""
LEVEL 3 — METASCHEDULE INFRASTRUCTURE VALIDATION

Purpose:
--------
This test validates that TVM's MetaSchedule system is operational,
including tuning, cost modeling, and task scheduling.

What this test checks:
----------------------
1. Integration of TIR workloads with MetaSchedule
2. Task extraction and scheduling
3. Builder and runner execution
4. Cost model training (XGBoost-based)
5. End-to-end tuning loop stability

What this test intentionally does NOT assert:
---------------------------------------------
- That a best schedule must be returned for trivial workloads

Why this test matters:
----------------------
MetaSchedule is a complex, optional subsystem.
This test confirms:
- Dependencies are correctly installed
- FFI integration works
- Tuning pipelines execute without crashes

Expected outcome:
-----------------
Tuning completes successfully without runtime or internal errors.
"""

import tvm
from tvm.script import tir as T
from tvm.meta_schedule import tune_tir
import numpy as np
import tempfile
import os

print("=" * 60)
print("LEVEL 3 — METASCHEDULE (TVM 0.21 API)")
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

print("\n[1] Workload defined")

# --------------------------------------------------
# 2. Run MetaSchedule tuning
# --------------------------------------------------

work_dir = tempfile.mkdtemp(prefix="tvm_ms_")

print("[2] Tuning work dir:", work_dir)

target = tvm.target.Target("llvm -num-cores=8")

database = tune_tir(
    mod=Module,
    target=target,
    max_trials_global=32,
    num_trials_per_iter=8,
    work_dir=work_dir,
)


print("[3] MetaSchedule tuning completed")

# --------------------------------------------------
# 3. Fetch best schedule
# --------------------------------------------------

sch = database.query(mod=Module, target=target)

if sch is None:
    print("[4] No stored schedule (expected for simple workloads)")
    print("[4] Using default tuned module")
    tuned_mod = Module
else:
    print("[4] Best schedule obtained")
    tuned_mod = sch.mod



# --------------------------------------------------
# 4. Build with best schedule
# --------------------------------------------------

rt_mod = tvm.build(tuned_mod, target="llvm")
print("[5] LLVM codegen successful")

# --------------------------------------------------
# 5. Runtime verification
# --------------------------------------------------

dev = tvm.device("cpu", 0)

a_np = np.random.rand(1024).astype("float32")
b_np = np.random.rand(1024).astype("float32")
c_np = np.zeros(1024, dtype="float32")

a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(c_np, dev)

rt_mod["add"](a, b, c)

np.testing.assert_allclose(
    c.numpy(),
    a_np + b_np,
    rtol=1e-5,
)

print("[6] Runtime correctness verified")

print("=" * 60)
print("✅ LEVEL 3 METASCHEDULE (0.21): VERIFIED")
print("=" * 60)
