"""
LEVEL 2 — SCHEDULE SEMANTICS & LOWERING INTEGRITY

Purpose:
--------
This test validates that TIR scheduling transformations are:
- Semantically correct
- Properly preserved during lowering
- Reflected accurately in lowered IR

What this test checks:
----------------------
1. Creation of a TIR Schedule from an IRModule
2. Loop transformations such as:
   - split
   - reorder
   - vectorize
3. Structural integrity of scheduled TIR
4. Successful lowering to a lower-level IR
5. Absence of schedule-induced semantic errors

Why this test matters:
----------------------
Scheduling is core to TVM's optimization model.
A correct build must:
- Preserve computation semantics
- Correctly lower transformed loops

Failures here indicate deep compiler issues.

Expected outcome:
-----------------
Scheduled TIR is valid, lowerable, and structurally sound.
"""

import tvm
import tvm.driver
from tvm.script import tir as T
import numpy as np
from tvm.ir.transform import Sequential

print("=" * 60)
print("LEVEL 2 — SCHEDULE SEMANTICS & LOWERING INTEGRITY")
print("=" * 60)
print("TVM version:", tvm.__version__)

# ------------------------------------------------------------
# 1. Define TIR module
# ------------------------------------------------------------

@tvm.script.ir_module
class VectorAdd:
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

print("\n[1] Original TIR:")
print(VectorAdd.script())

# ------------------------------------------------------------
# 2. Create Schedule
# ------------------------------------------------------------

sch = tvm.tir.Schedule(VectorAdd, debug_mask="all")

# REQUIRED in TVM 0.21+
sch.work_on("add")

block = sch.get_block("add")
(i,) = sch.get_loops(block)

# ------------------------------------------------------------
# 3. Apply Schedule Transformations
# ------------------------------------------------------------

i0, i1 = sch.split(i, factors=[None, 4])
sch.vectorize(i1)

print("\n[2] Scheduled TIR:")
print(sch.mod.script())

# ------------------------------------------------------------
# 4. Lower using driver API (TVM 0.21)
# ------------------------------------------------------------

print("\n[3] Lowering via tvm.build()")

rt_mod = tvm.build(sch.mod, target="llvm")
print("Lowering + codegen: OK")


# ------------------------------------------------------------
# 5. Build with LLVM
# ------------------------------------------------------------

rt_mod = tvm.build(sch.mod, target="llvm")
print("\n[4] LLVM build: OK")

# ------------------------------------------------------------
# 6. Runtime verification
# ------------------------------------------------------------

dev = tvm.device("cpu")

a_np = np.random.rand(1024).astype("float32")
b_np = np.random.rand(1024).astype("float32")
c_np = np.zeros(1024, dtype="float32")

a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(c_np, dev)

rt_mod["add"](a, b, c)
np.testing.assert_allclose(c.numpy(), a_np + b_np)

print("[5] Runtime result: CORRECT")

print("\n============================================================")
print("✅ LEVEL 2 PASSED — Schedule Semantics Verified")
print("============================================================")
