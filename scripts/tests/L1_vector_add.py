"""
LEVEL 1 — BASIC END-TO-END EXECUTION (VECTOR ADD)

Purpose:
--------
This test validates a minimal end-to-end TVM workflow:
TIR definition → build → runtime execution → correctness.

What this test checks:
----------------------
1. TIR primitive function definition using tvm.script
2. Successful lowering and LLVM code generation
3. Runtime execution through the TVM runtime
4. Correct NDArray handling and memory access
5. Numerical correctness of the output

Why this test matters:
----------------------
This is the first executable kernel test.
It confirms that TVM is not only buildable but runnable.

This test isolates correctness without involving:
- Scheduling complexity
- MetaSchedule
- Performance optimizations

Expected outcome:
-----------------
Vector addition executes correctly and matches NumPy output.
"""

import tvm
from tvm.script import tir as T
import numpy as np

print("TVM version:", tvm.__version__)

@tvm.script.ir_module
class VectorAdd:
    @T.prim_func
    def add(
        A: T.Buffer((1024,), "float32"),
        B: T.Buffer((1024,), "float32"),
        C: T.Buffer((1024,), "float32"),
    ):
        for i in range(1024):
            C[i] = A[i] + B[i]

# Build
rt = tvm.build(VectorAdd, target="llvm")
print("Build OK")

# Runtime test
a_np = np.random.rand(1024).astype("float32")
b_np = np.random.rand(1024).astype("float32")
c_np = np.zeros(1024, dtype="float32")

dev = tvm.device("cpu")
a = tvm.nd.array(a_np, dev)
b = tvm.nd.array(b_np, dev)
c = tvm.nd.array(c_np, dev)

rt["add"](a, b, c)

np.testing.assert_allclose(c.numpy(), a_np + b_np)
print("VECTOR ADD PASSED ✅")
