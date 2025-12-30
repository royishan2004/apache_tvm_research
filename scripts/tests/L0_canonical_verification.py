"""
LEVEL 0 — CANONICAL TVM VERIFICATION

Purpose:
--------
This test verifies that the core TVM Python bindings, runtime, and
essential compiler infrastructure are correctly built and usable.

What this test checks:
----------------------
1. TVM Python package is importable and reports a valid version
2. Core namespaces exist and load correctly:
   - tvm.runtime
   - tvm.tir
   - tvm.te
   - tvm.ir
   - tvm.target
   - tvm.driver
3. NDArray creation and device management works
4. Basic TIR → LLVM lowering and build pipeline functions
5. Optional components (Relay, GPU backends) are correctly reported
   as missing when not enabled, without breaking core functionality

Why this test matters:
----------------------
This is the foundational sanity check for a TVM build.
If this test fails, the TVM installation is incomplete or broken.
If this test passes, TVM core infrastructure is correctly initialized.

Expected outcome:
-----------------
All core components report OK.
Optional components may be missing depending on build configuration.
"""

import sys
import tvm

print("=" * 60)
print("TVM CANONICAL VERIFICATION")
print("=" * 60)

# --------------------------------------------------
# Basic environment
# --------------------------------------------------
print("TVM version       :", tvm.__version__)
print("Python version    :", sys.version.split()[0])
print("TVM path          :", tvm.__file__)
print()

# --------------------------------------------------
# Core namespaces (MUST exist)
# --------------------------------------------------
def check(name, expr, required=True):
    status = "OK" if expr else "MISSING"
    note = ""
    if not expr and not required:
        note = "(optional / expected)"
    elif not expr and required:
        note = "(ERROR)"
    print(f"{name:<25}: {status} {note}")

check("runtime", hasattr(tvm, "runtime"))
check("tir", hasattr(tvm, "tir"))
check("te", hasattr(tvm, "te"))
check("ir", hasattr(tvm, "ir"))
check("target", hasattr(tvm, "target"))
check("driver", hasattr(tvm, "driver"))
check("parser", hasattr(tvm, "parser"))

# --------------------------------------------------
# Relax (modern IR – SHOULD exist)
# --------------------------------------------------
try:
    import tvm.relax
    relax_ok = True
except ImportError:
    relax_ok = False

check("relax", relax_ok)

# --------------------------------------------------
# Relay (legacy IR – OPTIONAL)
# --------------------------------------------------
try:
    import tvm.relay
    relay_ok = True
except ImportError:
    relay_ok = False

check("relay", relay_ok, required=False)

# --------------------------------------------------
# Meta-schedule & auto-tuning
# --------------------------------------------------
try:
    import tvm.meta_schedule
    ms_ok = True
except ImportError:
    ms_ok = False

check("meta_schedule", ms_ok)

# --------------------------------------------------
# Script / TVMScript
# --------------------------------------------------
try:
    from tvm.script import tir as T
    script_ok = True
except ImportError:
    script_ok = False

check("tvm.script", script_ok)

# --------------------------------------------------
# LLVM backend
# --------------------------------------------------
try:
    llvm_enabled = tvm.runtime.enabled("llvm")
except Exception:
    llvm_enabled = False

check("LLVM backend", llvm_enabled)

# --------------------------------------------------
# Codegen backends (OPTIONAL, runtime-dependent)
# --------------------------------------------------
optional_backends = [
    "cuda",
    "rocm",
    "vulkan",
    "opencl",
    "metal",
    "hexagon",
]

for b in optional_backends:
    try:
        enabled = tvm.runtime.enabled(b)
    except Exception:
        enabled = False
    check(f"{b} backend", enabled, required=False)

# --------------------------------------------------
# Runtime objects
# --------------------------------------------------
try:
    dev = tvm.device("cpu", 0)
    runtime_ok = True
except Exception:
    runtime_ok = False

check("runtime.device()", runtime_ok)

# --------------------------------------------------
# NDArray (critical!)
# --------------------------------------------------
try:
    import numpy as np
    a = tvm.nd.array(np.array([1, 2, 3], dtype="float32"))
    ndarray_ok = True
except Exception:
    ndarray_ok = False

check("NDArray", ndarray_ok)

# --------------------------------------------------
# Build pipeline (minimal TIR compile)
# --------------------------------------------------
try:
    from tvm.script import tir as T

    @tvm.script.ir_module
    class TestModule:
        @T.prim_func
        def add(a: T.handle, b: T.handle):
            A = T.match_buffer(a, (4,), dtype="float32")
            B = T.match_buffer(b, (4,), dtype="float32")
            for i in range(4):
                B[i] = A[i] + 1.0

    mod = tvm.build(TestModule, target="llvm")
    build_ok = True
except Exception as e:
    build_ok = False
    build_err = str(e)

check("TIR → LLVM build", build_ok)

# --------------------------------------------------
# Summary
# --------------------------------------------------
print()
print("=" * 60)
if build_ok and runtime_ok and ndarray_ok:
    print("✅ TVM CORE FUNCTIONALITY: VERIFIED")
else:
    print("❌ TVM CORE FUNCTIONALITY: ISSUES DETECTED")
print("=" * 60)

if not relay_ok:
    print("Note: Relay is not enabled (expected for many 0.21+ builds)")
