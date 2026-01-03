import time
import numpy as np
import tvm
from tvm.runtime import device
from research.workloads.common.matmul_templates import matmul_tir

B = 128
H = 768

# Choose which kernel to test
USE_SCHEDULED = False

# -----------------------------
# Load kernel
# -----------------------------
if USE_SCHEDULED:
    from research.workloads.bert.matmul.qkv_matmul_sched_v1 import sch
    mod = sch.mod
else:
    mod = matmul_tir(B, H, H)

# -----------------------------
# Build
# -----------------------------
rt_mod = tvm.build(mod, target="llvm")

# -----------------------------
# Allocate data
# -----------------------------
dev = device("cpu", 0)

A_np = np.random.rand(B, H).astype("float32")
B_np = np.random.rand(H, H).astype("float32")
C_np = np.zeros((B, H), dtype="float32")

A = tvm.nd.array(A_np, dev)
B = tvm.nd.array(B_np, dev)
C = tvm.nd.array(C_np, dev)

# -----------------------------
# Run & time
# -----------------------------
f = rt_mod["main"]

# Warmup
for _ in range(5):
    f(A, B, C)

# Timing
n = 10
start = time.time()
for _ in range(n):
    f(A, B, C)
end = time.time()

latency = (end - start) * 1e6 / n
print("Latency (us):", latency)
