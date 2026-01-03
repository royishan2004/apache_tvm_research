#This script Improves memory locality + enables SIMD

import tvm
from tvm.script import tir as T
from research.workloads.common.matmul_templates import matmul_tir

# BERT QKV dimensions
B = 128
H = 768

# 1. Get baseline module
mod = matmul_tir(B, H, H)

# 2. Create schedule
sch = tvm.tir.Schedule(mod)

# Explicitly select function
sch.work_on("main")

# 3. Get computation block
block = sch.get_block("C")

# 4. Get loops
i, j, k = sch.get_loops(block)

# 5. Tile reduction axis
k0, k1 = sch.split(k, factors=[None, 32])

# 6. Reorder loops (important)
sch.reorder(i, j, k0, k1)

# 7. Vectorize inner j loop
sch.vectorize(j)

print(sch.mod)
