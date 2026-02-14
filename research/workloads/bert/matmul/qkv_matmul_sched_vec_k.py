#NOTE: EXPECTED FAILURE CASE! - Reduction-axis vectorization is illegal unless the reduction is transformed first

import tvm
from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.bert.bert_shapes import qkv_shape

M, K, N = qkv_shape()

# --------------------------------------------------
# 1. Canonical MatMul
# --------------------------------------------------
mod = matmul_tir(M, K, N)
sch = tvm.tir.Schedule(mod)
sch.work_on("main")

# --------------------------------------------------
# 2. Get block and loops
# --------------------------------------------------
block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# --------------------------------------------------
# 3. Split reduction axis
# --------------------------------------------------
k0, k1 = sch.split(k, factors=[None, 8])

# Vectorize inner reduction tile
sch.vectorize(k1)

# --------------------------------------------------
# 4. Print scheduled TIR
# --------------------------------------------------
if __name__ == "__main__":
    print(sch.mod)
