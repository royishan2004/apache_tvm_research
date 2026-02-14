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
# 3. Apply vectorization on output axis
# --------------------------------------------------
sch.vectorize(j)

# --------------------------------------------------
# 4. Print scheduled TIR
# --------------------------------------------------
if __name__ == "__main__":
    print(sch.mod)
