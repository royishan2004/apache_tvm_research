import tvm
from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.bert.bert_shapes import qkv_shape

M, K, N = qkv_shape()

mod = matmul_tir(M, K, N)
sch = tvm.tir.Schedule(mod)
sch.work_on("main")

block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Decompose axes
j0, j1 = sch.split(j, factors=[None, 8])
k0, k1 = sch.split(k, factors=[None, 16])

# Reorder for full composition
sch.reorder(i, j0, k0, j1, k1)

# Apply all primitives
sch.parallel(i)
sch.vectorize(j1)
sch.unroll(k1)

if __name__ == "__main__":
    print(sch.mod)
