import tvm
from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.bert.bert_shapes import qkv_shape

M, K, N = qkv_shape()

mod = matmul_tir(M, K, N)
sch = tvm.tir.Schedule(mod)
sch.work_on("main")

block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Split reduction axis
k0, k1 = sch.split(k, factors=[None, 16])

# Reorder for better locality
sch.reorder(i, j, k0, k1)

# Parallelize batch dimension
sch.parallel(i)

# Unroll inner reduction
sch.unroll(k1)

if __name__ == "__main__":
    print(sch.mod)
