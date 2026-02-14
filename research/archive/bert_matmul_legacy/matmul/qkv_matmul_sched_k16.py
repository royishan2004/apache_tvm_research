import tvm
from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.bert.bert_shapes import qkv_shape

M, K, N = qkv_shape()

mod = matmul_tir(M, K, N)
sch = tvm.tir.Schedule(mod)
sch.work_on("main")

block = sch.get_block("C")
i, j, k = sch.get_loops(block)

k0, k1 = sch.split(k, factors=[None, 16])
sch.reorder(i, j, k0, k1)
sch.vectorize(j)

if __name__ == "__main__":
    print(sch.mod)
