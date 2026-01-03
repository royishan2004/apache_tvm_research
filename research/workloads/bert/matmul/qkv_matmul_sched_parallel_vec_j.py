import tvm
from research.workloads.common.matmul_templates import matmul_tir

B = 128
H = 768

mod = matmul_tir(B, H, H)
sch = tvm.tir.Schedule(mod)
sch.work_on("main")

block = sch.get_block("C")
i, j, k = sch.get_loops(block)

# Split j for SIMD width
j0, j1 = sch.split(j, factors=[None, 8])

# Reorder
sch.reorder(i, j0, j1, k)

# Parallelize batch
sch.parallel(i)

# Vectorize contiguous axis
sch.vectorize(j1)

if __name__ == "__main__":
    print(sch.mod)
