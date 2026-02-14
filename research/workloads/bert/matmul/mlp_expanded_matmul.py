from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.bert.bert_shapes import mlp_expanded_shape, M_LIST

# Use first M from sweep list for a concrete module
# (the runner benchmarks across all M values)
M, K, N = mlp_expanded_shape(M_LIST[0])

mod = matmul_tir(M, K, N)

if __name__ == "__main__":
    print(mod.script())
