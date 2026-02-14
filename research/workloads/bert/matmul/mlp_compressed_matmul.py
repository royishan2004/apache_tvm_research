from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.bert.bert_shapes import mlp_compressed_shape

# Shape derived from BERT extraction (Phase 2)
M, K, N = mlp_compressed_shape()

mod = matmul_tir(M, K, N)

if __name__ == "__main__":
    print(mod.script())
