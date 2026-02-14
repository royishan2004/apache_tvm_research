from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.bert.bert_shapes import qkv_shape, DEFAULT_M, HIDDEN

# Shape derived from BERT extraction (Phase 2)
M, K, N = qkv_shape()

mod = matmul_tir(M, K, N)

if __name__ == "__main__":
    print(mod.script())
