from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.bert.bert_shapes import qkv_shape, M_LIST, HIDDEN

# Shape derived from BERT extraction — use first M from sweep list
# (this module is imported by the metaschedule tuner which needs a
# single concrete mod; tuning is done at M_LIST[0] then benchmarked
# across all M values by the runner).
M, K, N = qkv_shape(M_LIST[0])

mod = matmul_tir(M, K, N)

if __name__ == "__main__":
    print(mod.script())
