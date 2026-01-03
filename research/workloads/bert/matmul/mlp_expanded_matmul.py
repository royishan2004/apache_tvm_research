from research.workloads.common.matmul_templates import matmul_tir

# - ASSUMPTION
M = 128
H = 768
FF = 3072

mod = matmul_tir(M, H, FF)

if __name__ == "__main__":
    print(mod.script())
