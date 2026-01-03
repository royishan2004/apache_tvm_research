from research.workloads.common.matmul_templates import matmul_tir

# Typical batch*sequence length - ASSUMPTION
M = 128        
H = 768

mod = matmul_tir(M, H, H)

if __name__ == "__main__":
    print(mod.script())
