#IMPORTANT GROUND KERNEL FOR MATRIX MUL

import tvm
from tvm.script import tir as T

def matmul_tir(M, K, N, dtype="float32"):
    @tvm.script.ir_module
    class MatMul:
        @T.prim_func
        def main(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((K, N), dtype),
            C: T.Buffer((M, N), dtype),
        ):
            for i, j, k in T.grid(M, N, K):
                with T.block("C"):
                    vi = T.axis.spatial(M, i)
                    vj = T.axis.spatial(N, j)
                    vk = T.axis.reduce(K, k)
                    with T.init():
                        C[vi, vj] = T.cast(0, dtype)
                    C[vi, vj] += A[vi, vk] * B[vk, vj]

    return MatMul
