"""Rule-based CPU schedule for Transformer MatMul kernels.
Explanation of the rule-based system's derivation is present in the README.
https://github.com/royishan2004/apache_tvm_research/blob/main/README.md#empirical-trends--rule-based-schedule-derivation
"""

import tvm

# ─── Target micro-architecture constants (Intel i5-1235U) ─────────
_VEC_WIDTH = 8        # AVX2: 256 bit / 32-bit float = 8 lanes
_UNROLL_LIMIT = 16    # ki ≤ this → explicit unroll
_AUTO_UNROLL_STEP = 64  # pragma_auto_unroll_max_step for LLVM

def _select_tile_sizes(M, K, N, kernel="qkv"):
    """
    Return (TM, TN, TK) chosen by concise heuristics.
    """

    # F1: small reduction tile (TK = 8)
    TK = 8

    # F5: column tile (TN = 64 for large N)
    if N >= 512:
        TN = 64
    else:
        TN = max(_VEC_WIDTH, (N // _VEC_WIDTH) * _VEC_WIDTH)

    # F7: TM selection (use full M when small; prefer 64 if divisible)
    if M <= 32:
        TM = M                     # Full M — minimal outer-loop overhead
    elif M % 64 == 0:
        TM = 64                    # Clean division, larger tiles
    else:
        TM = 32                    # Fallback for non-64-aligned M (e.g. 96)

    # Safety clamp tiles to dims
    TM = min(TM, M)
    TN = min(TN, N)
    TK = min(TK, K)

    # Ensure TN >= vector width
    if TN < _VEC_WIDTH:
        TN = min(_VEC_WIDTH, N)

    return TM, TN, TK

def _should_unroll_k(TK):
    """Return True when reduction tile should be unrolled."""
    return TK <= _UNROLL_LIMIT

def apply_rule_based_schedule(mod, M, K, N, kernel="qkv"):
    """Apply concise rule-based schedule transforms to a MatMul TIR module.

    In-source comments are condensed; full rationale is in README.
    """

    TM, TN, TK = _select_tile_sizes(M, K, N, kernel)

    # Diagnostic — always printed so benchmark logs carry the decision
    print(f"    [rule_based] M={M} K={K} N={N}  kernel={kernel}  "
          f"→ TM={TM} TN={TN} TK={TK}  VEC={_VEC_WIDTH}  "
          f"unroll_k={'yes' if _should_unroll_k(TK) else 'no'}  "
          f"cache_write=yes  auto_unroll={_AUTO_UNROLL_STEP}")

    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)

    # Step 1: Tile axes (F1, F5, F7)
    i_outer, i_inner = sch.split(i, factors=[None, TM])
    j_outer, j_inner = sch.split(j, factors=[None, TN])
    k_outer, k_inner = sch.split(k, factors=[None, TK])

    # Step 2: Split j for SIMD lanes (F3)
    vec_width = min(_VEC_WIDTH, TN)
    j_inner_outer, j_vec = sch.split(j_inner, factors=[None, vec_width])

    # Step 3: Reorder loops for locality and parallelism
    sch.reorder(i_outer, j_outer, k_outer,
                i_inner, j_inner_outer, k_inner, j_vec)

    # Step 4: Cache-write local C tile (F6)
    C_write = sch.cache_write(block, 0, "global")
    sch.reverse_compute_at(C_write, j_outer)

    # Step 5: Fuse outer tiles for parallelism (F2, F4)
    fused = sch.fuse(i_outer, j_outer)
    sch.parallel(fused)

    # Step 6: Vectorise innermost j lane (F3)
    sch.vectorize(j_vec)

    # Step 7: Vectorise write-back loop
    write_loops = sch.get_loops(C_write)
    if write_loops:
        sch.vectorize(write_loops[-1])

    # Step 8: Unroll reduction inner loop (F1)
    if _should_unroll_k(TK):
        sch.unroll(k_inner)

    # Step 9: Apply LLVM auto-unroll pragmas (F6)
    sch.annotate(fused, "pragma_auto_unroll_max_step", _AUTO_UNROLL_STEP)
    sch.annotate(fused, "pragma_unroll_explicit", 1)

    # Step 10: Decompose reduction (F6)
    sch.decompose_reduction(block, k_outer)

    return sch.mod

def describe_tile_sizes(M, K, N, kernel="qkv"):
    """Return the tile-size dict the rule system would choose.
    """
    TM, TN, TK = _select_tile_sizes(M, K, N, kernel)
    par_tasks = (M // TM) * (N // TN) if TM <= M and TN <= N else "N/A"
    # With cache_write, C is in local buffer — L1 pressure is only A + B
    ws_ab_bytes = 4 * (TM * TK + TK * TN)           # A-strip + B-strip
    ws_c_bytes = 4 * (TM * TN)                       # C tile (local buffer)
    ws_total_bytes = ws_ab_bytes + ws_c_bytes         # total if C spills to L1
    return {
        "TM": TM, "TN": TN, "TK": TK,
        "VEC": _VEC_WIDTH,
        "unroll_k": _should_unroll_k(TK),
        "auto_unroll_step": _AUTO_UNROLL_STEP,
        "cache_write": True,
        "parallel_tasks": par_tasks,
        "working_set_ab_bytes": ws_ab_bytes,
        "working_set_c_local_bytes": ws_c_bytes,
        "working_set_total_bytes": ws_total_bytes,
        "working_set_pct_l1": round(100 * ws_ab_bytes / 32768, 1),  # E-core L1D
    }
