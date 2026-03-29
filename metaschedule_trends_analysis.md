# MetaSchedule Best Schedules Analysis

This document evaluates the decisions and generated Tensor Intermediate Representations (TIRs) from the `best_schedules.json` MetaSchedule traces, focusing on the `mlp_expand` benchmark kernel (M × 768 × 3072). 

## 1. Common Trends in Generated TIRs

### A. Tiling Structure (SSRSRS)
Every optimal trace uniformly applies a 4-level spatial split and a 2-level reduction split (`SSRSRS`). This provides significantly finer-grained control over register blocking and cache hierarchies than a traditional 2-level spatial tile. The hierarchy ensures that outermost loops handle multi-core parallelism while innermost loops fit precisely into L1 cache and SIMD registers.

### B. Parallelism and Vectorization Annotations
- **2x/4x Vectorization Width**: The innermost spatial N factor (`j`) frequently defaults to values larger than the hardware SIMD width limit (8 for AVX2), most commonly using 16 or 32. By using 16 (2x `_VEC_WIDTH`), the compiler is encouraged to unroll the micro-kernel over multiple YMM registers, hiding instruction latency and mitigating dependency stalls.
  - *Example (M=32)*: The trace defines the `N` splits as `decision=[12, 1, 16, 16]`. The innermost loop gets exactly 16 elements. 
  - *Example (M=64)*: The `N` splits are `decision=[96, 1, 2, 16]`.
  - *Example (M=96)*: The `N` splits are `decision=[24, 4, 1, 32]`. Here, MetaSchedule expands vectorization/unroll lanes to 4x (32).
- **Parallel Tasks**: The `meta_schedule.parallel` annotation dictates a massive parallelism degree, but the generated loops often dynamically fuse multiple outer spatial iterators.
  - *Example (M=64)*: `l40 = sch.fuse(l30, l31, preserve_unit_iters=True)` -> `sch.parallel(loop=l40)` to cleanly divide the matrix across the CPU cores based on available thread topology.

### C. M-Dimension Spatial Tiling (TM Equivalents)
In the 4-level split for spatial dimension `M`, a distinct pattern emerges:
- For larger `M` values (`M >= 64`), the trace allocates the majority of `M` to a single factor in the hierarchy. It does not arbitrarily clamp block sizes to 32 or 64. When `M` is larger, it allows block sizes like 96 or 128 if the register pressure from `N` and `K` allows it.
  - *Example (M=64)*: `M` split `[1, 1, 64, 1]` allocates the 64 entirely to spatial block level 2.
  - *Example (M=96)*: `M` split `[1, 1, 96, 1]`.
  - *Example (M=128)*: `M` split `[1, 1, 128, 1]`.

### D. K-Dimension Reduction Tiling (TK Equivalents)
The innermost reduction tile (`TK`, or `k1` level) is highly volatile and shape-dependent. 
Contrary to a fixed heuristic like `TK = 8`, MetaSchedule dynamically scales the `K` reduction block to balance the L1 capacity against the spatial block sizes.
- *Example (M=16)*: `K` split `[256, 3]` -> `TK = 3`
- *Example (M=32)*: `K` split `[96, 8]` -> `TK = 8`
- *Example (M=64)*: `K` split `[48, 16]` -> `TK = 16`
- *Example (M=128)*: `K` split `[48, 16]` -> `TK = 16`
- *Example (M=192)*: `K` split `[24, 32]` -> `TK = 32`

### E. Explicit and Auto-Unrolling
The use of `pragma_auto_unroll_max_step` oscillates heavily between extremes. Unroll thresholds tightly couple with workload size, shrinking to bound instruction cache limits when spatial tiles become excessively large.
- *Example (M=64)*: Trace applies `sch.annotate(..., "pragma_auto_unroll_max_step", 16)` due to heavy spatial memory mapping.
- *Example (M=96)*: Trace applies `sch.annotate(..., "pragma_auto_unroll_max_step", 512)` allowing maximal outer loop inner-instruction unfolding.


## 2. Extensive Investigation: Why is MetaSchedule Faster?

While dynamic parameters play a role, the primary structural reason MetaSchedule routinely achieves ~1.7x faster latencies than our rule-based system boils down strictly to **Loop Interleaving and Micro-Kernel construction**.

**A. The "Real GEMM" Inner Loop Order (The SSRSRS Power)**
MetaSchedule applies the reorder operation on split iterators like so:
`sch.reorder(i0, j0, i1, j1, k0, i2, j2, k1, i3, j3)`
Notice carefully where the reduction axes (`k0` and `k1`) are placed relative to the spatial axes (`i` and `j`):
- `k0` frames the outer block updates.
- `k1` is placed **OUTSIDE** `i3` and `j3`.

This layout orchestrates registers perfectly mathematically: The inner reduction step (`k1`) moves along the `K` axis, loading subsets of A and B, and performing outer-product accumulations directly into a static 2D register matrix of `C` dimensioned by `i3 × j3` (where `j3` is the vector lane width). 

*Contrast with the rule-based schedule:*
The rule-based deterministic design executes:
`sch.reorder(i_outer, j_outer, k_outer, i_inner, j_inner_outer, k_inner, j_vec)`
Here, the spatial blocks `i_inner` and `j_inner_outer` reside **OUTSIDE** `k_inner`. This produces a fundamental anti-pattern: For every independent spatial block/vector of `C` being traversed inside a cache block, you restart the `k_inner` iteration, forcing duplicate loads of `A` and `B` strips back into L1/registers independently. By the time the inner dimensions loop across spatial vectors, the data required for `K` is repeatedly dropped and reloaded. MetaSchedule guarantees true register reuse compared to our current loop nesting. 

**B. Granular Multi-Loop Fusions for Exact Thread Synchronization**
Our rule-based system uniformly uses `sch.fuse(i_outer, j_outer)`. MetaSchedule recursively groups `sch.fuse(l30, l31, l32, l33)` — frequently fusing elements like `i0, j0, i1, j1` to generate exactly enough concurrent parallelism chunks proportional to local thread counts, without forcing massive singular outer tiles.

**C. Adaptive `cache_write` placement**
Instead of broadly placing `sch.reverse_compute_at(C_write, j_outer)` like the rule-based version does, MetaSchedule meticulously locks compute boundaries dynamically via `index=-1`. For example, it frequently attaches to `l18` (`j1`) or `l17` (`j0`) depending exactly on how heavy `M` and `N` slice factors were distributed. This perfectly balances between maximizing local memory utilization and preventing L1 capacity evictions.


## 3. Crucial Leads for Optimizing the Rule-Based System

The rule-based system can be radically improved by shifting structural assumptions.

### Lead 1: Fix Loop Order (Implement True Outer Product Micro-Kernel)
The single biggest boost will come from altering the loop layout so that `k_inner` remains **outside** the inner spatial register unrolls (`j_vec` / innermost spatial iterators). Pushing the fast 2D spatial coordinates to the bottom-most level ensures `C` accumulates completely in CPU registers as an outer-product without reloading operand loops independently for every cell.

### Lead 2: Dynamic Auto-Unroll Limits
A static `_AUTO_UNROLL_STEP = 64` ceiling limits performance on smaller layers and blows out instruction caches on large layers. High temporal locality operations (like `M=16`) thrive on deep unrolling (`512`), whereas massive footprint allocations (`M=64` or wider) require tight unroll bounds (`16`).

### Lead 3: Broaden and Parametrize TM Splitting 
Permit larger, contiguous `TM` tiles natively when `M <= 256` instead of clamping bounds arbitrarily into `32` mismatching memory borders. Dedicating the full geometry of `M` (e.g., `TM=96`) prevents irregular modulo executions.

### Lead 4: Shape-Dependent Reduction Block Sizing
Abandon rigid `TK = 8`. Implement a derivation mathematically coupling `TK` to limit usage against calculated spatial allocations (e.g., matching the `L1_CACHE_SIZE / (TM * TN_INNER * datatype_size)` ratio), stretching it to sizes like `32` when shapes represent thin matrices.

### Lead 5: Massive Parallel Oversubscription
Our baseline rule-based system parallelizes tasks exactly proportional to spatial chunks (e.g., `(M/TM) * (N/TN)` = 48 tasks). MetaSchedule routinely injects `meta_schedule.parallel` values explicitly pinned to `128` or `96` even on small matrices. Aggressive oversubscription forces the OS CPU scheduler to balance thread workloads far more optimally across hybrid P and E cores than exact 1-to-1 task allocations.

### Lead 6: Abandoning Strict "Square" Spatial Tiles
Rule-based systems frequently default to `TN=64, TM=64` to preserve symmetry. MetaSchedule strictly opts for highly rectangular tiles when shapes scale (e.g., `M=128, N=3072` gets tiled into spatial workloads leveraging `TN=32` or even `TN=16` against `TM=128`). Rectangular caching drastically reduces L1 capacity contention between `A` (which depends purely on `TM x TK`) and `B` (which depends purely on `TK x TN`).

## 4. Re-check on 2026-03-29 and Implemented Change

### A. Peculiar trends re-confirmed from `best_schedules.json`

Re-parsing all 24 best traces produced these counts:

- Innermost `j` split factor distribution: `16` (12), `32` (5), `8` (3), `64` (2), `1` (2)
- Parallel fused-loop arity in postproc: mostly `4` loops fused (18/24), else `2` (6/24)
- `reverse_compute_at` anchor loop: mostly `l18` (19/24), rarely `l17` (2/24)

The actionable trend for the current 2-level deterministic scheduler is the
non-trivial frequency of `j`-inner = `32`, which was not fully exploited.

### B. Implemented optimization in `rule_based_schedule.py`

Applied change:

- `j_pack` widened from `_VEC_WIDTH * 2` (16) to `_VEC_WIDTH * 4` (32)
- Other core heuristics unchanged (`TM`, `TN`, `TK`, `cache_write` scope, unroll pragma)

Rationale:

- In the simpler 2-level schedule, wider `j` packing increases ILP in the
  innermost compute region and better matches the best-trace tendency to use
  >2x SIMD-width packed inner factors.

### C. Validation summary

Using `python -m research.workloads.bert.matmul.qkv_mlp_run rule_based --all-kernels`
before and after the change (24 shapes total):

- Geometric-mean speedup (new vs previous rule-based): **1.25x**
- Per-kernel speedups: **qkv 1.38x**, **mlp_expand 1.30x**, **mlp_reduce 1.09x**
- Geometric-mean `rule_based / metaschedule` ratio improved from **1.90x** to **1.52x**

### D. Additional leads tested in this pass but not adopted

- Dynamic auto-unroll mappings (`16/64/512` by shape): unstable, no consistent gain
- `reverse_compute_at` deeper at inner `j` loops: frequent regressions
- Reordering to move `k_inner` earlier: consistent slowdown
- Forcing `TK=16` or dynamic `TK`: cross-kernel regressions

Conclusion: the 4x `j_pack` change is the only tested lead from this pass that
consistently improved end-to-end performance without destabilizing other rules.
