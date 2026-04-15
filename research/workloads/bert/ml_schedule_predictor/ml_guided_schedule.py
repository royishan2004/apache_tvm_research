"""Apply a defensive TVM schedule seeded by ML-predicted schedule knobs."""

from __future__ import annotations

import argparse
import logging
from typing import Callable, Dict, List, Optional, Sequence

import tvm

from research.workloads.common.rule_based_schedule import apply_rule_based_schedule

try:
    from .predict_knobs import (
        ALLOWED_UNROLL_FACTORS,
        ALLOWED_VECTOR_WIDTHS,
        DEFAULT_KNOBS,
        predict_schedule_knobs,
    )
except ImportError:
    from predict_knobs import (  # type: ignore  # pylint: disable=import-error
        ALLOWED_UNROLL_FACTORS,
        ALLOWED_VECTOR_WIDTHS,
        DEFAULT_KNOBS,
        predict_schedule_knobs,
    )

LOGGER = logging.getLogger("ml_guided_schedule")


def _neighbor_values(value: int, choices: Sequence[int]) -> List[int]:
    ordered = sorted(int(item) for item in choices)
    if value not in ordered:
        nearest = min(ordered, key=lambda item: abs(item - value))
        value = nearest
    idx = ordered.index(value)
    start = max(0, idx - 1)
    end = min(len(ordered), idx + 2)
    return ordered[start:end]


def refine_knobs_locally(
    base_prediction: Dict[str, int],
    benchmark_fn: Optional[Callable[[Dict[str, int]], float]] = None,
    max_variants: int = 5,
) -> Dict[str, int]:
    """Refine predicted knobs with a tiny local search.

    The helper evaluates at most `max_variants` nearby combinations and is
    CPU-friendly by design. If `benchmark_fn` is not supplied, no refinement
    is attempted and `base_prediction` is returned.
    """
    if benchmark_fn is None:
        return dict(base_prediction)

    vec_candidates = _neighbor_values(base_prediction["vector_width"], ALLOWED_VECTOR_WIDTHS)
    unroll_candidates = _neighbor_values(base_prediction["unroll_factor"], ALLOWED_UNROLL_FACTORS)

    candidates: List[Dict[str, int]] = []
    for vec in vec_candidates:
        for unroll in unroll_candidates:
            candidate = dict(base_prediction)
            candidate["vector_width"] = int(vec)
            candidate["unroll_factor"] = int(unroll)
            candidates.append(candidate)

    if len(candidates) > max_variants:
        center = len(candidates) // 2
        half = max_variants // 2
        start = max(0, center - half)
        candidates = candidates[start : start + max_variants]

    best = dict(base_prediction)
    best_latency = float("inf")

    for candidate in candidates:
        try:
            latency = float(benchmark_fn(candidate))
            if latency < best_latency:
                best_latency = latency
                best = candidate
        except Exception as err:  # pylint: disable=broad-except
            LOGGER.debug("Local refinement candidate failed %s (%s)", candidate, err)

    if best_latency < float("inf"):
        LOGGER.info("Local refinement selected %s with latency %.3f us", best, best_latency)
    return best


def _predict_with_fallback(kernel_name: str, M: int, K: int, N: int) -> Dict[str, int]:
    fallback = dict(DEFAULT_KNOBS)
    try:
        prediction = predict_schedule_knobs(kernel_name=kernel_name, M=M, K=K, N=N)
        merged = dict(fallback)
        merged.update(prediction)
        return merged
    except Exception as err:  # pylint: disable=broad-except
        LOGGER.warning("Prediction unavailable, using defaults (%s)", err)
        return fallback


def _safe_split_for_vectorize(sch: tvm.tir.Schedule, loop, vector_width: int):
    try:
        return sch.split(loop, factors=[None, vector_width])
    except tvm.tir.ScheduleError:
        return None, None


def apply_ml_guided_schedule(
    mod,
    M: int,
    K: int,
    N: int,
    kernel_name: str = "qkv",
    enable_refinement: bool = False,
    benchmark_fn: Optional[Callable[[Dict[str, int]], float]] = None,
):
    """Apply an ML-guided TVM schedule with defensive fallbacks.

    If scheduling fails for any reason, this function falls back to the
    existing rule-based scheduler so tuning and benchmarking never break.
    """
    prediction = _predict_with_fallback(kernel_name=kernel_name, M=M, K=K, N=N)
    if enable_refinement:
        prediction = refine_knobs_locally(prediction, benchmark_fn=benchmark_fn)

    LOGGER.info(
        "[ml_guided] M=%d K=%d N=%d kernel=%s -> %s",
        M,
        K,
        N,
        kernel_name,
        prediction,
    )

    try:
        sch = tvm.tir.Schedule(mod)
        sch.work_on("main")

        block = sch.get_block("C")
        i, j, k = sch.get_loops(block)

        tm = min(64, max(1, int(M)))
        tn = min(max(prediction["vector_width"] * 4, prediction["vector_width"]), int(N))
        tk = min(8, max(1, int(K)))

        i_outer, i_inner = sch.split(i, factors=[None, tm])
        j_outer, j_inner = sch.split(j, factors=[None, max(1, tn)])
        k_outer, k_inner = sch.split(k, factors=[None, tk])

        vec_outer, vec_inner = _safe_split_for_vectorize(
            sch,
            j_inner,
            max(1, int(prediction["vector_width"])),
        )

        reorder_items = [i_outer, j_outer, k_outer, i_inner]
        if vec_outer is not None and vec_inner is not None:
            reorder_items.extend([vec_outer, k_inner, vec_inner])
        else:
            reorder_items.extend([j_inner, k_inner])
        sch.reorder(*reorder_items)

        cache_block = None
        if int(prediction["cache_write_used"]) == 1:
            try:
                cache_block = sch.cache_write(block, 0, "global")
                sch.reverse_compute_at(cache_block, j_outer)
            except tvm.tir.ScheduleError as err:
                LOGGER.debug("cache_write skipped: %s", err)

        fused = sch.fuse(i_outer, j_outer)
        sch.parallel(fused)

        if vec_inner is not None:
            try:
                sch.vectorize(vec_inner)
            except tvm.tir.ScheduleError as err:
                LOGGER.debug("vectorize skipped: %s", err)

        if cache_block is not None:
            try:
                write_loops = sch.get_loops(cache_block)
                if write_loops:
                    split = _safe_split_for_vectorize(
                        sch,
                        write_loops[-1],
                        max(1, int(prediction["vector_width"])),
                    )
                    if split[1] is not None:
                        sch.vectorize(split[1])
            except tvm.tir.ScheduleError as err:
                LOGGER.debug("write-back vectorization skipped: %s", err)

        sch.annotate(
            fused,
            "pragma_auto_unroll_max_step",
            int(prediction["unroll_factor"]),
        )
        sch.annotate(fused, "pragma_unroll_explicit", 1)

        if int(prediction["reduction_decompose_used"]) == 1:
            try:
                sch.decompose_reduction(block, k_outer)
            except tvm.tir.ScheduleError as err:
                LOGGER.debug("decompose_reduction skipped: %s", err)

        return sch.mod

    except Exception as err:  # pylint: disable=broad-except
        LOGGER.warning("ML-guided scheduling failed, falling back to rule-based (%s)", err)
        return apply_rule_based_schedule(mod, M=M, K=K, N=N, kernel=kernel_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dry-run an ML-guided schedule application")
    parser.add_argument("--kernel", default="qkv", help="Kernel type")
    parser.add_argument("--M", type=int, required=True, help="M dimension")
    parser.add_argument("--K", type=int, required=True, help="K dimension")
    parser.add_argument("--N", type=int, required=True, help="N dimension")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    prediction = _predict_with_fallback(args.kernel, args.M, args.K, args.N)
    LOGGER.info("Predicted knobs: %s", prediction)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
