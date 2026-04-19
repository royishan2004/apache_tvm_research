import argparse
import inspect
import json
import os
import hashlib
import statistics

import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm.meta_schedule.tune_context import _normalize_mod
from tvm.target import Target

from research.workloads.bert.bert_shapes import (
    M_LIST,
    mlp_compressed_shape,
    mlp_expanded_shape,
    qkv_shape,
)
from research.workloads.bert.metaschedule.metaschedule_best_schedules import save_best_schedule
from research.workloads.common.data_aggregator_client import (
    ensure_data_aggregator_connection_or_prompt,
    resolve_profile,
)
from research.workloads.common.benchmark_environment import ensure_benchmark_settings_or_prompt
from research.workloads.common.matmul_templates import matmul_tir

TARGET = Target("llvm -num-cores=8")
WORK_DIR_BASE_DEFAULT = "research/results/metaschedule/best_config"
BEST_CONFIG_PATH_DEFAULT = "research/results/metaschedule/best_pruned_config.json"
SCHEDULES_FILE_DEFAULT = "research/results/metaschedule/best_schedules_metaschedule_best_config.json"
RESULTS_FILE_DEFAULT = "research/results/bert_matmul_results.json"

KERNELS = {
    "qkv": qkv_shape,
    "mlp_expand": mlp_expanded_shape,
    "mlp_reduce": mlp_compressed_shape,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MetaSchedule using the selected 80/20 best_pruned_config tuning parameters"
    )
    parser.add_argument("--kernel", choices=sorted(KERNELS), help="Tune only one kernel")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Tune all kernels (default if --kernel is omitted)",
    )
    parser.add_argument("--iterations", type=int, default=1, help="Repeat full sweep N times")
    parser.add_argument(
        "--best-config-path",
        default=BEST_CONFIG_PATH_DEFAULT,
        help="Path to 80/20 best_pruned_config.json",
    )
    parser.add_argument(
        "--work-dir-base",
        default=WORK_DIR_BASE_DEFAULT,
        help="Base output folder for MetaSchedule databases",
    )
    parser.add_argument(
        "--variant",
        default="metaschedule_best_config",
        help="Variant label stored in bert_matmul_results.json",
    )
    parser.add_argument(
        "--schedules-file",
        default=SCHEDULES_FILE_DEFAULT,
        help="Path to schedule trace output JSON",
    )
    parser.add_argument(
        "--results-file",
        default=RESULTS_FILE_DEFAULT,
        help="Path to aggregate results JSON",
    )
    parser.add_argument("--profile", default=None, help="Optional data-aggregator profile")
    parser.add_argument("--seed", type=int, default=20260407, help="Deterministic seed matching the 80/20 tuner")
    args = parser.parse_args()

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    return args


def _resolve_kernel_list(args: argparse.Namespace):
    if args.all:
        if args.kernel:
            print(f"--all was specified, ignoring --kernel={args.kernel}")
        return list(KERNELS.keys())
    if args.kernel:
        return [args.kernel]
    return list(KERNELS.keys())


def _load_best_config(path: str) -> tuple:
    if not os.path.exists(path):
        raise FileNotFoundError(f"best-config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError("best-config payload must be a JSON object")

    cfg = payload.get("selected_config")
    if not isinstance(cfg, dict):
        cfg = payload.get("config")
    if not isinstance(cfg, dict):
        raise ValueError("best-config JSON must include 'selected_config' or 'config'")
    return cfg, payload.get("history_summary", {})


import math

def _shape_similarity_multiplier(m_val: int, kernel_stats: dict) -> float:
    m_values = kernel_stats.get("M_values", [])
    if not m_values:
        return 1.0
    nearest = min(m_values, key=lambda x: abs(x - m_val))
    rel_gap = abs(nearest - m_val) / max(float(m_val), 1.0)
    instability = float(kernel_stats.get("instability", 0.5))
    similarity = math.exp(-3.0 * rel_gap)
    multiplier = 1.0 - 0.12 * similarity * (1.0 - instability) + 0.10 * (1.0 - similarity) * instability
    return float(max(0.85, min(1.20, multiplier)))

def _apply_kernel_aware_bias(cfg: dict, kernel: str, m_val: int, history_summary: dict) -> dict:
    biased = dict(cfg)
    kernel_stats = history_summary.get("kernel_stats", {}).get(kernel)
    if not kernel_stats:
        return biased

    kernel_mult = float(kernel_stats.get("kernel_budget_multiplier", 1.0))
    shape_mult = _shape_similarity_multiplier(m_val, kernel_stats)
    combined = max(0.75, min(1.35, kernel_mult * shape_mult))

    num_trials_per_iter = int(cfg.get("num_trials_per_iter", 64))
    biased["max_trials_global"] = max(num_trials_per_iter, int(round(int(cfg.get("max_trials_global", 256)) * combined)))
    biased["max_trials_per_task"] = max(num_trials_per_iter, int(round(int(cfg.get("max_trials_per_task", 256)) * combined)))
    biased["population_size"] = max(64, int(round(int(cfg.get("population_size", 512)) * combined)))
    biased["design_space_samples"] = max(8, int(round(int(cfg.get("design_space_samples", 50)) * max(0.9, combined))))
    return biased

def _filter_supported_kwargs(callable_obj, kwargs: dict) -> dict:
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs

    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if accepts_var_kwargs:
        return kwargs

    supported = {}
    for key, value in kwargs.items():
        if key in sig.parameters:
            supported[key] = value
    return supported

def _stable_hash(payload: dict) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]

def _deterministic_seed(
    base_seed: int,
    iteration: int,
    config_hash: str,
    kernel: str,
    m_val: int,
) -> int:
    token = f"{base_seed}:{iteration}:{config_hash}:{kernel}:{m_val}"
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _build_search_strategy(cfg: dict):
    kwargs = {
        "population_size": int(cfg.get("population_size", 512)),
        "init_measured_ratio": float(cfg.get("init_measured_ratio", 0.20)),
        "init_min_unmeasured": int(cfg.get("design_space_samples", 50)),
        "max_fail_count": int(cfg.get("trace_replay_count", 5)),
        "genetic_num_iters": int(cfg.get("genetic_num_iters", 4)),
        "genetic_mutate_prob": float(cfg.get("mutation_aggressiveness", 0.85)),
        "genetic_max_fail_count": int(cfg.get("genetic_max_fail_count", 10)),
        "eps_greedy": float(cfg.get("eps_greedy", 0.05)),
    }
    ctor = ms.search_strategy.EvolutionarySearch
    return ctor(**_filter_supported_kwargs(ctor, kwargs))


def _run_tuning(mod, work_dir: str, cfg: dict, tune_seed: int):
    evaluator_cfg = ms.runner.EvaluatorConfig(
        number=int(cfg.get("evaluator_number", 5)),
        repeat=int(cfg.get("evaluator_repeat", 1)),
        min_repeat_ms=int(cfg.get("min_repeat_ms", 100)),
    )
    tune_kwargs = {
        "mod": mod,
        "target": TARGET,
        "work_dir": work_dir,
        "max_trials_global": int(cfg.get("max_trials_global", 256)),
        "max_trials_per_task": int(cfg.get("max_trials_per_task", 256)),
        "num_trials_per_iter": int(cfg.get("num_trials_per_iter", 64)),
        "builder": ms.builder.LocalBuilder(),
        "runner": ms.runner.LocalRunner(evaluator_config=evaluator_cfg),
        "space": ms.space_generator.PostOrderApply(),
        "strategy": _build_search_strategy(cfg),
        "seed": tune_seed,
        "num_tuning_cores": 8,
    }
    return ms.tir_integration.tune_tir(**_filter_supported_kwargs(ms.tir_integration.tune_tir, tune_kwargs))


def _rigorous_latency(mod, best_record, M: int, K: int, N: int, cfg: dict, seed: int):
    sch = tvm.tir.Schedule(mod)
    best_record.trace.apply_to_schedule(sch, remove_postproc=False)
    rt_mod = tvm.build(sch.mod, target=TARGET)

    dev = tvm.cpu(0)
    evaluator = rt_mod.time_evaluator(
        "main",
        dev=dev,
        number=int(cfg.get("rigorous_number", 50)),
        repeat=int(cfg.get("rigorous_repeat", 3)),
        min_repeat_ms=int(cfg.get("rigorous_min_repeat_ms", 50)),
    )

    validation_runs = max(1, int(cfg.get("rigorous_validation_runs", 1)))
    run_means = []
    eval_stds = []

    for run_id in range(validation_runs):
        rng = np.random.default_rng(seed + run_id * 9973)
        a_np = rng.normal(0.0, 1.0, size=(M, K)).astype("float32")
        b_np = rng.normal(0.0, 1.0, size=(K, N)).astype("float32")
        c_np = np.zeros((M, N), dtype="float32")
        res = evaluator(tvm.nd.array(a_np, dev), tvm.nd.array(b_np, dev), tvm.nd.array(c_np, dev))
        run_means.append(float(res.mean) * 1e6)
        eval_stds.append(float(res.std) * 1e6)

    latency_us = float(statistics.fmean(run_means)) if run_means else float("inf")
    if len(run_means) > 1:
        std_us = float(statistics.pstdev(run_means))
    else:
        std_us = float(eval_stds[0]) if eval_stds else 0.0
    return latency_us, std_us


def main() -> int:
    args = _parse_args()
    profile = resolve_profile(args.profile)

    if not ensure_benchmark_settings_or_prompt("metaschedule_tune_best_config"):
        return 1
    if not ensure_data_aggregator_connection_or_prompt("metaschedule_tune_best_config"):
        return 1

    cfg, history_summary = _load_best_config(args.best_config_path)
    selected_kernels = _resolve_kernel_list(args)

    os.makedirs(args.work_dir_base, exist_ok=True)
    os.makedirs(os.path.dirname(args.schedules_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    config_hash = _stable_hash(cfg)

    print("Starting MetaSchedule tuning with 80/20 selected config")
    print(f"Target: {TARGET}")
    print(f"Profile: {profile}")
    print(f"Variant: {args.variant}")
    print(f"Best-config path: {args.best_config_path}")
    print(f"Deterministic base seed: {args.seed}, Config Hash: {config_hash}")
    print(
        "Loaded config: "
        f"max_trials_global={cfg.get('max_trials_global')} "
        f"max_trials_per_task={cfg.get('max_trials_per_task')} "
        f"num_trials_per_iter={cfg.get('num_trials_per_iter')}"
    )

    for iteration in range(1, args.iterations + 1):
        if args.iterations > 1:
            print(f"\n{'#' * 120}")
            print(f"### Iteration {iteration}/{args.iterations}")
            print(f"{'#' * 120}")

        for kernel_name in selected_kernels:
            shape_fn = KERNELS[kernel_name]
            kernel_dir = os.path.join(args.work_dir_base, kernel_name)
            os.makedirs(kernel_dir, exist_ok=True)
            print(f"\n=== Kernel: {kernel_name} -> work dir: {kernel_dir} ===")

            for M_val in M_LIST:
                M, K, N = shape_fn(M_val)
                mod = matmul_tir(M, K, N)
                work_dir = os.path.join(kernel_dir, f"M_{M}")
                os.makedirs(work_dir, exist_ok=True)

                iter_label = f" [iter {iteration}/{args.iterations}]" if args.iterations > 1 else ""
                print(f"\n{'=' * 120}")
                print(
                    f"\nTuning kernel={kernel_name} M={M} K={K} N={N}{iter_label} -> work dir: {work_dir}"
                )
                print(f"\n{'=' * 120}\n")

                biased_cfg = _apply_kernel_aware_bias(cfg, kernel_name, M, history_summary)
                print(f"Effective tuning budget: global={biased_cfg.get('max_trials_global')}, "
                      f"per_task={biased_cfg.get('max_trials_per_task')}, "
                      f"population={biased_cfg.get('population_size')}")

                tune_seed = _deterministic_seed(args.seed, iteration, config_hash, kernel_name, M)
                print(f"Deterministic step tune_seed={tune_seed}")

                database = _run_tuning(mod=mod, work_dir=work_dir, cfg=biased_cfg, tune_seed=tune_seed)
                print(f"Completed tuning for kernel={kernel_name} M={M}")

                normalized_mod = _normalize_mod(mod["main"])
                best_record = database.query_tuning_record(normalized_mod, TARGET, "main")
                if best_record is None:
                    print(f"No tuning record found for kernel={kernel_name} M={M}")
                    continue

                latency_us, std_us = _rigorous_latency(mod, best_record, M, K, N, biased_cfg, seed=tune_seed)
                print(
                    f"Best schedule for kernel={kernel_name} M={M} "
                    f"(latency={latency_us:.2f} us +- {std_us:.2f} us)"
                )
                print(best_record.trace)

                save_best_schedule(
                    kernel_name=kernel_name,
                    M=M,
                    K=K,
                    N=N,
                    best_record=best_record,
                    latency_us=latency_us,
                    std_us=std_us,
                    profile=profile,
                    variant=args.variant,
                    runs_label=f"MetaSchedule ({args.variant})",
                    source_label="MetaSchedule-best-pruned-config",
                    schedules_file=args.schedules_file,
                    results_file=args.results_file,
                )
                print(
                    f"Persisted variant={args.variant} for kernel={kernel_name} M={M} "
                    f"into {args.schedules_file} and {args.results_file}"
                )

        if args.iterations > 1:
            print(f"\nIteration {iteration}/{args.iterations} completed")

    print("\nAll MetaSchedule best-config tuning runs completed successfully")
    print(f"Schedules file: {args.schedules_file}")
    print(f"Results file: {args.results_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
