#!/usr/bin/env python3
"""80/20 MetaSchedule tuner for BERT MatMul kernels.

This script auto-prunes MetaSchedule tuning configuration to target near-baseline
performance at substantially lower tuning cost.
"""

import argparse
import copy
import hashlib
import inspect
import json
import logging
import math
import os
import random
import re
import statistics
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
import sys

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
	upload_best_pruned_config,
	upload_pruning_experiments,
)
from research.workloads.common.matmul_templates import matmul_tir


TARGET = Target("llvm -num-cores=8")
WORK_DIR_BASE_DEFAULT = "research/results/metaschedule/8020"
RESULTS_DIR = "research/results/metaschedule"
BEST_SCHEDULES_FILE = os.path.join(RESULTS_DIR, "best_schedules.json")
BEST_PRUNED_CONFIG_FILE = os.path.join(RESULTS_DIR, "best_pruned_config.json")
PRUNING_EXPERIMENTS_FILE = os.path.join(RESULTS_DIR, "pruning_experiments.json")
COMPARE_RESULTS_FILE = os.path.join(RESULTS_DIR, "comparison_results.json")
LOG_FILE = os.path.join(WORK_DIR_BASE_DEFAULT, "metaschedule_8020_tuner.log")

KERNELS = {
	"qkv": qkv_shape,
	"mlp_expand": mlp_expanded_shape,
	"mlp_reduce": mlp_compressed_shape,
}

# Progressive budget pruning ladder.
GLOBAL_TRIAL_LEVELS = [256, 192, 128, 96, 64]
PER_ITER_LEVELS = [64, 32, 16]

# Measurement quality/cost ladder (index 0 mirrors baseline script).
MEASUREMENT_LEVELS = [
	{
		"evaluator_number": 5,
		"evaluator_repeat": 1,
		"min_repeat_ms": 100,
		"rigorous_number": 50,
		"rigorous_repeat": 3,
		"rigorous_min_repeat_ms": 50,
		"rigorous_validation_runs": 1,
	},
	{
		"evaluator_number": 4,
		"evaluator_repeat": 1,
		"min_repeat_ms": 80,
		"rigorous_number": 40,
		"rigorous_repeat": 2,
		"rigorous_min_repeat_ms": 40,
		"rigorous_validation_runs": 1,
	},
	{
		"evaluator_number": 3,
		"evaluator_repeat": 1,
		"min_repeat_ms": 60,
		"rigorous_number": 30,
		"rigorous_repeat": 2,
		"rigorous_min_repeat_ms": 30,
		"rigorous_validation_runs": 1,
	},
	{
		"evaluator_number": 2,
		"evaluator_repeat": 1,
		"min_repeat_ms": 40,
		"rigorous_number": 20,
		"rigorous_repeat": 1,
		"rigorous_min_repeat_ms": 20,
		"rigorous_validation_runs": 1,
	},
]

# Search breadth controls: evolutionary strategy pruning knobs.
BREADTH_LEVELS = [
	{
		"population_size": 512,
		"init_measured_ratio": 0.20,
		"design_space_samples": 50,
		"trace_replay_count": 5,
		"genetic_num_iters": 4,
		"mutation_aggressiveness": 0.85,
		"genetic_max_fail_count": 10,
		"eps_greedy": 0.05,
	},
	{
		"population_size": 384,
		"init_measured_ratio": 0.20,
		"design_space_samples": 40,
		"trace_replay_count": 5,
		"genetic_num_iters": 3,
		"mutation_aggressiveness": 0.80,
		"genetic_max_fail_count": 8,
		"eps_greedy": 0.08,
	},
	{
		"population_size": 256,
		"init_measured_ratio": 0.22,
		"design_space_samples": 32,
		"trace_replay_count": 4,
		"genetic_num_iters": 3,
		"mutation_aggressiveness": 0.75,
		"genetic_max_fail_count": 7,
		"eps_greedy": 0.10,
	},
	{
		"population_size": 192,
		"init_measured_ratio": 0.24,
		"design_space_samples": 24,
		"trace_replay_count": 3,
		"genetic_num_iters": 2,
		"mutation_aggressiveness": 0.70,
		"genetic_max_fail_count": 6,
		"eps_greedy": 0.12,
	},
	{
		"population_size": 128,
		"init_measured_ratio": 0.28,
		"design_space_samples": 16,
		"trace_replay_count": 2,
		"genetic_num_iters": 2,
		"mutation_aggressiveness": 0.65,
		"genetic_max_fail_count": 5,
		"eps_greedy": 0.15,
	},
]


LOGGER = logging.getLogger("metaschedule_8020_tuner")

# Color support: enable when stdout is a tty. Logs may still include codes if handler
# writes to the same stream; this keeps colors off in non-interactive runs.
COLOR_ENABLED = sys.stdout.isatty()
_ANSI_GREEN = "\x1b[32m"
_ANSI_RED = "\x1b[31m"
_ANSI_RESET = "\x1b[0m"


def _maybe_color(text: str, good: Optional[bool]) -> str:
	if not COLOR_ENABLED or good is None:
		return text
	return f"{_ANSI_GREEN}{text}{_ANSI_RESET}" if good else f"{_ANSI_RED}{text}{_ANSI_RESET}"


# Regex for stripping ANSI escape sequences when computing visible widths
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(s: str) -> str:
	return _ANSI_RE.sub("", s)


def _visible_len(s: str) -> int:
	return len(_strip_ansi(s))


def _arrow_prefix(good: Optional[bool]) -> str:
	if good is None:
		return "→ "
	return "↑ " if good else "↓ "


@dataclass(frozen=True)
class PruningState:
	global_idx: int = 0
	per_iter_idx: int = 0
	measurement_idx: int = 0
	breadth_idx: int = 0

	def token(self) -> str:
		return (
			f"g{self.global_idx}-p{self.per_iter_idx}"
			f"-m{self.measurement_idx}-b{self.breadth_idx}"
		)


@dataclass(frozen=True)
class TuningConfig:
	name: str
	state_token: str
	max_trials_global: int
	max_trials_per_task: int
	num_trials_per_iter: int
	evaluator_number: int
	evaluator_repeat: int
	min_repeat_ms: int
	rigorous_number: int
	rigorous_repeat: int
	rigorous_min_repeat_ms: int
	rigorous_validation_runs: int
	population_size: int
	init_measured_ratio: float
	design_space_samples: int
	trace_replay_count: int
	genetic_num_iters: int
	mutation_aggressiveness: float
	genetic_max_fail_count: int
	eps_greedy: float


@dataclass(frozen=True)
class TaskSpec:
	kernel: str
	M: int
	K: int
	N: int
	size_bucket: str


@dataclass
class RunProgressTracker:
	planned_runs: int = 0
	completed_runs: int = 0

	def register_config_runs(self, num_tasks: int) -> None:
		self.planned_runs += max(0, int(num_tasks))

	def next_run_index(self) -> int:
		return self.completed_runs + 1

	def mark_run_completed(self) -> None:
		self.completed_runs += 1


def _utc_now() -> str:
	return datetime.now(timezone.utc).isoformat()


def _configure_logging(log_path: str) -> None:
	os.makedirs(os.path.dirname(log_path), exist_ok=True)
	LOGGER.setLevel(logging.INFO)
	LOGGER.handlers.clear()
	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	console = logging.StreamHandler()
	console.setFormatter(formatter)
	LOGGER.addHandler(console)

	file_handler = logging.FileHandler(log_path)
	file_handler.setFormatter(formatter)
	LOGGER.addHandler(file_handler)


def _load_json(path: str, default: Any) -> Any:
	if not os.path.exists(path):
		return copy.deepcopy(default)
	try:
		with open(path, "r", encoding="utf-8") as f:
			return json.load(f)
	except Exception as err:  # pylint: disable=broad-except
		LOGGER.warning("Failed to parse JSON from %s: %s", path, err)
		return copy.deepcopy(default)


def _write_json(path: str, payload: Any) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w", encoding="utf-8") as f:
		json.dump(payload, f, indent=2)


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="80/20 MetaSchedule auto-pruning tuner for BERT MatMul kernels"
	)
	parser.add_argument("--kernel", choices=sorted(KERNELS), help="Tune only one kernel")
	parser.add_argument(
		"--all",
		action="store_true",
		help="Tune all kernels (default if --kernel is not provided)",
	)
	parser.add_argument("--iterations", type=int, default=1, help="Outer search iterations")
	parser.add_argument(
		"--benchmark-only",
		action="store_true",
		help="Benchmark existing work_dirs without running new tuning",
	)
	parser.add_argument(
		"--compare-against-baseline",
		action="store_true",
		help="Run only baseline vs pruned-config comparison",
	)
	parser.add_argument(
		"--best-config-path",
		default=BEST_PRUNED_CONFIG_FILE,
		help="Path to best_pruned_config.json used in compare/benchmark mode",
	)
	parser.add_argument(
		"--work-dir-base",
		default=WORK_DIR_BASE_DEFAULT,
		help="Base work dir for 80/20 tuning runs",
	)
	parser.add_argument("--seed", type=int, default=20260407, help="Base deterministic seed")
	parser.add_argument(
		"--max-pruning-steps",
		type=int,
		default=8,
		help="Max iterative pruning steps per outer iteration",
	)
	parser.add_argument(
		"--max-latency-loss-pct",
		type=float,
		default=12.5,
		help="Stop pruning once latency loss exceeds this percentage",
	)
	parser.add_argument(
		"--target-retention",
		type=float,
		default=0.85,
		help="Target latency retention (baseline/candidate)",
	)
	parser.add_argument(
		"--min-score-gain",
		type=float,
		default=0.02,
		help="Minimal score gain required to continue pruning",
	)
	parser.add_argument(
		"--min-time-reduction-gain",
		type=float,
		default=0.05,
		help="Minimal additional time reduction required to continue pruning",
	)
	parser.add_argument(
		"--force-rerun",
		action="store_true",
		help="Ignore cached experiments and rerun the same configuration",
	)
	parser.add_argument(
		"--profile",
		default=None,
		help="Optional explicit data-aggregator profile",
	)
	args = parser.parse_args()
	if args.iterations < 1:
		parser.error("--iterations must be >= 1")
	if args.max_pruning_steps < 1:
		parser.error("--max-pruning-steps must be >= 1")
	if not 0.0 < args.target_retention <= 1.0:
		parser.error("--target-retention must be in (0, 1]")
	return args


def _resolve_kernel_list(args: argparse.Namespace) -> List[str]:
	if args.all:
		if args.kernel:
			LOGGER.info("--all was specified, ignoring --kernel=%s", args.kernel)
		return list(KERNELS.keys())
	if args.kernel:
		return [args.kernel]
	return list(KERNELS.keys())


def _representative_m_values(all_m_values: Sequence[int]) -> List[int]:
	ordered = sorted({int(v) for v in all_m_values})
	if len(ordered) <= 3:
		return ordered
	return sorted({ordered[0], ordered[len(ordered) // 2], ordered[-1]})


def _m_bucket(m_val: int, ordered_m: Sequence[int]) -> str:
	ordered = sorted(ordered_m)
	if not ordered:
		return "medium"
	p33 = ordered[len(ordered) // 3]
	p66 = ordered[(2 * len(ordered)) // 3]
	if m_val <= p33:
		return "small"
	if m_val >= p66:
		return "large"
	return "medium"


def _build_tasks(selected_kernels: Sequence[str]) -> List[TaskSpec]:
	m_values = _representative_m_values(M_LIST)
	tasks: List[TaskSpec] = []
	for kernel in selected_kernels:
		shape_fn = KERNELS[kernel]
		for m_val in m_values:
			M, K, N = shape_fn(m_val)
			tasks.append(
				TaskSpec(
					kernel=kernel,
					M=M,
					K=K,
					N=N,
					size_bucket=_m_bucket(M, sorted(M_LIST)),
				)
			)
	return tasks


def _state_to_config(state: PruningState) -> TuningConfig:
	g = GLOBAL_TRIAL_LEVELS[state.global_idx]
	per_iter = min(PER_ITER_LEVELS[state.per_iter_idx], g)
	measurement = MEASUREMENT_LEVELS[state.measurement_idx]
	breadth = BREADTH_LEVELS[state.breadth_idx]
	name = "baseline" if state == PruningState() else f"pruned_{state.token()}"

	return TuningConfig(
		name=name,
		state_token=state.token(),
		max_trials_global=int(g),
		max_trials_per_task=int(g),
		num_trials_per_iter=int(per_iter),
		evaluator_number=int(measurement["evaluator_number"]),
		evaluator_repeat=int(measurement["evaluator_repeat"]),
		min_repeat_ms=int(measurement["min_repeat_ms"]),
		rigorous_number=int(measurement["rigorous_number"]),
		rigorous_repeat=int(measurement["rigorous_repeat"]),
		rigorous_min_repeat_ms=int(measurement["rigorous_min_repeat_ms"]),
		rigorous_validation_runs=int(measurement["rigorous_validation_runs"]),
		population_size=int(breadth["population_size"]),
		init_measured_ratio=float(breadth["init_measured_ratio"]),
		design_space_samples=int(breadth["design_space_samples"]),
		trace_replay_count=int(breadth["trace_replay_count"]),
		genetic_num_iters=int(breadth["genetic_num_iters"]),
		mutation_aggressiveness=float(breadth["mutation_aggressiveness"]),
		genetic_max_fail_count=int(breadth["genetic_max_fail_count"]),
		eps_greedy=float(breadth["eps_greedy"]),
	)


def _next_candidate_states(current: PruningState, seen: set) -> List[PruningState]:
	g_max = len(GLOBAL_TRIAL_LEVELS) - 1
	p_max = len(PER_ITER_LEVELS) - 1
	m_max = len(MEASUREMENT_LEVELS) - 1
	b_max = len(BREADTH_LEVELS) - 1

	moves = [
		(1, 0, 0, 0),
		(0, 1, 0, 0),
		(0, 0, 1, 0),
		(0, 0, 0, 1),
		(1, 1, 0, 0),
		(1, 0, 1, 0),
		(1, 0, 0, 1),
	]

	candidates: List[PruningState] = []
	for dg, dp, dm, db in moves:
		nxt = PruningState(
			global_idx=min(current.global_idx + dg, g_max),
			per_iter_idx=min(current.per_iter_idx + dp, p_max),
			measurement_idx=min(current.measurement_idx + dm, m_max),
			breadth_idx=min(current.breadth_idx + db, b_max),
		)
		if nxt == current or nxt in seen:
			continue
		candidates.append(nxt)

	def _severity(st: PruningState) -> Tuple[int, int, int, int]:
		return (st.global_idx + st.per_iter_idx + st.measurement_idx + st.breadth_idx,
				st.global_idx,
				st.per_iter_idx,
				st.measurement_idx + st.breadth_idx)

	candidates.sort(key=_severity)
	return candidates


def _stable_hash(payload: Any) -> str:
	canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
	return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:16]


def _config_hash(config: TuningConfig) -> str:
	return _stable_hash(asdict(config))


def _tasks_signature(tasks: Sequence[TaskSpec]) -> str:
	payload = [asdict(t) for t in tasks]
	return _stable_hash(payload)


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


def _filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
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
	for k, v in kwargs.items():
		if k in sig.parameters:
			supported[k] = v
	return supported


def _count_database_records(work_dir: str) -> int:
	path = os.path.join(work_dir, "database_tuning_record.json")
	if not os.path.exists(path):
		return 0
	count = 0
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			if line.strip():
				count += 1
	return count


def _load_existing_database(work_dir: str):
	workload_path = os.path.join(work_dir, "database_workload.json")
	tuning_path = os.path.join(work_dir, "database_tuning_record.json")
	if not (os.path.exists(workload_path) and os.path.exists(tuning_path)):
		raise FileNotFoundError(
			f"Missing database files in benchmark-only mode: {work_dir}"
		)
	return ms.database.JSONDatabase(work_dir=work_dir, allow_missing=False)


def _build_space_generator() -> Any:
	try:
		return ms.space_generator.PostOrderApply()
	except Exception as err:  # pylint: disable=broad-except
		LOGGER.warning("Falling back to default space generator string: %s", err)
		return "post-order-apply"


def _build_search_strategy(config: TuningConfig) -> Any:
	kwargs = {
		"population_size": config.population_size,
		"init_measured_ratio": config.init_measured_ratio,
		"init_min_unmeasured": config.design_space_samples,
		"max_fail_count": config.trace_replay_count,
		"genetic_num_iters": config.genetic_num_iters,
		"genetic_mutate_prob": config.mutation_aggressiveness,
		"genetic_max_fail_count": config.genetic_max_fail_count,
		"eps_greedy": config.eps_greedy,
	}
	ctor = ms.search_strategy.EvolutionarySearch
	kwargs = _filter_supported_kwargs(ctor, kwargs)
	try:
		return ctor(**kwargs)
	except Exception as err:  # pylint: disable=broad-except
		LOGGER.warning("Falling back to default evolutionary strategy string: %s", err)
		return "evolutionary"


def _run_tuning(
	mod: tvm.ir.IRModule,
	work_dir: str,
	config: TuningConfig,
	tune_seed: int,
):
	evaluator_cfg = ms.runner.EvaluatorConfig(
		number=config.evaluator_number,
		repeat=config.evaluator_repeat,
		min_repeat_ms=config.min_repeat_ms,
	)

	tune_kwargs = {
		"mod": mod,
		"target": TARGET,
		"work_dir": work_dir,
		"max_trials_global": config.max_trials_global,
		"max_trials_per_task": config.max_trials_per_task,
		"num_trials_per_iter": config.num_trials_per_iter,
		"builder": ms.builder.LocalBuilder(),
		"runner": ms.runner.LocalRunner(evaluator_config=evaluator_cfg),
		"space": _build_space_generator(),
		"strategy": _build_search_strategy(config),
		"seed": tune_seed,
		"num_tuning_cores": 8,
	}

	supported_kwargs = _filter_supported_kwargs(ms.tir_integration.tune_tir, tune_kwargs)
	return ms.tir_integration.tune_tir(**supported_kwargs)


_TILE_DECISION_RE = re.compile(r"sample_perfect_tile\([^\)]*decision=\[([^\]]+)\]")
_VECTOR_ANN_RE = re.compile(r"ann_key=\"meta_schedule\.vectorize\", ann_val=([0-9]+)")
_UNROLL_ANN_RE = re.compile(r"ann_key=\"meta_schedule\.unroll_explicit\", ann_val=([0-9]+)")
_CATEGORICAL_RE = re.compile(
	r"sample_categorical\(candidates=\[([^\]]+)\][^\)]*decision=([0-9]+)"
)


def _parse_ints(text: str) -> List[int]:
	return [int(v) for v in re.findall(r"-?[0-9]+", text)]


def _summarize_trace_text(trace_text: str) -> Dict[str, Any]:
	split_factors: List[List[int]] = []
	for match in _TILE_DECISION_RE.finditer(trace_text):
		factors = _parse_ints(match.group(1))
		if factors:
			split_factors.append(factors)

	vector_widths = [int(x) for x in _VECTOR_ANN_RE.findall(trace_text)]
	unroll_values = [int(x) for x in _UNROLL_ANN_RE.findall(trace_text)]

	for match in _CATEGORICAL_RE.finditer(trace_text):
		candidates = _parse_ints(match.group(1))
		decision_raw = int(match.group(2))
		if candidates:
			if 0 <= decision_raw < len(candidates):
				unroll_values.append(int(candidates[decision_raw]))
			else:
				unroll_values.append(decision_raw)

	thread_tiling_patterns = []
	for factors in split_factors:
		if len(factors) >= 2:
			thread_tiling_patterns.append(tuple(factors[-2:]))

	summary = {
		"sample_perfect_tile_count": len(split_factors),
		"split_factor_signatures": split_factors,
		"vector_widths": vector_widths,
		"unroll_values": unroll_values,
		"reduction_decompose_count": trace_text.count("decompose_reduction("),
		"cache_write_count": trace_text.count("cache_write("),
		"reverse_compute_at_count": trace_text.count("reverse_compute_at("),
		"thread_tiling_patterns": [list(p) for p in thread_tiling_patterns],
		"trace_length_chars": len(trace_text),
	}
	return summary


def _trace_quality_score(summary: Dict[str, Any]) -> float:
	tile_count = int(summary.get("sample_perfect_tile_count", 0))
	vector_width = max(summary.get("vector_widths", [0]) or [0])
	unroll = max(summary.get("unroll_values", [0]) or [0])
	has_cache = 1.0 if int(summary.get("cache_write_count", 0)) > 0 else 0.0
	has_decompose = 1.0 if int(summary.get("reduction_decompose_count", 0)) > 0 else 0.0

	score = 0.0
	score += min(tile_count / 3.0, 1.0) * 0.25
	score += min(vector_width / 64.0, 1.0) * 0.20
	score += min(unroll / 512.0, 1.0) * 0.15
	score += has_cache * 0.20
	score += has_decompose * 0.20
	return round(float(score), 6)


def _normalized_entropy(values: Sequence[Any]) -> float:
	if not values:
		return 0.0
	counts = Counter(values)
	if len(counts) == 1:
		return 0.0
	total = float(sum(counts.values()))
	entropy = 0.0
	for count in counts.values():
		p = count / total
		entropy -= p * math.log(p)
	return float(entropy / math.log(len(counts)))


def _top_patterns(values: Sequence[Any], top_k: int = 5) -> List[Dict[str, Any]]:
	c = Counter(values)
	result = []
	for pattern, freq in c.most_common(top_k):
		result.append({"pattern": pattern, "count": int(freq)})
	return result


def _load_historical_pattern_stats(path: str) -> Dict[str, Any]:
	records = _load_json(path, default=[])
	if not isinstance(records, list):
		return {"num_records": 0, "kernel_stats": {}}

	by_kernel: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
	for row in records:
		if not isinstance(row, dict):
			continue
		kernel = row.get("kernel")
		if kernel not in KERNELS:
			continue
		trace_text = str(row.get("trace", ""))
		M = int(row.get("M", 0))
		by_kernel[kernel].append(
			{
				"M": M,
				"summary": _summarize_trace_text(trace_text),
			}
		)

	kernel_stats: Dict[str, Dict[str, Any]] = {}
	for kernel, items in by_kernel.items():
		vectors: List[int] = []
		unrolls: List[int] = []
		split_signatures: List[Tuple[int, ...]] = []
		thread_patterns: List[Tuple[int, ...]] = []
		m_values: List[int] = []
		cache_presence: List[int] = []
		decompose_presence: List[int] = []

		for item in items:
			summary = item["summary"]
			m_values.append(item["M"])
			vectors.extend(summary.get("vector_widths", []))
			unrolls.extend(summary.get("unroll_values", []))
			split_signatures.extend(tuple(x) for x in summary.get("split_factor_signatures", []))
			thread_patterns.extend(tuple(x) for x in summary.get("thread_tiling_patterns", []))
			cache_presence.append(1 if summary.get("cache_write_count", 0) > 0 else 0)
			decompose_presence.append(1 if summary.get("reduction_decompose_count", 0) > 0 else 0)

		instability = 0.0
		instability += 0.50 * _normalized_entropy(split_signatures)
		instability += 0.20 * _normalized_entropy(vectors)
		instability += 0.20 * _normalized_entropy(unrolls)
		instability += 0.10 * _normalized_entropy(thread_patterns)
		instability = max(0.0, min(1.0, instability))

		kernel_budget_multiplier = 1.0 + 0.30 * instability

		kernel_stats[kernel] = {
			"num_records": len(items),
			"instability": instability,
			"kernel_budget_multiplier": kernel_budget_multiplier,
			"M_values": sorted({int(x) for x in m_values if x > 0}),
			"recurring_patterns": {
				"split_factors": _top_patterns(split_signatures),
				"vector_width": _top_patterns(vectors),
				"unroll_factor": _top_patterns(unrolls),
				"thread_tiling": _top_patterns(thread_patterns),
				"cache_write_rate": (
					float(sum(cache_presence) / len(cache_presence)) if cache_presence else 0.0
				),
				"reduction_decompose_rate": (
					float(sum(decompose_presence) / len(decompose_presence)) if decompose_presence else 0.0
				),
			},
		}

	return {
		"num_records": sum(len(v) for v in by_kernel.values()),
		"kernel_stats": kernel_stats,
	}


def _shape_similarity_multiplier(m_val: int, kernel_stats: Dict[str, Any]) -> float:
	m_values = kernel_stats.get("M_values", [])
	if not m_values:
		return 1.0
	nearest = min(m_values, key=lambda x: abs(x - m_val))
	rel_gap = abs(nearest - m_val) / max(float(m_val), 1.0)
	instability = float(kernel_stats.get("instability", 0.5))
	similarity = math.exp(-3.0 * rel_gap)
	# Stable and similar shapes deserve less tuning budget; unstable shapes get more.
	multiplier = 1.0 - 0.12 * similarity * (1.0 - instability) + 0.10 * (1.0 - similarity) * instability
	return float(max(0.85, min(1.20, multiplier)))


def _apply_kernel_aware_bias(
	config: TuningConfig,
	task: TaskSpec,
	history_stats: Dict[str, Any],
) -> Tuple[TuningConfig, Dict[str, float]]:
	kernel_stats = history_stats.get("kernel_stats", {}).get(task.kernel)
	if not kernel_stats:
		return config, {
			"kernel_multiplier": 1.0,
			"shape_multiplier": 1.0,
			"combined_multiplier": 1.0,
		}

	kernel_mult = float(kernel_stats.get("kernel_budget_multiplier", 1.0))
	shape_mult = _shape_similarity_multiplier(task.M, kernel_stats)
	combined = max(0.75, min(1.35, kernel_mult * shape_mult))

	tuned_global = max(config.num_trials_per_iter, int(round(config.max_trials_global * combined)))
	tuned_per_task = max(config.num_trials_per_iter, int(round(config.max_trials_per_task * combined)))
	tuned_population = max(64, int(round(config.population_size * combined)))
	tuned_design = max(8, int(round(config.design_space_samples * max(0.9, combined))))
	tuned_trace_replay = max(2, int(round(config.trace_replay_count * max(0.9, combined))))

	biased = replace(
		config,
		max_trials_global=tuned_global,
		max_trials_per_task=tuned_per_task,
		population_size=tuned_population,
		design_space_samples=tuned_design,
		trace_replay_count=tuned_trace_replay,
	)

	return biased, {
		"kernel_multiplier": kernel_mult,
		"shape_multiplier": shape_mult,
		"combined_multiplier": combined,
	}


def _query_best_record(database: Any, mod: tvm.ir.IRModule):
	normalized_mod = _normalize_mod(mod["main"])
	return database.query_tuning_record(normalized_mod, TARGET, "main")


def _rigorous_latency(
	mod: tvm.ir.IRModule,
	best_record: Any,
	task: TaskSpec,
	config: TuningConfig,
	seed: int,
) -> Tuple[float, float, List[float]]:
	sch = tvm.tir.Schedule(mod)
	best_record.trace.apply_to_schedule(sch, remove_postproc=False)
	rt_mod = tvm.build(sch.mod, target=TARGET)

	dev = tvm.cpu(0)
	evaluator = rt_mod.time_evaluator(
		"main",
		dev=dev,
		number=config.rigorous_number,
		repeat=config.rigorous_repeat,
		min_repeat_ms=config.rigorous_min_repeat_ms,
	)

	run_means: List[float] = []
	eval_stds: List[float] = []
	for run_id in range(config.rigorous_validation_runs):
		rng = np.random.default_rng(seed + run_id * 9973)
		a_np = rng.normal(0.0, 1.0, size=(task.M, task.K)).astype("float32")
		b_np = rng.normal(0.0, 1.0, size=(task.K, task.N)).astype("float32")
		c_np = np.zeros((task.M, task.N), dtype="float32")
		res = evaluator(tvm.nd.array(a_np, dev), tvm.nd.array(b_np, dev), tvm.nd.array(c_np, dev))
		run_means.append(float(res.mean) * 1e6)
		eval_stds.append(float(res.std) * 1e6)

	latency = float(statistics.fmean(run_means)) if run_means else float("inf")
	if len(run_means) > 1:
		std = float(statistics.pstdev(run_means))
	else:
		std = float(eval_stds[0]) if eval_stds else 0.0
	return latency, std, run_means


def _task_work_dir(base: str, iteration: int, config_hash: str, task: TaskSpec) -> str:
	return os.path.join(
		base,
		f"iter_{iteration:02d}",
		config_hash,
		task.kernel,
		f"M_{task.M}",
	)


def _run_single_task(
	task: TaskSpec,
	config: TuningConfig,
	config_hash: str,
	iteration: int,
	args: argparse.Namespace,
	history_stats: Dict[str, Any],
	benchmark_only: bool,
	persist_best_schedules: bool,
	profile: str,
) -> Dict[str, Any]:
	biased_config, bias_meta = _apply_kernel_aware_bias(config, task, history_stats)
	work_dir = _task_work_dir(args.work_dir_base, iteration, config_hash, task)
	os.makedirs(work_dir, exist_ok=True)

	tune_seed = _deterministic_seed(args.seed, iteration, config_hash, task.kernel, task.M)

	result = {
		"kernel": task.kernel,
		"M": task.M,
		"K": task.K,
		"N": task.N,
		"size_bucket": task.size_bucket,
		"work_dir": work_dir,
		"seed": tune_seed,
		"status": "fail",
		"error": None,
		"effective_config": asdict(biased_config),
		"kernel_aware_budget_bias": bias_meta,
		"trials_before": 0,
		"trials_after": 0,
		"trials_new": 0,
		"latency_us": None,
		"std_us": None,
		"tuning_wall_time_sec": None,
		"rigorous_validation_samples_us": [],
		"trace_summary": {},
		"trace_quality": {},
	}

	mod = matmul_tir(task.M, task.K, task.N)
	before_trials = _count_database_records(work_dir)
	result["trials_before"] = before_trials

	started = time.perf_counter()
	try:
		if benchmark_only:
			database = _load_existing_database(work_dir)
		else:
			database = _run_tuning(mod=mod, work_dir=work_dir, config=biased_config, tune_seed=tune_seed)
		tuning_time = time.perf_counter() - started

		after_trials = _count_database_records(work_dir)
		result["trials_after"] = after_trials
		result["trials_new"] = max(0, after_trials - before_trials)
		result["tuning_wall_time_sec"] = round(float(tuning_time), 6)

		best_record = _query_best_record(database, mod)
		if best_record is None:
			result["error"] = "No tuning record found in database"
			return result

		latency_us, std_us, run_samples = _rigorous_latency(
			mod=mod,
			best_record=best_record,
			task=task,
			config=biased_config,
			seed=tune_seed,
		)
		trace_summary = _summarize_trace_text(str(best_record.trace))
		quality = _trace_quality_score(trace_summary)

		result["latency_us"] = latency_us
		result["std_us"] = std_us
		result["rigorous_validation_samples_us"] = [round(float(x), 6) for x in run_samples]
		result["trace_summary"] = trace_summary
		result["trace_quality"] = {
			"quality_score": quality,
			"sample_perfect_tile_count": int(trace_summary.get("sample_perfect_tile_count", 0)),
			"cache_write_count": int(trace_summary.get("cache_write_count", 0)),
			"reduction_decompose_count": int(trace_summary.get("reduction_decompose_count", 0)),
			"dominant_vector_width": max(trace_summary.get("vector_widths", [0]) or [0]),
			"dominant_unroll": max(trace_summary.get("unroll_values", [0]) or [0]),
		}

		if persist_best_schedules:
			save_best_schedule(
				kernel_name=task.kernel,
				M=task.M,
				K=task.K,
				N=task.N,
				best_record=best_record,
				latency_us=latency_us,
				std_us=std_us,
				profile=profile,
			)

		result["status"] = "ok"
		return result
	except Exception as err:  # pylint: disable=broad-except
		result["error"] = str(err)
		result["tuning_wall_time_sec"] = round(float(time.perf_counter() - started), 6)
		LOGGER.exception(
			"Task failed for kernel=%s M=%d with config=%s",
			task.kernel,
			task.M,
			config.name,
		)
		return result


def _geometric_mean(values: Sequence[float]) -> float:
	vals = [float(v) for v in values if v > 0.0 and math.isfinite(v)]
	if not vals:
		return float("inf")
	return float(math.exp(sum(math.log(v) for v in vals) / len(vals)))


def _baseline_latency_map(experiment: Dict[str, Any]) -> Dict[Tuple[str, int], float]:
	mapping: Dict[Tuple[str, int], float] = {}
	for t in experiment.get("task_results", []):
		if t.get("status") != "ok":
			continue
		mapping[(t["kernel"], int(t["M"]))] = float(t["latency_us"])
	return mapping


def _aggregate_metrics(
	task_results: Sequence[Dict[str, Any]],
	baseline_experiment: Optional[Dict[str, Any]],
	is_baseline: bool,
) -> Dict[str, Any]:
	successful = [t for t in task_results if t.get("status") == "ok"]
	latencies = [float(t["latency_us"]) for t in successful if t.get("latency_us") is not None]
	tuning_times = [
		float(t["tuning_wall_time_sec"]) for t in successful if t.get("tuning_wall_time_sec") is not None
	]
	trials_new = [int(t.get("trials_new", 0)) for t in successful]
	quality_scores = [
		float(t.get("trace_quality", {}).get("quality_score", 0.0)) for t in successful
	]

	latency_geomean = _geometric_mean(latencies) if latencies else float("inf")
	total_tuning_time = float(sum(tuning_times))
	total_trials = int(sum(trials_new))
	avg_trace_quality = float(statistics.fmean(quality_scores)) if quality_scores else 0.0
	all_tasks_succeeded = len(successful) == len(task_results)

	best_trace_summary = {}
	if successful:
		best_trace = max(
			successful,
			key=lambda t: float(t.get("trace_quality", {}).get("quality_score", 0.0)),
		)
		best_trace_summary = {
			"kernel": best_trace.get("kernel"),
			"M": best_trace.get("M"),
			"quality": best_trace.get("trace_quality", {}),
			"summary": best_trace.get("trace_summary", {}),
		}

	aggregate = {
		"num_tasks": len(task_results),
		"num_successful_tasks": len(successful),
		"all_tasks_succeeded": all_tasks_succeeded,
		"latency_geomean_us": latency_geomean,
		"total_tuning_time_sec": total_tuning_time,
		"total_trials": total_trials,
		"avg_trace_quality": round(avg_trace_quality, 6),
		"best_trace_summary": best_trace_summary,
		"latency_retention": 1.0,
		"time_reduction": 1.0,
		"trial_reduction": 1.0,
		"time_fraction": 1.0,
		"trial_fraction": 1.0,
		"latency_loss_pct": 0.0,
		"score": 1.0,
	}

	if is_baseline or baseline_experiment is None:
		return aggregate

	baseline_map = _baseline_latency_map(baseline_experiment)
	retention_ratios: List[float] = []
	for t in successful:
		key = (t["kernel"], int(t["M"]))
		if key not in baseline_map:
			continue
		cand = float(t["latency_us"])
		base = float(baseline_map[key])
		if cand > 0 and base > 0:
			retention_ratios.append(base / cand)

	latency_retention = _geometric_mean(retention_ratios) if retention_ratios else 0.0
	baseline_time = float(baseline_experiment["aggregate"].get("total_tuning_time_sec", 0.0))
	baseline_trials = int(baseline_experiment["aggregate"].get("total_trials", 0))

	time_reduction = baseline_time / max(total_tuning_time, 1e-9)
	trial_reduction = baseline_trials / max(total_trials, 1e-9)
	score = latency_retention * time_reduction
	latency_loss_pct = max(0.0, (1.0 / max(latency_retention, 1e-9) - 1.0) * 100.0)

	aggregate.update(
		{
			"latency_retention": latency_retention,
			"time_reduction": time_reduction,
			"trial_reduction": trial_reduction,
			"time_fraction": 1.0 / max(time_reduction, 1e-9),
			"trial_fraction": 1.0 / max(trial_reduction, 1e-9),
			"latency_loss_pct": latency_loss_pct,
			"score": score,
		}
	)
	return aggregate


def _find_cached_experiment(store: Dict[str, Any], run_id: str) -> Optional[Dict[str, Any]]:
	for exp in store.get("experiments", []):
		if exp.get("run_id") == run_id:
			return copy.deepcopy(exp)
	return None


def _upsert_experiment(store: Dict[str, Any], experiment: Dict[str, Any]) -> None:
	run_id = experiment.get("run_id")
	experiments = store.setdefault("experiments", [])
	for i, exp in enumerate(experiments):
		if exp.get("run_id") == run_id:
			experiments[i] = experiment
			return
	experiments.append(experiment)


def _evaluate_config(
	*,
	mode_label: str,
	iteration: int,
	config: TuningConfig,
	tasks: Sequence[TaskSpec],
	args: argparse.Namespace,
	history_stats: Dict[str, Any],
	store: Dict[str, Any],
	baseline_experiment: Optional[Dict[str, Any]],
	benchmark_only: bool,
	force_rerun: bool,
	is_baseline: bool,
	persist_best_schedules: bool,
	run_tracker: RunProgressTracker,
	profile: str,
) -> Dict[str, Any]:
	config_hash = _config_hash(config)
	tasks_sig = _tasks_signature(tasks)
	mode_suffix = "bench" if benchmark_only else "tune"
	run_id = f"{mode_label}:iter{iteration}:{config_hash}:{tasks_sig}:{mode_suffix}"

	cached = _find_cached_experiment(store, run_id)
	if cached is not None and not force_rerun and not persist_best_schedules:
		cached["aggregate"] = _aggregate_metrics(
			cached.get("task_results", []),
			baseline_experiment=baseline_experiment,
			is_baseline=is_baseline,
		)
		LOGGER.info("Reusing cached experiment: %s", run_id)
		return cached

	LOGGER.info(
		"Evaluating config=%s (state=%s) iteration=%d mode=%s",
		config.name,
		config.state_token,
		iteration,
		mode_label,
	)

	task_results: List[Dict[str, Any]] = []
	total_runs = len(tasks)
	run_tracker.register_config_runs(total_runs)
	for run_idx, task in enumerate(tasks, start=1):
		remaining_runs = total_runs - run_idx
		global_run_idx = run_tracker.next_run_index()
		global_remaining = max(0, run_tracker.planned_runs - global_run_idx)
		LOGGER.info("\n%s", "#" * 120)
		LOGGER.info(
			"Run %d/%d (%d remaining) | global %d/%d (%d remaining) | iteration=%d | mode=%s | config=%s",
			run_idx,
			total_runs,
			remaining_runs,
			global_run_idx,
			run_tracker.planned_runs,
			global_remaining,
			iteration,
			mode_label,
			config.name,
		)
		LOGGER.info(
			"Task: kernel=%s M=%d K=%d N=%d bucket=%s",
			task.kernel,
			task.M,
			task.K,
			task.N,
			task.size_bucket,
		)
		LOGGER.info("%s\n", "#" * 120)
		task_result = _run_single_task(
			task=task,
			config=config,
			config_hash=config_hash,
			iteration=iteration,
			args=args,
			history_stats=history_stats,
			benchmark_only=benchmark_only,
			persist_best_schedules=persist_best_schedules,
			profile=profile,
		)
		run_tracker.mark_run_completed()
		task_results.append(task_result)

	aggregate = _aggregate_metrics(
		task_results,
		baseline_experiment=baseline_experiment,
		is_baseline=is_baseline,
	)

	experiment = {
		"run_id": run_id,
		"timestamp": _utc_now(),
		"mode": mode_label,
		"iteration": iteration,
		"config_name": config.name,
		"config_hash": config_hash,
		"config": asdict(config),
		"is_baseline": is_baseline,
		"benchmark_only": benchmark_only,
		"tasks_signature": tasks_sig,
		"task_results": task_results,
		"aggregate": aggregate,
	}
	_upsert_experiment(store, experiment)
	return experiment


def _pareto_frontier(experiments: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
	candidates = [
		e for e in experiments if e.get("aggregate", {}).get("all_tasks_succeeded", False)
	]
	frontier: List[Dict[str, Any]] = []
	for e in candidates:
		a = e["aggregate"]
		dominated = False
		for other in candidates:
			if other is e:
				continue
			b = other["aggregate"]
			dominates = (
				b.get("latency_retention", 0.0) >= a.get("latency_retention", 0.0)
				and b.get("time_reduction", 0.0) >= a.get("time_reduction", 0.0)
				and (
					b.get("latency_retention", 0.0) > a.get("latency_retention", 0.0)
					or b.get("time_reduction", 0.0) > a.get("time_reduction", 0.0)
				)
			)
			if dominates:
				dominated = True
				break
		if not dominated:
			frontier.append(e)

	frontier.sort(
		key=lambda e: (
			e["aggregate"].get("time_reduction", 0.0),
			e["aggregate"].get("latency_retention", 0.0),
		),
		reverse=True,
	)
	return frontier


def _select_best_candidate(
	candidates: Sequence[Dict[str, Any]],
	target_retention: float,
	max_latency_loss_pct: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
	valid = [
		c
		for c in candidates
		if c.get("aggregate", {}).get("all_tasks_succeeded", False)
		and c["aggregate"].get("latency_loss_pct", 999.0) <= max_latency_loss_pct
	]
	if not valid:
		return None, "No candidate satisfied latency-loss constraint"

	strict_8020 = [
		c
		for c in valid
		if c["aggregate"].get("latency_retention", 0.0) >= target_retention
		and c["aggregate"].get("time_fraction", 999.0) <= 0.30
		and c["aggregate"].get("trial_fraction", 999.0) <= 0.30
	]
	if strict_8020:
		best = max(strict_8020, key=lambda x: x["aggregate"].get("score", -1e9))
		return best, "Selected candidate satisfies latency retention and <=30% time/trial costs"

	retention_ok = [
		c for c in valid if c["aggregate"].get("latency_retention", 0.0) >= target_retention
	]
	if retention_ok:
		best = max(retention_ok, key=lambda x: x["aggregate"].get("score", -1e9))
		return best, "Selected highest score among candidates meeting retention target"

	best = max(valid, key=lambda x: x["aggregate"].get("score", -1e9))
	return best, "Fallback: selected highest score within latency-loss bound"


def _render_recommended_tune_tir_block(config: TuningConfig) -> str:
	return textwrap.dedent(
		f"""
		database = ms.tir_integration.tune_tir(
			mod=mod,
			target=TARGET,
			work_dir=work_dir,
			max_trials_global={config.max_trials_global},
			max_trials_per_task={config.max_trials_per_task},
			num_trials_per_iter={config.num_trials_per_iter},
			builder=ms.builder.LocalBuilder(),
			runner=ms.runner.LocalRunner(
				evaluator_config=ms.runner.EvaluatorConfig(
					number={config.evaluator_number},
					repeat={config.evaluator_repeat},
					min_repeat_ms={config.min_repeat_ms},
				)
			),
			space=ms.space_generator.PostOrderApply(),
			strategy=ms.search_strategy.EvolutionarySearch(
				population_size={config.population_size},
				init_measured_ratio={config.init_measured_ratio},
				init_min_unmeasured={config.design_space_samples},
				max_fail_count={config.trace_replay_count},
				genetic_num_iters={config.genetic_num_iters},
				genetic_mutate_prob={config.mutation_aggressiveness},
				genetic_max_fail_count={config.genetic_max_fail_count},
				eps_greedy={config.eps_greedy},
			),
			seed=seed,
			num_tuning_cores=8,
		)
		"""
	).strip()


def _config_from_dict(raw: Dict[str, Any]) -> TuningConfig:
	baseline = asdict(_state_to_config(PruningState()))
	merged = dict(baseline)
	for key in baseline.keys():
		if key in raw:
			merged[key] = raw[key]
	return TuningConfig(**merged)


def _extract_saved_pruned_config(best_cfg_payload: Dict[str, Any]) -> Optional[TuningConfig]:
	if not isinstance(best_cfg_payload, dict):
		return None

	config_payload = best_cfg_payload.get("config")
	if isinstance(config_payload, dict):
		return _config_from_dict(config_payload)

	# Newer payloads persist this key name.
	selected_config_payload = best_cfg_payload.get("selected_config")
	if isinstance(selected_config_payload, dict):
		return _config_from_dict(selected_config_payload)

	return None


def _safe_latency_value(task_result: Dict[str, Any]) -> Optional[float]:
	return _safe_metric_value(task_result, "latency_us")


def _safe_metric_value(task_result: Dict[str, Any], key: str, allow_zero: bool = False) -> Optional[float]:
	if task_result.get("status") != "ok":
		return None
	value_raw = task_result.get(key)
	if value_raw is None:
		return None
	try:
		value = float(value_raw)
	except (TypeError, ValueError):
		return None
	if not math.isfinite(value):
		return None
	if allow_zero:
		if value < 0.0:
			return None
		return value
	if value <= 0.0:
		return None
	return value


def _index_task_results(experiment: Dict[str, Any]) -> Dict[Tuple[str, int, int, int], Dict[str, Any]]:
	index: Dict[Tuple[str, int, int, int], Dict[str, Any]] = {}
	for row in experiment.get("task_results", []):
		if not isinstance(row, dict):
			continue
		kernel = row.get("kernel")
		if not isinstance(kernel, str):
			continue
		try:
			m_val = int(row.get("M"))
			k_val = int(row.get("K"))
			n_val = int(row.get("N"))
		except (TypeError, ValueError):
			continue
		index[(kernel, m_val, k_val, n_val)] = row
	return index


def _build_latency_comparison_rows(
	baseline_experiment: Dict[str, Any],
	candidate_experiment: Dict[str, Any],
) -> List[Dict[str, Any]]:
	baseline_index = _index_task_results(baseline_experiment)
	candidate_index = _index_task_results(candidate_experiment)
	all_keys = sorted(
		set(baseline_index.keys()) | set(candidate_index.keys()),
		key=lambda item: (item[0], item[1], item[2], item[3]),
	)

	rows: List[Dict[str, Any]] = []
	for kernel, m_val, k_val, n_val in all_keys:
		baseline_task = baseline_index.get((kernel, m_val, k_val, n_val), {})
		candidate_task = candidate_index.get((kernel, m_val, k_val, n_val), {})

		baseline_latency = _safe_latency_value(baseline_task)
		candidate_latency = _safe_latency_value(candidate_task)
		baseline_exec_time = _safe_metric_value(baseline_task, "tuning_wall_time_sec")
		candidate_exec_time = _safe_metric_value(candidate_task, "tuning_wall_time_sec")

		baseline_trials = baseline_task.get("trials_new")
		candidate_trials = candidate_task.get("trials_new")
		if isinstance(baseline_trials, (int, float)):
			baseline_trials = int(baseline_trials)
		else:
			baseline_trials = None
		if isinstance(candidate_trials, (int, float)):
			candidate_trials = int(candidate_trials)
		else:
			candidate_trials = None

		retention = None
		latency_delta_pct = None
		if baseline_latency is not None and candidate_latency is not None and candidate_latency > 0.0:
			retention = baseline_latency / candidate_latency
			latency_delta_pct = ((candidate_latency - baseline_latency) / baseline_latency) * 100.0

		exec_reduction = None
		exec_delta_pct = None
		if (
			baseline_exec_time is not None
			and candidate_exec_time is not None
			and candidate_exec_time > 0.0
			and baseline_exec_time > 0.0
		):
			exec_reduction = baseline_exec_time / candidate_exec_time
			exec_delta_pct = ((candidate_exec_time - baseline_exec_time) / baseline_exec_time) * 100.0

		trial_reduction = None
		if (
			baseline_trials is not None
			and candidate_trials is not None
			and baseline_trials > 0
			and candidate_trials > 0
		):
			trial_reduction = float(baseline_trials) / float(candidate_trials)

		rows.append(
			{
				"kernel": kernel,
				"M": m_val,
				"K": k_val,
				"N": n_val,
				"baseline_latency_us": (
					round(float(baseline_latency), 6) if baseline_latency is not None else None
				),
				"candidate_latency_us": (
					round(float(candidate_latency), 6) if candidate_latency is not None else None
				),
				"baseline_execution_time_sec": (
					round(float(baseline_exec_time), 6) if baseline_exec_time is not None else None
				),
				"candidate_execution_time_sec": (
					round(float(candidate_exec_time), 6) if candidate_exec_time is not None else None
				),
				"baseline_trials": baseline_trials,
				"candidate_trials": candidate_trials,
				"retention": round(float(retention), 6) if retention is not None else None,
				"latency_delta_pct": (
					round(float(latency_delta_pct), 6) if latency_delta_pct is not None else None
				),
				"execution_time_reduction": (
					round(float(exec_reduction), 6) if exec_reduction is not None else None
				),
				"execution_time_delta_pct": (
					round(float(exec_delta_pct), 6) if exec_delta_pct is not None else None
				),
				"trial_reduction": round(float(trial_reduction), 6) if trial_reduction is not None else None,
				"baseline_status": str(baseline_task.get("status", "missing")),
				"candidate_status": str(candidate_task.get("status", "missing")),
			}
		)
	return rows


def _build_kernel_latency_summaries(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
	by_kernel: Dict[str, Dict[str, List[float]]] = defaultdict(
		lambda: {
			"baseline_latency": [],
			"candidate_latency": [],
			"baseline_exec": [],
			"candidate_exec": [],
		}
	)
	for row in rows:
		kernel = str(row.get("kernel", "unknown"))
		baseline_latency = row.get("baseline_latency_us")
		candidate_latency = row.get("candidate_latency_us")
		baseline_exec = row.get("baseline_execution_time_sec")
		candidate_exec = row.get("candidate_execution_time_sec")

		if baseline_latency is not None and float(baseline_latency) > 0.0:
			by_kernel[kernel]["baseline_latency"].append(float(baseline_latency))
		if candidate_latency is not None and float(candidate_latency) > 0.0:
			by_kernel[kernel]["candidate_latency"].append(float(candidate_latency))
		if baseline_exec is not None and float(baseline_exec) > 0.0:
			by_kernel[kernel]["baseline_exec"].append(float(baseline_exec))
		if candidate_exec is not None and float(candidate_exec) > 0.0:
			by_kernel[kernel]["candidate_exec"].append(float(candidate_exec))

	summaries: List[Dict[str, Any]] = []
	for kernel, values in sorted(by_kernel.items()):
		baseline_latencies = values["baseline_latency"]
		candidate_latencies = values["candidate_latency"]
		baseline_execs = values["baseline_exec"]
		candidate_execs = values["candidate_exec"]

		if not baseline_latencies or not candidate_latencies:
			continue

		baseline_geomean = _geometric_mean(baseline_latencies)
		candidate_geomean = _geometric_mean(candidate_latencies)
		baseline_exec_total = float(sum(baseline_execs))
		candidate_exec_total = float(sum(candidate_execs))
		exec_reduction = (
			baseline_exec_total / candidate_exec_total
			if baseline_exec_total > 0.0 and candidate_exec_total > 0.0
			else None
		)
		retention = (
			baseline_geomean / candidate_geomean
			if baseline_geomean > 0.0 and candidate_geomean > 0.0
			else None
		)
		latency_delta_pct = (
			((candidate_geomean - baseline_geomean) / baseline_geomean) * 100.0
			if baseline_geomean > 0.0
			else None
		)
		exec_delta_pct = (
			((candidate_exec_total - baseline_exec_total) / baseline_exec_total) * 100.0
			if baseline_exec_total > 0.0
			else None
		)
		summaries.append({
			"kernel": kernel,
			"num_tasks": min(len(baseline_latencies), len(candidate_latencies)),
			"baseline_geomean_us": round(float(baseline_geomean), 6),
			"candidate_geomean_us": round(float(candidate_geomean), 6),
			"baseline_execution_time_sec": round(float(baseline_exec_total), 6),
			"candidate_execution_time_sec": round(float(candidate_exec_total), 6),
			"retention": round(float(retention), 6) if retention is not None else None,
			"latency_delta_pct": (
				round(float(latency_delta_pct), 6) if latency_delta_pct is not None else None
			),
			"execution_time_reduction": (
				round(float(exec_reduction), 6) if exec_reduction is not None else None
			),
			"execution_time_delta_pct": (
				round(float(exec_delta_pct), 6) if exec_delta_pct is not None else None
			),
		})
	return summaries


def _build_overall_comparison_summary(
	rows: Sequence[Dict[str, Any]],
	baseline_total_tune_tir_time_sec: Optional[float],
	candidate_total_tune_tir_time_sec: Optional[float],
) -> Dict[str, Any]:
	baseline_lat = [
		float(r["baseline_latency_us"])
		for r in rows
		if r.get("baseline_latency_us") is not None and float(r["baseline_latency_us"]) > 0.0
	]
	candidate_lat = [
		float(r["candidate_latency_us"])
		for r in rows
		if r.get("candidate_latency_us") is not None and float(r["candidate_latency_us"]) > 0.0
	]
	baseline_lat_gm = _geometric_mean(baseline_lat) if baseline_lat else None
	candidate_lat_gm = _geometric_mean(candidate_lat) if candidate_lat else None

	latency_retention = (
		baseline_lat_gm / candidate_lat_gm
		if baseline_lat_gm is not None and candidate_lat_gm is not None and candidate_lat_gm > 0.0
		else None
	)
	latency_delta_pct = (
		((candidate_lat_gm - baseline_lat_gm) / baseline_lat_gm) * 100.0
		if baseline_lat_gm is not None and baseline_lat_gm > 0.0 and candidate_lat_gm is not None
		else None
	)
	execution_time_reduction = (
		baseline_total_tune_tir_time_sec / candidate_total_tune_tir_time_sec
		if baseline_total_tune_tir_time_sec is not None
		and candidate_total_tune_tir_time_sec is not None
		and baseline_total_tune_tir_time_sec > 0.0
		and candidate_total_tune_tir_time_sec > 0.0
		else None
	)
	execution_time_delta_pct = (
		((candidate_total_tune_tir_time_sec - baseline_total_tune_tir_time_sec)
		 / baseline_total_tune_tir_time_sec) * 100.0
		if baseline_total_tune_tir_time_sec is not None
		and candidate_total_tune_tir_time_sec is not None
		and baseline_total_tune_tir_time_sec > 0.0
		else None
	)
	return {
		"num_shapes": len(rows),
		"num_comparable_latency_shapes": min(len(baseline_lat), len(candidate_lat)),
		"baseline_latency_geomean_us": (
			round(float(baseline_lat_gm), 6) if baseline_lat_gm is not None else None
		),
		"candidate_latency_geomean_us": (
			round(float(candidate_lat_gm), 6) if candidate_lat_gm is not None else None
		),
		"baseline_total_tune_tir_time_sec": (
			round(float(baseline_total_tune_tir_time_sec), 6)
			if baseline_total_tune_tir_time_sec is not None
			else None
		),
		"candidate_total_tune_tir_time_sec": (
			round(float(candidate_total_tune_tir_time_sec), 6)
			if candidate_total_tune_tir_time_sec is not None
			else None
		),
		"baseline_total_tune_tir_time_human": _fmt_time_human(baseline_total_tune_tir_time_sec),
		"candidate_total_tune_tir_time_human": _fmt_time_human(candidate_total_tune_tir_time_sec),
		"baseline_execution_time_sec": (
			round(float(baseline_total_tune_tir_time_sec), 6)
			if baseline_total_tune_tir_time_sec is not None
			else None
		),
		"candidate_execution_time_sec": (
			round(float(candidate_total_tune_tir_time_sec), 6)
			if candidate_total_tune_tir_time_sec is not None
			else None
		),
		"latency_retention": round(float(latency_retention), 6) if latency_retention is not None else None,
		"latency_delta_pct": (
			round(float(latency_delta_pct), 6) if latency_delta_pct is not None else None
		),
		"execution_time_reduction": (
			round(float(execution_time_reduction), 6)
			if execution_time_reduction is not None
			else None
		),
		"execution_time_delta_pct": (
			round(float(execution_time_delta_pct), 6)
			if execution_time_delta_pct is not None
			else None
		),
	}


def _render_text_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
	if not headers:
		return ""

	# Compute visible widths (strip ANSI sequences) so colored cells align correctly
	widths: List[int] = [ _visible_len(str(h)) for h in headers ]
	for row in rows:
		for idx, cell in enumerate(row):
			widths[idx] = max(widths[idx], _visible_len(str(cell)))

	def pad_cell(cell: str, width: int) -> str:
		s = str(cell)
		pad = max(0, width - _visible_len(s))
		return s + " " * pad

	header_row = " | ".join(pad_cell(str(h), widths[idx]) for idx, h in enumerate(headers))
	separator = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
	data_rows = [
		" | ".join(pad_cell(str(cell), widths[idx]) for idx, cell in enumerate(row))
		for row in rows
	]
	return "\n".join([header_row, separator] + data_rows)


def _fmt_table_number(value: Optional[float], precision: int) -> str:
	if value is None:
		return "n/a"
	if not math.isfinite(float(value)):
		return "n/a"
	return f"{float(value):.{precision}f}"


def _fmt_table_percent(value: Optional[float]) -> str:
	if value is None:
		return "n/a"
	if not math.isfinite(float(value)):
		return "n/a"
	return f"{float(value):+.2f}%"


def _fmt_table_percent_unsigned(value: Optional[float]) -> str:
	if value is None:
		return "n/a"
	if not math.isfinite(float(value)):
		return "n/a"
	return f"{float(value):.2f}%"


def _fmt_time_human(seconds: Optional[float]) -> str:
	if seconds is None:
		return "n/a"
	try:
		s = float(seconds)
	except (TypeError, ValueError):
		return "n/a"
	if not math.isfinite(s):
		return "n/a"
	mins = int(max(0.0, s) // 60)
	rem = max(0.0, s) - mins * 60
	return f"{mins}m {rem:.3f}s"


def _format_latency_rows_table(rows: Sequence[Dict[str, Any]]) -> str:
	if not rows:
		return "No per-task latency rows were available for comparison."

	headers = [
		"shape",
		"baseline_latency_us",
		"candidate_latency_us",
		"baseline_task_tune_time",
		"candidate_task_tune_time",
		"latency_retention",
		"exec_time_reduction",
	]
	data_rows: List[List[str]] = []
	for row in rows:
		shape_label = (
			f"{row.get('kernel', '')}[M={row.get('M', '')}]"
		)

		# Decide coloring: candidate better => green, worse => red. None => no color.
		base_lat = row.get("baseline_latency_us")
		cand_lat = row.get("candidate_latency_us")
		base_exec = row.get("baseline_execution_time_sec")
		cand_exec = row.get("candidate_execution_time_sec")

		# compare latencies: lower is better; treat near-equal as neutral
		cand_lat_better = None
		base_lat_better = None
		if base_lat is not None and cand_lat is not None:
			try:
				b = float(base_lat)
				c = float(cand_lat)
				if abs(b - c) <= 1e-9:
					cand_lat_better = None
					base_lat_better = None
				elif c < b:
					cand_lat_better = True
					base_lat_better = False
				else:
					cand_lat_better = False
					base_lat_better = True
			except Exception:
				cand_lat_better = None
				base_lat_better = None

		cand_exec_better = None
		base_exec_better = None
		if base_exec is not None and cand_exec is not None:
			try:
				b = float(base_exec)
				c = float(cand_exec)
				if abs(b - c) <= 1e-9:
					cand_exec_better = None
					base_exec_better = None
				elif c < b:
					cand_exec_better = True
					base_exec_better = False
				else:
					cand_exec_better = False
					base_exec_better = True
			except Exception:
				cand_exec_better = None
				base_exec_better = None

		retention_val = row.get("retention")
		exec_red_val = row.get("execution_time_reduction")
		retention_good = None
		if retention_val is not None:
			try:
				retention_good = float(retention_val) >= 1.0
			except Exception:
				retention_good = None
		exec_red_good = None
		if exec_red_val is not None:
			try:
				exec_red_good = float(exec_red_val) >= 1.0
			except Exception:
				exec_red_good = None

		data_rows.append([
			shape_label,
			_maybe_color(_fmt_table_number(base_lat, 3), base_lat_better),
			_maybe_color(_fmt_table_number(cand_lat, 3), cand_lat_better),
			_maybe_color(_fmt_time_human(base_exec), base_exec_better),
			_maybe_color(_fmt_time_human(cand_exec), cand_exec_better),
			_maybe_color(_arrow_prefix(retention_good) + _fmt_table_number(retention_val, 3), retention_good),
			_maybe_color(_arrow_prefix(exec_red_good) + _fmt_table_number(exec_red_val, 3), exec_red_good),
		])
	return _render_text_table(headers, data_rows)


def _format_kernel_summary_table(rows: Sequence[Dict[str, Any]]) -> str:
	if not rows:
		return "No kernel-level summary rows were available for comparison."

	headers = [
		"kernel",
		"shapes",
		"baseline_lat_gm_us",
		"candidate_lat_gm_us",
		"baseline_tune_time",
		"candidate_tune_time",
		"latency_retention",
		"exec_time_reduction",
	]
	data_rows: List[List[str]] = []
	for row in rows:
		# symmetric comparisons for baseline vs candidate
		bg = row.get("baseline_geomean_us")
		cg = row.get("candidate_geomean_us")
		base_geomean_better = None
		cand_geomean_better = None
		if bg is not None and cg is not None:
			try:
				b = float(bg)
				c = float(cg)
				if abs(b - c) <= 1e-9:
					base_geomean_better = None
					cand_geomean_better = None
				elif c < b:
					cand_geomean_better = True
					base_geomean_better = False
				else:
					cand_geomean_better = False
					base_geomean_better = True
			except Exception:
				base_geomean_better = None
				cand_geomean_better = None

		eb = row.get("baseline_execution_time_sec")
		ec = row.get("candidate_execution_time_sec")
		base_exec_better = None
		cand_exec_better = None
		if eb is not None and ec is not None:
			try:
				b = float(eb)
				c = float(ec)
				if abs(b - c) <= 1e-9:
					base_exec_better = None
					cand_exec_better = None
				elif c < b:
					cand_exec_better = True
					base_exec_better = False
				else:
					cand_exec_better = False
					base_exec_better = True
			except Exception:
				base_exec_better = None
				cand_exec_better = None

		data_rows.append([
			str(row.get("kernel", "")),
			str(row.get("num_tasks", "")),
			_maybe_color(_fmt_table_number(row.get("baseline_geomean_us"), 3), base_geomean_better),
			_maybe_color(_fmt_table_number(row.get("candidate_geomean_us"), 3), cand_geomean_better),
			_maybe_color(_fmt_time_human(row.get("baseline_execution_time_sec")), base_exec_better),
			_maybe_color(_fmt_time_human(row.get("candidate_execution_time_sec")), cand_exec_better),
			_maybe_color(_arrow_prefix(None if row.get("retention") is None else (float(row.get("retention") or 0.0) >= 1.0)) + _fmt_table_number(row.get("retention"), 4), None if row.get("retention") is None else float(row.get("retention") or 0.0) >= 1.0),
			_maybe_color(_arrow_prefix(None if row.get("execution_time_reduction") is None else (float(row.get("execution_time_reduction") or 0.0) >= 1.0)) + _fmt_table_number(row.get("execution_time_reduction"), 4), None if row.get("execution_time_reduction") is None else float(row.get("execution_time_reduction") or 0.0) >= 1.0),
		])
	return _render_text_table(headers, data_rows)


def _format_overall_summary_table(summary: Dict[str, Any]) -> str:
	headers = [
		"num_shapes",
		"base_lat_gm_us",
		"cand_lat_gm_us",
		"base_total_tune_tir",
		"cand_total_tune_tir",
		"base_total_tune_s",
		"cand_total_tune_s",
		"latency_retention",
		"exec_time_reduction",
	]
	# symmetric comparisons
	base_lat = summary.get("baseline_latency_geomean_us")
	cand_lat = summary.get("candidate_latency_geomean_us")
	base_lat_better = None
	cand_lat_better = None
	if base_lat is not None and cand_lat is not None:
		try:
			b = float(base_lat)
			c = float(cand_lat)
			if abs(b - c) <= 1e-9:
				base_lat_better = None
				cand_lat_better = None
			elif c < b:
				cand_lat_better = True
				base_lat_better = False
			else:
				cand_lat_better = False
				base_lat_better = True
		except Exception:
			base_lat_better = None
			cand_lat_better = None

	base_time = summary.get("baseline_total_tune_tir_time_sec")
	cand_time = summary.get("candidate_total_tune_tir_time_sec")
	base_time_better = None
	cand_time_better = None
	if base_time is not None and cand_time is not None:
		try:
			b = float(base_time)
			c = float(cand_time)
			if abs(b - c) <= 1e-9:
				base_time_better = None
				cand_time_better = None
			elif c < b:
				cand_time_better = True
				base_time_better = False
			else:
				cand_time_better = False
				base_time_better = True
		except Exception:
			base_time_better = None
			cand_time_better = None

	rows = [
		[
			str(summary.get("num_shapes", "")),
			_maybe_color(_fmt_table_number(summary.get("baseline_latency_geomean_us"), 3), base_lat_better),
			_maybe_color(_fmt_table_number(summary.get("candidate_latency_geomean_us"), 3), cand_lat_better),
			_maybe_color(str(summary.get("baseline_total_tune_tir_time_human", "n/a")), base_time_better),
			_maybe_color(str(summary.get("candidate_total_tune_tir_time_human", "n/a")), cand_time_better),
			_maybe_color(_fmt_table_number(summary.get("baseline_total_tune_tir_time_sec"), 3), base_time_better),
			_maybe_color(_fmt_table_number(summary.get("candidate_total_tune_tir_time_sec"), 3), cand_time_better),
			_maybe_color(_arrow_prefix(None if summary.get("latency_retention") is None else (float(summary.get("latency_retention") or 0.0) >= 1.0)) + _fmt_table_number(summary.get("latency_retention"), 4), None if summary.get("latency_retention") is None else float(summary.get("latency_retention") or 0.0) >= 1.0),
			_maybe_color(_arrow_prefix(None if summary.get("execution_time_reduction") is None else (float(summary.get("execution_time_reduction") or 0.0) >= 1.0)) + _fmt_table_number(summary.get("execution_time_reduction"), 4), None if summary.get("execution_time_reduction") is None else float(summary.get("execution_time_reduction") or 0.0) >= 1.0),
		]
	]
	return _render_text_table(headers, rows)


def _persist_comparison_result(
	comparison: Dict[str, Any],
	args: argparse.Namespace,
	tasks: Sequence[TaskSpec],
) -> str:
	store = _load_json(COMPARE_RESULTS_FILE, default={"metadata": {}, "comparisons": []})
	store.setdefault("metadata", {})
	store.setdefault("comparisons", [])

	compare_id = _stable_hash(
		{
			"timestamp": comparison.get("timestamp"),
			"mode": comparison.get("mode"),
			"baseline_config": comparison.get("baseline", {}).get("config", {}),
			"candidate_config": comparison.get("candidate", {}).get("config", {}),
		}
	)

	entry = copy.deepcopy(comparison)
	entry["compare_id"] = compare_id
	entry["inputs"] = {
		"best_config_path": args.best_config_path,
		"benchmark_only": bool(args.benchmark_only),
		"force_rerun": bool(args.force_rerun),
		"task_count": len(tasks),
	}

	store["comparisons"].append(entry)
	store["latest_compare"] = entry
	store["metadata"].update(
		{
			"last_updated": _utc_now(),
			"target": str(TARGET),
			"schema": "comparison_results.v2",
		}
	)
	_write_json(COMPARE_RESULTS_FILE, store)
	return compare_id


def _safe_total_tuning_time_sec(experiment: Optional[Dict[str, Any]]) -> Optional[float]:
	if not isinstance(experiment, dict):
		return None
	aggregate = experiment.get("aggregate", {})
	if not isinstance(aggregate, dict):
		return None
	raw_value = aggregate.get("total_tuning_time_sec")
	if raw_value is None:
		return None
	try:
		value = float(raw_value)
	except (TypeError, ValueError):
		return None
	if not math.isfinite(value) or value < 0.0:
		return None
	return value


def _lookup_historical_tune_series_time_sec(
	store: Dict[str, Any],
	config_hash: str,
	tasks_signature: str,
) -> Optional[float]:
	best_time: Optional[float] = None
	best_timestamp = ""
	for experiment in store.get("experiments", []):
		if not isinstance(experiment, dict):
			continue
		if str(experiment.get("config_hash", "")) != config_hash:
			continue
		if str(experiment.get("tasks_signature", "")) != tasks_signature:
			continue
		if bool(experiment.get("benchmark_only", False)):
			continue

		time_sec = _safe_total_tuning_time_sec(experiment)
		if time_sec is None:
			continue

		timestamp = str(experiment.get("timestamp", ""))
		if best_time is None or timestamp > best_timestamp:
			best_time = time_sec
			best_timestamp = timestamp

	return best_time


def _resolve_total_tune_series_time_sec(
	store: Dict[str, Any],
	experiment: Dict[str, Any],
) -> Optional[float]:
	current_time = _safe_total_tuning_time_sec(experiment)
	if not bool(experiment.get("benchmark_only", False)):
		return current_time

	config_hash = str(experiment.get("config_hash", ""))
	tasks_signature = str(experiment.get("tasks_signature", ""))
	historical_time = _lookup_historical_tune_series_time_sec(
		store=store,
		config_hash=config_hash,
		tasks_signature=tasks_signature,
	)
	if historical_time is not None:
		return historical_time
	return current_time


def _run_pruning_iteration(
	*,
	iteration: int,
	tasks: Sequence[TaskSpec],
	args: argparse.Namespace,
	history_stats: Dict[str, Any],
	store: Dict[str, Any],
	run_tracker: RunProgressTracker,
	profile: str,
) -> Dict[str, Any]:
	baseline_state = PruningState()
	baseline_config = _state_to_config(baseline_state)

	baseline_exp = _evaluate_config(
		mode_label="pruning",
		iteration=iteration,
		config=baseline_config,
		tasks=tasks,
		args=args,
		history_stats=history_stats,
		store=store,
		baseline_experiment=None,
		benchmark_only=args.benchmark_only,
		force_rerun=args.force_rerun,
		is_baseline=True,
		persist_best_schedules=False,
		run_tracker=run_tracker,
		profile=profile,
	)

	evaluated: List[Dict[str, Any]] = [baseline_exp]
	current_state = baseline_state
	current_exp = baseline_exp
	seen_states = {baseline_state}
	stop_reason = "Reached maximum pruning steps"
	accepted_path = [baseline_exp]

	latency_retention_floor = 1.0 - (args.max_latency_loss_pct / 100.0)

	for step in range(1, args.max_pruning_steps + 1):
		candidates = _next_candidate_states(current_state, seen_states)
		if not candidates:
			stop_reason = "No unexplored candidate states remain"
			break

		LOGGER.info("Pruning step %d: evaluating %d neighboring states", step, len(candidates))
		step_results: List[Tuple[PruningState, Dict[str, Any]]] = []
		total_candidates = len(candidates)
		for candidate_idx, state in enumerate(candidates, start=1):
			remaining_candidates = total_candidates - candidate_idx
			config = _state_to_config(state)
			LOGGER.info(
				"Pruning candidate %d/%d (%d remaining) | step=%d | state=%s | config=%s",
				candidate_idx,
				total_candidates,
				remaining_candidates,
				step,
				state.token(),
				config.name,
			)
			exp = _evaluate_config(
				mode_label="pruning",
				iteration=iteration,
				config=config,
				tasks=tasks,
				args=args,
				history_stats=history_stats,
				store=store,
				baseline_experiment=baseline_exp,
				benchmark_only=args.benchmark_only,
				force_rerun=args.force_rerun,
				is_baseline=False,
				persist_best_schedules=False,
				run_tracker=run_tracker,
				profile=profile,
			)
			evaluated.append(exp)
			step_results.append((state, exp))
			seen_states.add(state)

		valid = [
			(state, exp)
			for state, exp in step_results
			if exp["aggregate"].get("all_tasks_succeeded", False)
			and exp["aggregate"].get("latency_retention", 0.0) >= latency_retention_floor
		]
		if not valid:
			stop_reason = (
				"All neighbor candidates violated latency retention floor "
				f"({latency_retention_floor:.3f})"
			)
			break

		best_state, best_exp = max(valid, key=lambda item: item[1]["aggregate"].get("score", -1e9))
		score_gain = best_exp["aggregate"].get("score", 0.0) - current_exp["aggregate"].get("score", 0.0)
		time_gain = best_exp["aggregate"].get("time_reduction", 0.0) - current_exp["aggregate"].get("time_reduction", 0.0)

		LOGGER.info(
			"Best candidate at step %d: %s | score=%.4f retention=%.4f time_reduction=%.4f",
			step,
			best_exp.get("config_name"),
			best_exp["aggregate"].get("score", 0.0),
			best_exp["aggregate"].get("latency_retention", 0.0),
			best_exp["aggregate"].get("time_reduction", 0.0),
		)

		if score_gain < args.min_score_gain and time_gain < args.min_time_reduction_gain:
			stop_reason = (
				"Diminishing returns: insufficient score/time improvement "
				f"(score_gain={score_gain:.4f}, time_gain={time_gain:.4f})"
			)
			break

		current_state = best_state
		current_exp = best_exp
		accepted_path.append(best_exp)

		if best_exp["aggregate"].get("latency_loss_pct", 0.0) > args.max_latency_loss_pct:
			stop_reason = "Latency loss exceeded pruning threshold"
			break

	non_baseline = [e for e in evaluated if not e.get("is_baseline", False)]
	chosen, selection_reason = _select_best_candidate(
		non_baseline,
		target_retention=args.target_retention,
		max_latency_loss_pct=args.max_latency_loss_pct,
	)
	if chosen is None:
		chosen = baseline_exp
		selection_reason = "Fallback to baseline because no valid pruned candidate was found"

	pareto = _pareto_frontier(non_baseline)

	return {
		"iteration": iteration,
		"baseline": baseline_exp,
		"evaluated": evaluated,
		"accepted_path": accepted_path,
		"stop_reason": stop_reason,
		"selected": chosen,
		"selection_reason": selection_reason,
		"pareto_frontier": pareto,
	}


def _select_global_best(
	iteration_outputs: Sequence[Dict[str, Any]],
	target_retention: float,
	max_latency_loss_pct: float,
) -> Tuple[Dict[str, Any], Dict[str, Any], str]:
	grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
	for out in iteration_outputs:
		for exp in out.get("evaluated", []):
			if exp.get("is_baseline", False):
				continue
			if not exp.get("aggregate", {}).get("all_tasks_succeeded", False):
				continue
			grouped[exp["config_hash"]].append(exp)

	if not grouped:
		baseline = iteration_outputs[0]["baseline"]
		return baseline, {"config_hash": baseline["config_hash"], "stats": []}, "No successful pruned configs"

	scored: List[Dict[str, Any]] = []
	for config_hash, exps in grouped.items():
		retentions = [float(e["aggregate"].get("latency_retention", 0.0)) for e in exps]
		time_reds = [float(e["aggregate"].get("time_reduction", 0.0)) for e in exps]
		trial_reds = [float(e["aggregate"].get("trial_reduction", 0.0)) for e in exps]
		losses = [float(e["aggregate"].get("latency_loss_pct", 999.0)) for e in exps]

		mean_ret = float(statistics.fmean(retentions))
		mean_time = float(statistics.fmean(time_reds))
		mean_trial = float(statistics.fmean(trial_reds))
		mean_score = mean_ret * mean_time
		stability_penalty = float(statistics.pstdev(retentions)) if len(retentions) > 1 else 0.0
		stability_score = mean_score - 0.10 * stability_penalty

		scored.append(
			{
				"config_hash": config_hash,
				"prototype": exps[0],
				"num_observations": len(exps),
				"mean_retention": mean_ret,
				"mean_time_reduction": mean_time,
				"mean_trial_reduction": mean_trial,
				"mean_latency_loss_pct": float(statistics.fmean(losses)),
				"stability_penalty": stability_penalty,
				"stability_score": stability_score,
			}
		)

	valid = [
		s
		for s in scored
		if s["mean_latency_loss_pct"] <= max_latency_loss_pct
		and s["mean_retention"] >= target_retention
	]
	if valid:
		best = max(valid, key=lambda x: x["stability_score"])
		return best["prototype"], {"stats": scored}, "Stable best config meeting retention target"

	fallback = [s for s in scored if s["mean_latency_loss_pct"] <= max_latency_loss_pct]
	if fallback:
		best = max(fallback, key=lambda x: x["stability_score"])
		return best["prototype"], {"stats": scored}, "Stable best config within loss bound"

	best = max(scored, key=lambda x: x["stability_score"])
	return best["prototype"], {"stats": scored}, "Fallback: highest stability score"


def _persist_best_schedules_for_selection(
	selected_experiment: Dict[str, Any],
	tasks: Sequence[TaskSpec],
	args: argparse.Namespace,
	history_stats: Dict[str, Any],
	store: Dict[str, Any],
	run_tracker: RunProgressTracker,
	profile: str,
) -> None:
	config = _config_from_dict(selected_experiment["config"])
	iteration = int(selected_experiment.get("iteration", 1))
	LOGGER.info(
		"Persisting best schedules for selected config=%s iteration=%d",
		config.name,
		iteration,
	)

	_evaluate_config(
		mode_label="final_persist",
		iteration=iteration,
		config=config,
		tasks=tasks,
		args=args,
		history_stats=history_stats,
		store=store,
		baseline_experiment=None,
		benchmark_only=True,
		force_rerun=True,
		is_baseline=False,
		persist_best_schedules=True,
		run_tracker=run_tracker,
		profile=profile,
	)


def _run_compare_mode(
	*,
	tasks: Sequence[TaskSpec],
	args: argparse.Namespace,
	history_stats: Dict[str, Any],
	store: Dict[str, Any],
	run_tracker: RunProgressTracker,
	profile: str,
) -> Dict[str, Any]:
	baseline_config = _state_to_config(PruningState())

	best_cfg_payload = _load_json(args.best_config_path, default={})
	pruned_config = _extract_saved_pruned_config(best_cfg_payload)
	if pruned_config is None:
		raise FileNotFoundError(
			f"Unable to load pruned config from {args.best_config_path}. "
			"Expected key `config` or `selected_config`. "
			"Run pruning mode first to generate best_pruned_config.json"
		)

	baseline_exp = _evaluate_config(
		mode_label="compare",
		iteration=1,
		config=baseline_config,
		tasks=tasks,
		args=args,
		history_stats=history_stats,
		store=store,
		baseline_experiment=None,
		benchmark_only=args.benchmark_only,
		force_rerun=args.force_rerun,
		is_baseline=True,
		persist_best_schedules=False,
		run_tracker=run_tracker,
		profile=profile,
	)

	pruned_exp = _evaluate_config(
		mode_label="compare",
		iteration=1,
		config=pruned_config,
		tasks=tasks,
		args=args,
		history_stats=history_stats,
		store=store,
		baseline_experiment=baseline_exp,
		benchmark_only=args.benchmark_only,
		force_rerun=args.force_rerun,
		is_baseline=False,
		persist_best_schedules=False,
		run_tracker=run_tracker,
		profile=profile,
	)

	latency_rows = _build_latency_comparison_rows(
		baseline_experiment=baseline_exp,
		candidate_experiment=pruned_exp,
	)
	kernel_summaries = _build_kernel_latency_summaries(latency_rows)
	baseline_total_tune_tir_time_sec = _resolve_total_tune_series_time_sec(store, baseline_exp)
	candidate_total_tune_tir_time_sec = _resolve_total_tune_series_time_sec(store, pruned_exp)
	baseline_aggregate = copy.deepcopy(baseline_exp.get("aggregate", {}))
	candidate_aggregate = copy.deepcopy(pruned_exp.get("aggregate", {}))
	if isinstance(baseline_aggregate, dict):
		baseline_aggregate.pop("score", None)
	if isinstance(candidate_aggregate, dict):
		candidate_aggregate.pop("score", None)
	overall_summary = _build_overall_comparison_summary(
		latency_rows,
		baseline_total_tune_tir_time_sec=baseline_total_tune_tir_time_sec,
		candidate_total_tune_tir_time_sec=candidate_total_tune_tir_time_sec,
	)

	LOGGER.info("Per-task latency comparison table:\n%s", _format_latency_rows_table(latency_rows))
	LOGGER.info("Per-kernel geomean comparison table:\n%s", _format_kernel_summary_table(kernel_summaries))
	LOGGER.info(
		"Full tune_tir series time | baseline=%s (%ss) | candidate=%s (%ss)",
		_fmt_time_human(baseline_total_tune_tir_time_sec),
		_fmt_table_number(baseline_total_tune_tir_time_sec, 3),
		_fmt_time_human(candidate_total_tune_tir_time_sec),
		_fmt_table_number(candidate_total_tune_tir_time_sec, 3),
	)
	LOGGER.info("Overall comparison summary:\n%s", _format_overall_summary_table(overall_summary))

	comparison = {
		"timestamp": _utc_now(),
		"mode": "benchmark-only" if args.benchmark_only else "compare-against-baseline",
		"baseline": {
			"config": baseline_exp["config"],
			"aggregate": baseline_aggregate,
			"total_tune_tir_time_sec": (
				round(float(baseline_total_tune_tir_time_sec), 6)
				if baseline_total_tune_tir_time_sec is not None
				else None
			),
			"total_tune_tir_time_human": _fmt_time_human(baseline_total_tune_tir_time_sec),
		},
		"candidate": {
			"config": pruned_exp["config"],
			"aggregate": candidate_aggregate,
			"total_tune_tir_time_sec": (
				round(float(candidate_total_tune_tir_time_sec), 6)
				if candidate_total_tune_tir_time_sec is not None
				else None
			),
			"total_tune_tir_time_human": _fmt_time_human(candidate_total_tune_tir_time_sec),
		},
		"config_tune_tir_series_time": {
			"baseline_sec": (
				round(float(baseline_total_tune_tir_time_sec), 6)
				if baseline_total_tune_tir_time_sec is not None
				else None
			),
			"baseline_human": _fmt_time_human(baseline_total_tune_tir_time_sec),
			"candidate_sec": (
				round(float(candidate_total_tune_tir_time_sec), 6)
				if candidate_total_tune_tir_time_sec is not None
				else None
			),
			"candidate_human": _fmt_time_human(candidate_total_tune_tir_time_sec),
		},
		"shape_comparison_table": latency_rows,
		"latency_comparison_table": latency_rows,
		"kernel_latency_summary": kernel_summaries,
		"overall_summary": overall_summary,
	}
	return comparison


def _build_best_pruned_payload(
	*,
	selected_experiment: Dict[str, Any],
	iteration_outputs: Sequence[Dict[str, Any]],
	tasks: Sequence[TaskSpec],
	history_stats: Dict[str, Any],
	selection_reason: str,
) -> Dict[str, Any]:
	config = _config_from_dict(selected_experiment["config"])
	pareto_candidates = []
	for out in iteration_outputs:
		for exp in out.get("pareto_frontier", []):
			pareto_candidates.append(
				{
					"iteration": exp.get("iteration"),
					"config_name": exp.get("config_name"),
					"config_hash": exp.get("config_hash"),
					"latency_retention": exp.get("aggregate", {}).get("latency_retention"),
					"time_reduction": exp.get("aggregate", {}).get("time_reduction"),
					"trial_reduction": exp.get("aggregate", {}).get("trial_reduction"),
					"score": exp.get("aggregate", {}).get("score"),
				}
			)

	recommended_block = _render_recommended_tune_tir_block(config)

	return {
		"timestamp": _utc_now(),
		"target": str(TARGET),
		"objective": {
			"latency_retention_goal": "80%-90%",
			"tuning_cost_goal": "<=20%-30% trials/time",
		},
		"benchmark_subset": [asdict(t) for t in tasks],
		"history_summary": {
			"records_loaded": history_stats.get("num_records", 0),
			"kernel_stats": history_stats.get("kernel_stats", {}),
		},
		"selected_config": asdict(config),
		"selected_metrics": selected_experiment.get("aggregate", {}),
		"selection_reason": selection_reason,
		"recommended_tune_tir_block": recommended_block,
		"pareto_frontier": pareto_candidates,
	}


def main() -> int:
	args = _parse_args()

	os.makedirs(args.work_dir_base, exist_ok=True)
	_configure_logging(LOG_FILE)

	LOGGER.info("Starting 80/20 MetaSchedule tuner")
	LOGGER.info("Target: %s", TARGET)
	LOGGER.info("Work dir base: %s", args.work_dir_base)

	random.seed(args.seed)
	np.random.seed(args.seed)

	profile = resolve_profile(args.profile)
	LOGGER.info("Profile: %s", profile)

	if not ensure_data_aggregator_connection_or_prompt("metaschedule_8020_tuner"):
		return 1

	selected_kernels = _resolve_kernel_list(args)
	tasks = _build_tasks(selected_kernels)
	LOGGER.info("Kernels: %s", ", ".join(selected_kernels))
	LOGGER.info("Representative subset tasks: %d", len(tasks))
	for task in tasks:
		LOGGER.info("  - kernel=%s M=%d K=%d N=%d (%s)", task.kernel, task.M, task.K, task.N, task.size_bucket)

	history_stats = _load_historical_pattern_stats(BEST_SCHEDULES_FILE)
	LOGGER.info("Loaded %d historical best schedules for heuristic biasing", history_stats.get("num_records", 0))

	store = _load_json(PRUNING_EXPERIMENTS_FILE, default={"metadata": {}, "experiments": []})
	run_tracker = RunProgressTracker()
	store.setdefault("metadata", {})
	store["metadata"].update(
		{
			"last_run_timestamp": _utc_now(),
			"target": str(TARGET),
			"work_dir_base": args.work_dir_base,
			"seed": args.seed,
		}
	)

	if args.benchmark_only or args.compare_against_baseline:
		comparison = _run_compare_mode(
			tasks=tasks,
			args=args,
			history_stats=history_stats,
			store=store,
			run_tracker=run_tracker,
			profile=profile,
		)
		compare_id = _persist_comparison_result(comparison, args=args, tasks=tasks)
		store["latest_compare_ref"] = {
			"compare_id": compare_id,
			"path": COMPARE_RESULTS_FILE,
			"timestamp": comparison.get("timestamp"),
		}
		_write_json(PRUNING_EXPERIMENTS_FILE, store)
		upload_pruning_experiments(store, profile=profile)

		LOGGER.info("Comparison completed")
		LOGGER.info("Comparison details written to %s (id=%s)", COMPARE_RESULTS_FILE, compare_id)
		latency_retention_val = float(comparison.get("overall_summary", {}).get("latency_retention") or 0.0)
		time_reduction_val = float(comparison.get("overall_summary", {}).get("execution_time_reduction") or 0.0)
		latency_retention_good = None if comparison.get("overall_summary", {}).get("latency_retention") is None else float(comparison.get("overall_summary", {}).get("latency_retention") or 0.0) >= 1.0
		time_reduction_good = None if comparison.get("overall_summary", {}).get("execution_time_reduction") is None else float(comparison.get("overall_summary", {}).get("execution_time_reduction") or 0.0) >= 1.0
		LOGGER.info(
			"Comparison retention= %s total_tune_tir_time_reduction= %s",
			_maybe_color(_arrow_prefix(latency_retention_good) + f"{latency_retention_val:.4f}", latency_retention_good),
			_maybe_color(_arrow_prefix(time_reduction_good) + f"{time_reduction_val:.4f}", time_reduction_good),
		)
		return 0

	iteration_outputs: List[Dict[str, Any]] = []
	for iteration in range(1, args.iterations + 1):
		LOGGER.info("%s", "#" * 90)
		LOGGER.info("Pruning iteration %d/%d", iteration, args.iterations)
		LOGGER.info("%s", "#" * 90)

		out = _run_pruning_iteration(
			iteration=iteration,
			tasks=tasks,
			args=args,
			history_stats=history_stats,
			store=store,
			run_tracker=run_tracker,
			profile=profile,
		)
		iteration_outputs.append(out)
		LOGGER.info(
			"Iteration %d stop reason: %s | selected=%s score=%.4f",
			iteration,
			out.get("stop_reason"),
			out["selected"].get("config_name"),
			out["selected"].get("aggregate", {}).get("score", 0.0),
		)

	selected_exp, stability_summary, selection_reason = _select_global_best(
		iteration_outputs=iteration_outputs,
		target_retention=args.target_retention,
		max_latency_loss_pct=args.max_latency_loss_pct,
	)

	# Persist selected schedules via existing helper pipeline.
	_persist_best_schedules_for_selection(
		selected_experiment=selected_exp,
		tasks=tasks,
		args=args,
		history_stats=history_stats,
		store=store,
		run_tracker=run_tracker,
		profile=profile,
	)

	best_payload = _build_best_pruned_payload(
		selected_experiment=selected_exp,
		iteration_outputs=iteration_outputs,
		tasks=tasks,
		history_stats=history_stats,
		selection_reason=selection_reason,
	)

	store["latest_pruning_run"] = {
		"timestamp": _utc_now(),
		"selected_config_hash": selected_exp.get("config_hash"),
		"selected_config_name": selected_exp.get("config_name"),
		"selection_reason": selection_reason,
		"selected_metrics": selected_exp.get("aggregate", {}),
		"stability_summary": stability_summary,
		"iteration_summaries": [
			{
				"iteration": out.get("iteration"),
				"stop_reason": out.get("stop_reason"),
				"selected_config_name": out.get("selected", {}).get("config_name"),
				"selected_score": out.get("selected", {}).get("aggregate", {}).get("score"),
			}
			for out in iteration_outputs
		],
	}

	_write_json(BEST_PRUNED_CONFIG_FILE, best_payload)
	_write_json(PRUNING_EXPERIMENTS_FILE, store)
	upload_best_pruned_config(best_payload, profile=profile)
	upload_pruning_experiments(store, profile=profile)

	selected_agg = selected_exp.get("aggregate", {})
	LOGGER.info("Selected best pruned configuration: %s", selected_exp.get("config_name"))
	LOGGER.info(
		"Retention=%.4f TimeReduction=%.4f TrialReduction=%.4f Score=%.4f",
		selected_agg.get("latency_retention", 0.0),
		selected_agg.get("time_reduction", 0.0),
		selected_agg.get("trial_reduction", 0.0),
		selected_agg.get("score", 0.0),
	)
	LOGGER.info("Wrote %s", BEST_PRUNED_CONFIG_FILE)
	LOGGER.info("Wrote %s", PRUNING_EXPERIMENTS_FILE)
	LOGGER.info("Recommended production config block:\n%s", best_payload["recommended_tune_tir_block"])

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
