import sys
import json
import os
import time
import numpy as np
import tvm
from tvm.runtime import device

from research.workloads.bert.bert_shapes import (
    M_LIST,
    qkv_shape, mlp_expanded_shape, mlp_compressed_shape,
    print_config,
)
from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.common.schedule_recipes import (
    apply_schedule, available_variants,
)

KERNELS = {
    "qkv":        qkv_shape,
    "mlp_expand": mlp_expanded_shape,
    "mlp_reduce": mlp_compressed_shape,
}

VARIANT_LIST = available_variants()

TARGET = "llvm"
DEV = tvm.cpu(0)

EVAL_NUMBER = 50
EVAL_REPEAT = 3
EVAL_MIN_MS = 50

ALL_TAG = "--all"
ALL_VARIANTS_TAG = "--all-variants"
ALL_KERNELS_TAG = "--all-kernels"

def _parse_args():
    valid_choices = VARIANT_LIST + [ALL_TAG, ALL_VARIANTS_TAG]
    if len(sys.argv) < 2 or sys.argv[1] not in valid_choices:
        print("Usage:")
        print(
            "  python3 -m research.workloads.bert.matmul.qkv_mlp_run "
            "<variant|--all-variants|--all> "
            "[--kernel K|--all-kernels] [--iterations N]"
        )
        print("\nAvailable variants:")
        for v in VARIANT_LIST:
            print(f"  {v}")
        print(f"  {ALL_VARIANTS_TAG}   (run every variant × every kernel)")
        print(f"  {ALL_TAG}            (alias of {ALL_VARIANTS_TAG})")
        print(f"\nOptions:")
        print(f"  --kernel K   one of: {', '.join(KERNELS)}  (default: qkv)")
        print(f"  {ALL_KERNELS_TAG}   run all kernels for the selected variant")
        print(f"  --iterations N   repeat the selected run N times (default: 1)")
        print(f"               ignored when variant is '{ALL_VARIANTS_TAG}'")
        print(f"\nEvery run sweeps M_LIST = {M_LIST}")
        sys.exit(1)

    variant = sys.argv[1]
    run_all_variants = variant in (ALL_TAG, ALL_VARIANTS_TAG)

    run_all_kernels = ALL_KERNELS_TAG in sys.argv
    if run_all_variants and run_all_kernels:
        print(f"{ALL_KERNELS_TAG} is not needed with {ALL_VARIANTS_TAG}.")
        sys.exit(1)

    iterations = 1
    if "--iterations" in sys.argv:
        idx = sys.argv.index("--iterations")
        if idx + 1 >= len(sys.argv):
            print("--iterations requires a positive integer argument")
            sys.exit(1)
        try:
            iterations = int(sys.argv[idx + 1])
            if iterations < 1:
                raise ValueError
        except ValueError:
            print(f"Invalid iterations value '{sys.argv[idx + 1]}'. Must be a positive integer.")
            sys.exit(1)

    kernel = "qkv"
    if "--kernel" in sys.argv and run_all_kernels:
        print(f"Use either --kernel K or {ALL_KERNELS_TAG}, not both.")
        sys.exit(1)

    if "--kernel" in sys.argv:
        idx = sys.argv.index("--kernel")
        if idx + 1 >= len(sys.argv):
            print("Missing value for --kernel")
            sys.exit(1)
        kernel = sys.argv[idx + 1]
        if kernel not in KERNELS:
            print(f"Unknown kernel '{kernel}'. Choose from: {', '.join(KERNELS)}")
            sys.exit(1)

    return variant, kernel, run_all_variants, run_all_kernels, iterations

variant, kernel, run_all_variants, run_all_kernels, iterations = _parse_args()

if run_all_variants:
    run_plan = [
        #INTENTIONAL exclude of vec_k and rule_based for bypassing exception case and creating training data for rule-based model
        (k, v) for k in KERNELS for v in VARIANT_LIST if v != "vec_k" and v != "rule_based" 
    ]
    print(f"Mode   : {ALL_VARIANTS_TAG}")
    print(f"Kernels: {', '.join(KERNELS)}")
    print(f"Variants per kernel: {len(VARIANT_LIST)}")
elif run_all_kernels:
    run_plan = [(k, variant) for k in KERNELS]
    print(f"Mode   : {ALL_KERNELS_TAG}")
    print(f"Variant: {variant}")
    print(f"Kernels: {', '.join(KERNELS)}")
else:
    run_plan = [(kernel, variant)]
    print(f"Kernel : {kernel}")
    print(f"Variant: {variant}")

print(f"M sweep: {M_LIST}")
if iterations > 1:
    print(f"Iterations: {iterations}")
print_config()
print()

RESULTS_DIR = "research/results"
RESULTS_FILE = os.path.join(RESULTS_DIR, "bert_matmul_results.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

results = []
if os.path.exists(RESULTS_FILE):
    try:
        with open(RESULTS_FILE, "r") as f_in:
            loaded = json.load(f_in)
            if isinstance(loaded, list):
                results = [r for r in loaded if isinstance(r, dict)]
    except json.JSONDecodeError:
        print("⚠️  Corrupted results file — starting fresh")

total_combos = len(run_plan)
for iteration in range(1, iterations + 1):
    if iterations > 1:
        print(f"\n{'#' * 80}")
        print(f"### Iteration {iteration}/{iterations}")
        print(f"{'#' * 80}")

    for combo_idx, (cur_kernel, cur_variant) in enumerate(run_plan, 1):
        cur_shape_fn = KERNELS[cur_kernel]
        if total_combos > 1:
            combo_prefix = ""
            if iterations > 1:
                combo_prefix = f"iter={iteration}/{iterations} "
            variant_ts = time.strftime("%Y-%m-%d %H:%M:%S")
            print(
                f"\n=== [{combo_idx}/{total_combos}] {combo_prefix}"
                f"kernel={cur_kernel}  variant={cur_variant}  >>> STARTED: {variant_ts} <<< ===\n"
            )

        for M_val in M_LIST:
            M, K, N = cur_shape_fn(M_val)
            base_mod = matmul_tir(M, K, N)
            scheduled_mod = apply_schedule(
                base_mod, cur_variant, M=M, K=K, N=N, kernel=cur_kernel
            )

            rt_mod = tvm.build(scheduled_mod, target=TARGET)

            # Allocate inputs
            A_np = np.random.randn(M, K).astype("float32")
            B_np = np.random.randn(K, N).astype("float32")
            C_np = np.zeros((M, N), dtype="float32")

            A = tvm.nd.array(A_np, DEV)
            B_mat = tvm.nd.array(B_np, DEV)
            C = tvm.nd.array(C_np, DEV)

            evaluator = rt_mod.time_evaluator(
                "main",
                dev=DEV,
                number=EVAL_NUMBER,
                repeat=EVAL_REPEAT,
                min_repeat_ms=EVAL_MIN_MS,
            )

            result = evaluator(A, B_mat, C)

            mean_us = result.mean * 1e6
            std_us = result.std * 1e6

            iter_label = f" [iter {iteration}/{iterations}]" if iterations > 1 else ""
            print(
                f"  M={M:4d}  K={K:4d}  N={N:4d}  "
                f"latency = {mean_us:.4f} µs  (± {std_us:.4f} µs){iter_label}"
            )

            results.append({
                "kernel":     cur_kernel,
                "variant":    cur_variant,
                "M": M,
                "K": K,
                "N": N,
                "latency_us": float(mean_us),
                "std_us":     float(std_us),
                "number":     EVAL_NUMBER,
                "repeat":     EVAL_REPEAT,
                "min_repeat_ms": EVAL_MIN_MS,
                "target":     TARGET,
                "iteration":  iteration,
                "total_iterations": iterations,
                "timestamp":  time.strftime("%Y-%m-%d %H:%M:%S"),
            })

        with open(RESULTS_FILE, "w") as f_out:
            json.dump(results, f_out, indent=2)

print(f"\nResults saved to {RESULTS_FILE}")
