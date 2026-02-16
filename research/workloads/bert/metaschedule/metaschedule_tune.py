import os
from tvm import meta_schedule as ms
from tvm.target import Target

from research.workloads.common.matmul_templates import matmul_tir
from research.workloads.bert.bert_shapes import (
    qkv_shape,
    mlp_expanded_shape,
    mlp_compressed_shape,
    M_LIST,
)

TARGET = Target("llvm -num-cores=8")
WORK_DIR_BASE = "research/results/metaschedule"

KERNELS = {
    "qkv": qkv_shape,
    "mlp_expand": mlp_expanded_shape,
    "mlp_reduce": mlp_compressed_shape,
}

os.makedirs(WORK_DIR_BASE, exist_ok=True)

print("Starting MetaSchedule tuning for BERT MatMul kernels (per-kernel, per-M)")
print(f"Target: {TARGET}")
print(f"Work dir base: {WORK_DIR_BASE}")


def _parse_args():
    """Parse CLI args. Usage: --kernel <qkv|mlp_expand|mlp_reduce>"""
    import sys
    kernel = None
    if "--kernel" in sys.argv:
        idx = sys.argv.index("--kernel")
        if idx + 1 >= len(sys.argv):
            print("Usage: python3 -m research.workloads.bert.metaschedule.metaschedule_tune [--kernel <name>]")
            print(f"Available kernels: {', '.join(sorted(KERNELS))}")
            sys.exit(1)
        kernel = sys.argv[idx + 1]
        if kernel not in KERNELS:
            print(f"Unknown kernel '{kernel}'. Choose from: {', '.join(sorted(KERNELS))}")
            sys.exit(1)
    return kernel


selected_kernel = _parse_args()
if selected_kernel:
    print(f"Selected kernel: {selected_kernel} (other kernels will be skipped)")

for kernel_name, shape_fn in KERNELS.items():
    if selected_kernel and kernel_name != selected_kernel:
        continue
    kernel_dir = os.path.join(WORK_DIR_BASE, kernel_name)
    os.makedirs(kernel_dir, exist_ok=True)
    print(f"\n=== Kernel: {kernel_name}  -> work dir: {kernel_dir} ===")

    for M_val in M_LIST:
        M, K, N = shape_fn(M_val)
        mod = matmul_tir(M, K, N)
        work_dir = os.path.join(kernel_dir, f"M_{M}")
        os.makedirs(work_dir, exist_ok=True)

        print(f"\n{'=' * 120}")
        print(f"\nTuning kernel={kernel_name}  M={M}  K={K}  N={N}  -> work dir: {work_dir}")
        print(f"\n{'=' * 120}\n")
        ms.tir_integration.tune_tir(
            mod=mod,
            target=TARGET,
            work_dir=work_dir,
            max_trials_global=256,
            num_trials_per_iter=64,
            max_trials_per_task=256,
            builder=ms.builder.LocalBuilder(),
            runner=ms.runner.LocalRunner(
                evaluator_config=ms.runner.EvaluatorConfig(
                    number=5,
                    repeat=1,
                    min_repeat_ms=100,
                )
            ),
        )

        print(f"✔ Completed tuning for kernel={kernel_name} M={M}")

print("\n✔ All MetaSchedule tuning runs completed successfully")

import subprocess
import sys

print("\nRunning MetaSchedule log parser...")
subprocess.check_call([
    sys.executable,
    "-m",
    "research.workloads.bert.metaschedule.metaschedule_log_parse",
])
print("✔ MetaSchedule log parser completed successfully")
