import os
from tvm import meta_schedule as ms
from tvm.target import Target

from research.workloads.bert.matmul.qkv_matmul import mod

TARGET = Target("llvm -num-cores=8")
WORK_DIR = "research/results/metaschedule/qkv"

os.makedirs(WORK_DIR, exist_ok=True)

print("Starting MetaSchedule tuning for BERT QKV MatMul")
print(f"Target: {TARGET}")
print(f"Work dir: {WORK_DIR}")

ms.tir_integration.tune_tir(
    mod=mod,                      # SAME object
    target=TARGET,
    work_dir=WORK_DIR,
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

print("✔ MetaSchedule tuning completed successfully")

import subprocess
import sys

print("\nRunning MetaSchedule log parser...")
subprocess.check_call([
    sys.executable,
    "-m",
    "research.workloads.bert.metaschedule.qkv_metaschedule_log_parse",
])
print("✔ MetaSchedule log parser completed successfully")