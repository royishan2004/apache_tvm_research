import os
import tvm
from tvm import meta_schedule as ms
from tvm.target import Target

from research.workloads.common.matmul_templates import matmul_tir

B = 128
H = 768

def workload():
    return matmul_tir(B, H, H)

TARGET = Target("llvm -num-cores=8")

WORK_DIR = "research/results/metaschedule/qkv"
DB_PATH = os.path.join(WORK_DIR, "database.json")

os.makedirs(WORK_DIR, exist_ok=True)

print("Starting MetaSchedule tuning for BERT QKV MatMul")
print(f"Target: {TARGET}")
print(f"Work dir: {WORK_DIR}")

database = ms.tir_integration.tune_tir(
    mod=workload(),
    target=TARGET,
    work_dir=WORK_DIR,
    max_trials_global=256,     # SAFE: finishes in minutes on CPU
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


print("MetaSchedule database saved automatically to:", WORK_DIR)

print("MetaSchedule tuning completed successfully")
print(f"Database saved at: {DB_PATH}")
