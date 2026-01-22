# TVM-Based Transformer MatMul Optimization (Research Workspace)

## Objective

This project studies and optimizes **Transformer MatMul kernels (starting with BERT)** using **Apache TVM (TIR + MetaSchedule)**.

The primary goals are to:

- Extract **real Transformer MatMul workloads**
- Construct **canonical TIR kernels**
- Systematically evaluate **manual scheduling strategies**
- Compare against **automated schedule search (MetaSchedule)**
- Produce **reproducible, quantitative performance results**

The project emphasizes **correctness, controlled experimentation, and explainable performance gains**.

---

## Execution Guide (What to run, where, and why)

> All commands are run from the **Apache_TVM/** project root unless stated otherwise.

---

## View Collected Results

```bash
python3 -m research.analysis.print_qkv_results
```

**Why:**  
Prints a consolidated table of all recorded QKV MatMul results, including:
- baseline
- all manual schedules
- MetaSchedule best result

This is the **primary comparison artifact** for Phase 4.3.

---

## Phase 0 — Environment Validation

```bash
source venv/bin/activate
python3 research/workloads/common/env_check.py
```

---

## Phase 1 — Load Transformer Model

```bash
python3 research/workloads/bert/load_bert.py
```

---

## Phase 2 — Extract MatMul Shapes from BERT

```bash
python3 research/workloads/bert/extract_matmul_shapes.py
```

```bash
python3 research/workloads/bert/filter_qkv.py
```

---

## Phase 3 — Canonical TIR Kernel Construction

```bash
python3 -m research.workloads.bert.matmul.qkv_matmul
```

---

## Phase 3.1 — Baseline Performance

```bash
python3 -m research.workloads.bert.matmul.qkv_run baseline
```

---

## Phase 3.2 — Reduction Axis Splitting

```bash
python3 -m research.workloads.bert.matmul.qkv_run k16
python3 -m research.workloads.bert.matmul.qkv_run k32
python3 -m research.workloads.bert.matmul.qkv_run k64
```

---

## Phase 3.3 — Parallelism & Vectorization

```bash
python3 -m research.workloads.bert.matmul.qkv_run parallel
python3 -m research.workloads.bert.matmul.qkv_run vec_j
python3 -m research.workloads.bert.matmul.qkv_run parallel_k16
python3 -m research.workloads.bert.matmul.qkv_run parallel_vec_j
python3 -m research.workloads.bert.matmul.qkv_run vec_j_k16
python3 -m research.workloads.bert.matmul.qkv_run full
```

---

## Phase 4 — Automated Scheduling with MetaSchedule

### Phase 4.1 — MetaSchedule Tuning

```bash
python3 -m research.workloads.bert.metaschedule.qkv_metaschedule_tune
```

---

### Phase 4.2 — Result Extraction

Results are recorded directly from tuning logs into:

```
research/results/bert_qkv_results.json
```

---

### Phase 4.3 — Comparative Analysis

Manual vs MetaSchedule performance comparison completed.

---

## Additional Canonical Kernels (MLP Layers)

```bash
python3 research/workloads/bert/matmul/mlp_expanded_matmul.py
python3 research/workloads/bert/matmul/mlp_compressed_matmul.py
```

---

## Current Status

- ✔ Environment validated  
- ✔ BERT MatMul shapes extracted  
- ✔ Canonical kernels created  
- ✔ Manual schedules benchmarked  
- ✔ MetaSchedule comparison completed  

**Next step:** Phase 5 — Generalization to additional Transformer workloads
