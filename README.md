# TVM-Based Transformer MatMul Optimization (Research Workspace)

## Objective
This project analyzes and optimizes **Transformer MatMul kernels (starting with BERT)** using **TVM TIR scheduling**, focusing on correctness, performance, and reproducibility.

The primary goal is to **extract real Transformer MatMul workloads, create canonical TIR kernels, and systematically evaluate scheduling strategies**.

---

## Execution Guide (What to run, where, and why)

> All commands are run from the **Apache_TVM/** root unless stated otherwise.

---

### Phase 0 — Environment Validation

```bash
python3 research/workloads/common/env_check.py
```
**Why:** Verify that TVM, LLVM, and Python bindings are correctly set up.

---

### Phase 1 — Load Transformer Model

```bash
python3 research/workloads/bert/load_bert.py
```
**Why:** Downloads and loads BERT for static graph inspection (no training).

---

### Phase 2 — Extract MatMul Shapes from BERT

```bash
python3 research/workloads/bert/extract_matmul_shapes.py
```
**Why:** Extracts *all* MatMul shapes used internally by BERT.

Output:
```
bert_matmul_shapes_raw.json
```

---

```bash
python3 research/workloads/bert/filter_qkv.py
```
**Why:** Filters only **Q/K/V projection MatMuls**  
(shape: `[*, 768] x [768, 768]`).

Output:
```
bert_matmul_shapes_qkv.json
```

---

### Phase 3 — Canonical TIR Kernel Construction

```bash
python3 -m research.workloads.bert.matmul.qkv_matmul
```
**Why:** Generates the **baseline canonical TIR MatMul kernel** for BERT QKV.

---

### Phase 3.1 — Baseline Performance

```bash
python3 -m research.workloads.bert.matmul.qkv_run baseline
```
**Why:** Measures unscheduled baseline performance.

---

### Phase 3.2 — Reduction Axis Splitting

```bash
python3 -m research.workloads.bert.matmul.qkv_run k16
python3 -m research.workloads.bert.matmul.qkv_run k32
python3 -m research.workloads.bert.matmul.qkv_run k64
```
**Why:** Studies impact of **K-axis tiling** on cache and performance.

---

### Phase 3.3 — Parallelism & Vectorization

```bash
python3 -m research.workloads.bert.matmul.qkv_run parallel
python3 -m research.workloads.bert.matmul.qkv_run vec_j
python3 -m research.workloads.bert.matmul.qkv_run parallel_k16
python3 -m research.workloads.bert.matmul.qkv_run parallel_vec_j
python3 -m research.workloads.bert.matmul.qkv_run vec_j_k16
python3 -m research.workloads.bert.matmul.qkv_run full
```
**Why:** Evaluates **composite schedules** combining:
- parallelism
- reduction splitting
- vectorization

Used to identify best-performing manual schedules.

---

### Additional Canonical Kernels (MLP Layers)

```bash
python3 research/workloads/bert/matmul/mlp_expanded_matmul.py
python3 research/workloads/bert/matmul/mlp_compressed_matmul.py
```
**Why:** Generates canonical MatMul kernels for **Transformer MLP layers**  
(used in later optimization phases).

---

## Current Status
- ✔ TVM correctness validated  
- ✔ BERT MatMul shapes extracted  
- ✔ Canonical kernels created  
- ✔ Manual scheduling strategies benchmarked  

Next step: **Phase 4 — Automated scheduling / MetaSchedule**
