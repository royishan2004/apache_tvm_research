# 80/20 MetaSchedule Tuner – Documentation and Postmortem

## Overview
This document explains the purpose of the **80/20 MetaSchedule tuner**, how it worked, what it tried to optimize, what went well, and why we finally decided **not to continue using it as the main optimization path**.



---

## What problem this tuner was trying to solve
Running full Apache TVM MetaSchedule tuning for every BERT GEMM shape is expensive.

A normal tuning flow usually spends:
- many search trials
- large evolutionary populations
- multiple validation repeats
- full design-space exploration

This gives good schedules, but:
- it takes a lot of time
- it consumes CPU heavily
- it is hard to reproduce
- it slows research iteration speed

So the idea behind this tuner was simple:

> **Can we get 80–90% of the best latency while spending only 20–30% of the tuning cost?**

That is why this became the **“80/20 tuner.”**

---

## What the tuner actually did
The tuner did **automatic configuration pruning** on top of TVM MetaSchedule.

Instead of tuning the kernel schedule directly, it tuned the **MetaSchedule tuning parameters themselves**.

In simple terms:

- start from a strong baseline tuning setup
- slowly reduce search cost
- keep checking whether latency remains close to baseline
- stop when performance begins to drop too much

---

## The 4 main things it pruned
The script progressively reduced four important tuning dimensions.

### 1) Global trial budget
This controls how many total schedules TVM is allowed to test.

Example:
- baseline: 256
- reduced: 192 → 128 → 96 → 64

This directly reduced tuning time.

---

### 2) Trials per iteration
This controls how many new candidates are measured in each search wave.

Lowering this made tuning faster, but it also reduced exploration diversity.

---

### 3) Measurement rigor
The script also reduced benchmark rigor:
- fewer repeats
- lower `min_repeat_ms`
- fewer validation reruns

This saved time, but also increased measurement noise.

---

### 4) Search breadth
This was the most important pruning dimension.

It reduced:
- evolutionary population size
- number of design-space samples
- mutation aggressiveness
- replay counts
- number of genetic iterations

This had the largest effect on both:
- tuning speed
- schedule quality

---

## Smart additions beyond basic pruning
The tuner was actually quite advanced.

### Historical schedule pattern reuse
It reused information from previously saved best schedules:
- recurring tile factors
- vector widths
- unroll values
- cache-write frequency
- reduction decomposition frequency

This helped bias tuning budgets.

---

### Kernel-aware biasing
Different kernels behave differently:
- `qkv`
- `mlp_expand`
- `mlp_reduce`

The tuner tried to detect:
- which kernels were unstable
- which shapes historically needed more tuning
- which shapes were similar to previous good schedules

Then it automatically increased or decreased the budget.

This was one of the strongest ideas in the workflow.

---

### Shape similarity budget scaling
It also tried to reuse tuning confidence across nearby shapes.

For example:
if a shape with similar `M` had a stable history,
the tuner reduced budget for the new shape.

This was intended to reduce redundant tuning work.

---

## Why we dropped it
Even though the engineering was strong, the **main project goal is latency improvement**, not tuning-cost reduction.

And this is where the 80/20 tuner stopped helping enough.

Below are the major reasons.

---

# 1) The core objective was cost reduction, not latency gain
The tuner’s success metric was mainly:

> **retain near-baseline performance while reducing tuning cost**

That is useful for productivity.

But this project’s real research goal is:

> **achieve lower final kernel latency**

The 80/20 tuner mostly optimized:
- tuning wall time
- trial counts
- benchmark cost

It did **not fundamentally improve the final search quality**.

At best, it tried to preserve it.

So it became more of a **workflow optimization tool**, not a latency optimization system.

---

# 2) Reduced search breadth removed rare high-quality schedules
Some of the best schedules in MetaSchedule come from:
- unusual split factors
- lucky mutation paths
- rare cache placements
- uncommon vectorization decisions

These are often discovered only after:
- larger populations
- more generations
- more mutation retries
- wider exploration

The 80/20 tuner aggressively reduced exactly these.

So the search became faster, but it also became **less capable of finding exceptional schedules**.

This directly hurt latency.

---

# 3) Measurement pruning introduced noisy decisions
Reducing measurement rigor caused an important issue:

> the tuner sometimes pruned based on noisy latency numbers.

When:
- repeats are reduced
- min repeat time is reduced
- validation runs are reduced

small benchmark fluctuations start looking like real improvements.

This means the tuner may incorrectly:
- keep weak configs
- reject strong configs
- stop pruning too early
- continue pruning too far

This instability was likely one of the major hidden causes of poor results.

---

# 4) Kernel-aware budget bias was still heuristic
The kernel-aware multiplier system was clever, but still heuristic.

It relied on:
- instability scores
- entropy of historical patterns
- nearest shape similarity
- handcrafted multipliers

These signals are useful, but they do not directly model:

> **which schedule knobs actually reduce latency**

So the tuner was still making decisions using indirect signals.

That limits real latency gains.

---

# 5) Historical traces do not always transfer well
A major hidden issue:

> good historical patterns are not always reusable.

Even nearby shapes can prefer different:
- tile sizes
- vector widths
- decompose positions
- cache-write locations

This happens because GEMM scheduling is highly shape-sensitive.

So historical reuse sometimes biased the search toward:
- previously common
- but currently suboptimal

regions of the search space.

This likely explains why even some representative shapes still performed badly.

---

# 6) Budget pruning compounded across multiple dimensions
The biggest structural issue:

The tuner reduced **multiple dimensions at the same time**:
- global trials
- per-iteration trials
- measurement rigor
- search breadth

Each individual reduction may be safe.

But when combined, the effect compounds.

This can cause:
- poor exploration
- noisy ranking
- premature convergence
- repeated mediocre trace families

This compounded degradation is a strong reason why results worsened even on shapes included in the tuning subset.

---

# Final conclusion
The 80/20 tuner was **not a failed idea**.

It successfully proved that:
- tuning cost can be reduced
- historical traces can guide search budgets
- pruning strategies can be automated
- comparison pipelines can be standardized

But for the larger research objective of:
> **real latency improvements**

it hit a natural ceiling.


---


This folder stores retired artifacts from the deprecated 80/20 pruning and best-config workflows.

Archived scripts:
- `scripts/metaschedule_8020_tuner.py`
- `scripts/metaschedule_tune_best_config.py`

Archived results/artifacts:
- `results/metaschedule/best_pruned_config.json`
- `results/metaschedule/pruning_experiments.json`
- `results/metaschedule/comparison_results.json`
- `results/metaschedule/best_schedules_metaschedule_best_config.json`
- `results/metaschedule/8020/`
- `results/metaschedule/best_config/`
- `results/metaschedule/best_config_smoke/`

Active baseline MetaSchedule workflow remains under:
- `research/workloads/bert/metaschedule/metaschedule_tune.py`
- `research/results/metaschedule/best_schedules.json`
