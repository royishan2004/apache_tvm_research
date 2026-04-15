# Residual Refiner --- Postmortem and Archive Notes

## What the Residual Refiner Was

The **Residual Refiner** was an experimental optimization layer added
after a strong baseline TVM schedule was already chosen.

Its purpose was to make **small corrective improvements** to
already-good schedules by exploring only nearby knob variations instead
of doing a full search again.

The core intuition was:

> many schedules are already close to optimal, and only need one or two
> final knob corrections.

------------------------------------------------------------------------

## How It Worked

### 1) Start from a strong baseline

The starting point was usually: - a good manual schedule - a
MetaSchedule best trace - a historically strong shape-adjacent trace - a
selected 80/20 tuning configuration

### 2) Residual local corrections

It then tested only **tiny nearby changes** on high-impact knobs: -
vector width - unroll factor - innermost j tile - cache write usage -
reduction decomposition - minor reorder tweaks

Examples: - vector width 8 → 16 - unroll 256 → 512 - tile 16 → 32 -
decompose on/off

### 3) Tiny local benchmarking

Instead of broad search, only **3--8 neighboring candidates** were
benchmarked.

This kept it CPU-friendly and easy to plug into the workflow.

### 4) Accept only measurable wins

Only statistically consistent improvements were accepted. If the gain
was within CPU timing noise, the original schedule was retained.

------------------------------------------------------------------------

## Why We Archived It

### 1) Gains were inconsistent

Some shapes improved slightly, but many: - stayed the same - regressed -
failed to generalize across all BERT shapes

So it lacked reliable global gains.

### 2) Noise hid the gains

Residual improvements were often only a few microseconds.

That was too close to normal CPU benchmark variance, making many wins
hard to trust.

### 3) Local minimum problem

The refiner only searched **nearby decisions**.

If the base schedule was in the wrong optimization region, tiny local
fixes could not recover it.

### 4) ML-guided prediction is strategically better

With the shift toward **LightGBM-guided knob prediction**, it became
better to: \> choose better defaults upfront

instead of: \> patching weak defaults afterward

This gives larger latency upside.

### 5) Maintenance overhead

Even a small refiner still added: - extra benchmarking passes - more
branching logic - more result bookkeeping - more regression surface area

The maintenance cost no longer matched the gains.

### 6) Fix of CPU Thermal Throttling - Elimination of the Purpose of this Experiement

After small cool-downs were implemented into the rule-based schedule, it was observed that
the rule-based schedule was initially underperforming due to throttling, but is now outperforming MetaSchedule for almost all shapes. 

In such a situation, there are minimal to zero residual readings that can be refined, 
thus defying the point of this experiment altogether.

------------------------------------------------------------------------

## Final Takeaway

The residual refiner was still a valuable research step.

It proved an important lesson:

> local corrections help only when the starting schedule is already in
> the right optimization neighborhood.