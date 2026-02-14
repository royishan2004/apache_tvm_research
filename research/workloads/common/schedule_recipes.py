import tvm

def _sched_k16(mod):
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    k0, k1 = sch.split(k, factors=[None, 16])
    sch.reorder(i, j, k0, k1)
    sch.vectorize(j)
    return sch


def _sched_k32(mod):
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    k0, k1 = sch.split(k, factors=[None, 32])
    sch.reorder(i, j, k0, k1)
    sch.vectorize(j)
    return sch


def _sched_k64(mod):
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    k0, k1 = sch.split(k, factors=[None, 64])
    sch.reorder(i, j, k0, k1)
    sch.vectorize(j)
    return sch


def _sched_parallel(mod):
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    sch.parallel(i)
    sch.vectorize(j)
    return sch


def _sched_vec_j(mod):
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    sch.vectorize(j)
    return sch


def _sched_vec_k(mod):
    """NOTE: Expected failure case — reduction-axis vectorization is
    illegal unless the reduction is transformed first."""
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    k0, k1 = sch.split(k, factors=[None, 8])
    sch.vectorize(k1)
    return sch


def _sched_parallel_k16(mod):
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    k0, k1 = sch.split(k, factors=[None, 16])
    sch.reorder(i, j, k0, k1)
    sch.parallel(i)
    sch.unroll(k1)
    return sch


def _sched_parallel_vec_j(mod):
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    j0, j1 = sch.split(j, factors=[None, 8])
    sch.reorder(i, j0, j1, k)
    sch.parallel(i)
    sch.vectorize(j1)
    return sch


def _sched_vec_j_k16(mod):
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    j0, j1 = sch.split(j, factors=[None, 8])
    k0, k1 = sch.split(k, factors=[None, 16])
    sch.reorder(i, j0, k0, j1, k1)
    sch.vectorize(j1)
    sch.unroll(k1)
    return sch


def _sched_full(mod):
    sch = tvm.tir.Schedule(mod)
    sch.work_on("main")
    block = sch.get_block("C")
    i, j, k = sch.get_loops(block)
    j0, j1 = sch.split(j, factors=[None, 8])
    k0, k1 = sch.split(k, factors=[None, 16])
    sch.reorder(i, j0, k0, j1, k1)
    sch.parallel(i)
    sch.vectorize(j1)
    sch.unroll(k1)
    return sch



RECIPES = {
    "k16":           _sched_k16,
    "k32":           _sched_k32,
    "k64":           _sched_k64,
    "parallel":      _sched_parallel,
    "vec_j":         _sched_vec_j,
    "vec_k":         _sched_vec_k,
    "parallel_k16":  _sched_parallel_k16,
    "parallel_vec_j": _sched_parallel_vec_j,
    "vec_j_k16":     _sched_vec_j_k16,
    "full":          _sched_full,
}


def apply_schedule(mod, variant: str):
    """Apply a named schedule recipe to a MatMul TIR module.

    Parameters
    ----------
    mod : tvm.ir.IRModule
        TIR module from matmul_tir(M, K, N).
    variant : str
        One of RECIPES.keys() or "baseline".

    Returns
    -------
    tvm.ir.IRModule
        Scheduled (or unmodified for baseline) module ready for tvm.build.
    """
    if variant == "baseline":
        return mod
    if variant not in RECIPES:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Available: baseline, {', '.join(sorted(RECIPES))}"
        )
    sch = RECIPES[variant](mod)
    return sch.mod


def available_variants():
    """Return sorted list of all variant names including baseline."""
    return ["baseline"] + sorted(RECIPES.keys())
