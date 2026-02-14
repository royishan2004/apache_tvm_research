import json
import os
import warnings
from typing import Dict, List, Tuple

# ─── paths ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_SHAPES_JSON = os.path.join(_HERE, "bert_matmul_shapes.json")

# ─── BERT-base defaults (fallback) ─────────────────────────────────
_DEFAULT_HIDDEN = 768
_DEFAULT_FF     = 3072  # 4 × hidden

# ─── loader ─────────────────────────────────────────────────────────

def _load_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _by_type(data: list, layer_type: str) -> list:
    """Filter records by layer_type."""
    return [r for r in data if r["layer_type"] == layer_type]


def _unique_shape(records: list) -> Tuple[int, int]:
    """Assert all records share the same (K, N) and return it."""
    shapes = {(r["K"], r["N"]) for r in records}
    assert len(shapes) == 1, f"Inconsistent shapes: {shapes}"
    return shapes.pop()


# ─── load once at import time ──────────────────────────────────────
_data = _load_json(_SHAPES_JSON)

if not _data:
    warnings.warn(
        f"Shape file not found ({_SHAPES_JSON}). "
        "Run extract_matmul_shapes.py first. "
        "Using BERT-base defaults (H=768, FF=3072)."
    )

# ─── derive constants from extracted data ──────────────────────────
_qkv_records    = _by_type(_data, "QKV_Proj")
_mlp_exp_records = _by_type(_data, "MLP_Expand")
_mlp_red_records = _by_type(_data, "MLP_Reduce")

if _qkv_records:
    _qkv_K, _qkv_N = _unique_shape(_qkv_records)
else:
    _qkv_K, _qkv_N = _DEFAULT_HIDDEN, _DEFAULT_HIDDEN

if _mlp_exp_records:
    _mlp_exp_K, _mlp_exp_N = _unique_shape(_mlp_exp_records)
else:
    _mlp_exp_K, _mlp_exp_N = _DEFAULT_HIDDEN, _DEFAULT_FF

if _mlp_red_records:
    _mlp_red_K, _mlp_red_N = _unique_shape(_mlp_red_records)
else:
    _mlp_red_K, _mlp_red_N = _DEFAULT_FF, _DEFAULT_HIDDEN

# ─── PUBLIC CONSTANTS ──────────────────────────────────────────────

HIDDEN: int = _qkv_K
"""Hidden dimension verified from BERT QKV weights (768 for bert-base)."""

FF: int = _mlp_exp_N
"""Feed-forward intermediate dimension (3072 for bert-base)."""

NUM_QKV_LAYERS: int = len(_qkv_records)
"""Number of QKV projection weight tensors captured (36 for bert-base: 12 layers × 3)."""

# M is a benchmarking variable (batch_size × seq_len), not fixed by the model.
# Every run sweeps all values in M_LIST — there is no single default M.
M_LIST: List[int] = [16, 32, 64, 96, 128, 192, 256, 384]
"""Standard M values swept in every benchmark run."""

# ─── shape helpers ─────────────────────────────────────────────────

def qkv_shape(M: int) -> Tuple[int, int, int]:
    """Return (M, K, N) for a QKV projection MatMul."""
    return (M, _qkv_K, _qkv_N)

def mlp_expanded_shape(M: int) -> Tuple[int, int, int]:
    """Return (M, K, N) for the MLP expanded linear (hidden → ff)."""
    return (M, _mlp_exp_K, _mlp_exp_N)

def mlp_compressed_shape(M: int) -> Tuple[int, int, int]:
    """Return (M, K, N) for the MLP compressed linear (ff → hidden)."""
    return (M, _mlp_red_K, _mlp_red_N)

# ─── representative layer keys (one per type, layer 0) ────────────

LAYER_KEYS: Dict[str, str] = {}
"""One representative state_dict key per layer type (from layer 0)."""
for lt in ("QKV_Proj", "Attn_Out", "MLP_Expand", "MLP_Reduce"):
    recs = _by_type(_data, lt)
    layer0 = [r for r in recs if r.get("layer_idx") == 0]
    if layer0:
        LAYER_KEYS[lt] = layer0[0]["layer_key"]

# ─── summary helper ────────────────────────────────────────────────

def print_config() -> None:
    """Print shape configuration (no default M — all runs sweep M_LIST)."""
    src = "extracted JSON" if _data else "defaults (JSON not found)"
    M0 = M_LIST[0]  # just for display
    print(f"BERT shape config  [source: {src}]")
    print(f"  HIDDEN       = {HIDDEN}")
    print(f"  FF           = {FF}")
    print(f"  QKV shape    = {qkv_shape(M0)}  (example M={M0})")
    print(f"  MLP expanded = {mlp_expanded_shape(M0)}")
    print(f"  MLP compress = {mlp_compressed_shape(M0)}")
    if _data:
        print(f"  QKV weights  = {NUM_QKV_LAYERS}")
    print(f"  M sweep      = {M_LIST}")


if __name__ == "__main__":
    print_config()
