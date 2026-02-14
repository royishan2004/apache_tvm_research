#THIS IS SPECIFIC TO BERT.  It captures the key dimensions of the QKV projection and MLP layers, as extracted from real BERT models.  These constants are used across the BERT matmul workloads to ensure consistency and to avoid hardcoding the same values in multiple places.

import json
import os
import warnings
from typing import Dict, List, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_RAW_JSON = os.path.join(_HERE, "bert_matmul_shapes_raw.json")
_QKV_JSON = os.path.join(_HERE, "bert_matmul_shapes_qkv.json")

_DEFAULT_HIDDEN = 768
_DEFAULT_FF     = 3072  # 4 × hidden


def _load_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def _derive_qkv_hidden(qkv_data: list) -> int:
    if not qkv_data:
        return _DEFAULT_HIDDEN
    # Every QKV entry has K == N == hidden
    hidden_sizes = {entry["K"] for entry in qkv_data}
    assert len(hidden_sizes) == 1, f"Inconsistent QKV hidden sizes: {hidden_sizes}"
    return hidden_sizes.pop()


def _derive_mlp_shapes(raw_data: list, hidden: int) -> Dict[str, Tuple[int, int]]:
    """
    From the raw shapes, find the two MLP linear layers:
      expanded:   (M, hidden) × (hidden, ff)   → weight shape (ff, hidden)
      compressed: (M, ff)     × (ff, hidden)    → weight shape (hidden, ff)

    Returns {"ff": ff_dim} once we identify the intermediate size.
    """
    ff_sizes = set()
    for item in raw_data:
        w = item["weight"]  # [out_features, in_features]
        out_f, in_f = w[0], w[1]
        # Expanded: weight (3072, 768) → in=768, out=3072
        if in_f == hidden and out_f != hidden:
            ff_sizes.add(out_f)
        # Compressed: weight (768, 3072) → in=3072, out=768
        if out_f == hidden and in_f != hidden:
            ff_sizes.add(in_f)
    if not ff_sizes:
        return {"ff": _DEFAULT_FF}
    assert len(ff_sizes) == 1, f"Inconsistent FF sizes: {ff_sizes}"
    return {"ff": ff_sizes.pop()}


# ─── load once at import time ──────────────────────────────────────
_qkv_data = _load_json(_QKV_JSON)
_raw_data = _load_json(_RAW_JSON)

if not _qkv_data:
    warnings.warn(
        f"QKV shape file not found ({_QKV_JSON}). "
        "Run extract_matmul_shapes.py → filter_qkv.py first. "
        "Using BERT-base defaults (H=768, FF=3072)."
    )
if not _raw_data:
    warnings.warn(
        f"Raw shape file not found ({_RAW_JSON}). "
        "Run extract_matmul_shapes.py first. "
        "Using BERT-base defaults."
    )


HIDDEN: int = _derive_qkv_hidden(_qkv_data)

FF: int = _derive_mlp_shapes(_raw_data, HIDDEN)["ff"]

NUM_QKV_LAYERS: int = len(_qkv_data)

# Default to be changed to accomodate more values of M.
DEFAULT_M: int = 128

def qkv_shape(M: int = DEFAULT_M) -> Tuple[int, int, int]:
    return (M, HIDDEN, HIDDEN)

def mlp_expanded_shape(M: int = DEFAULT_M) -> Tuple[int, int, int]:
    return (M, HIDDEN, FF)

def mlp_compressed_shape(M: int = DEFAULT_M) -> Tuple[int, int, int]:
    return (M, FF, HIDDEN)


def print_config(M: int = DEFAULT_M) -> None:
    src = "extracted JSON" if _qkv_data else "defaults (JSON not found)"
    print(f"BERT shape config  [source: {src}]")
    print(f"  HIDDEN       = {HIDDEN}")
    print(f"  FF           = {FF}")
    print(f"  M (default)  = {M}")
    print(f"  QKV shape    = {qkv_shape(M)}")
    print(f"  MLP expanded = {mlp_expanded_shape(M)}")
    print(f"  MLP compress = {mlp_compressed_shape(M)}")
    if _qkv_data:
        print(f"  QKV layers   = {NUM_QKV_LAYERS}")


if __name__ == "__main__":
    print_config()
