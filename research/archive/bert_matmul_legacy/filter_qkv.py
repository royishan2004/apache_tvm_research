"""
DEPRECATED — no longer needed.

The new extract_matmul_shapes.py reads weight shapes directly from
BERT's state_dict and categorises them by layer_type (QKV_Proj,
MLP_Expand, MLP_Reduce, Attn_Out) in a single JSON file
(bert_matmul_shapes.json).

bert_shapes.py now reads that single file; the old two-file pipeline
(bert_matmul_shapes_raw.json → filter_qkv.py → bert_matmul_shapes_qkv.json)
is no longer used.
"""

import warnings
warnings.warn(
    "filter_qkv.py is deprecated. "
    "Run extract_matmul_shapes.py instead — it produces "
    "bert_matmul_shapes.json with layer_type labels directly.",
    DeprecationWarning,
    stacklevel=1,
)
