import json
import os
import torch
from transformers import BertModel

_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT_JSON = os.path.join(_HERE, "bert_matmul_shapes.json")

print("Loading BERT-base-uncased weights...")
model = BertModel.from_pretrained("bert-base-uncased")
state = model.state_dict()

_sample_keys = [k for k in state if "attention.self.query.weight" in k]
assert _sample_keys, "Could not find any attention query weights in state_dict"
_prefix = _sample_keys[0].rsplit("attention.self.query.weight", 1)[0]
# _prefix is e.g. "encoder.layer.0." → strip the layer number part
_prefix = _prefix.split("0.")[0]  # e.g. "encoder.layer."
print(f"Detected key prefix: '{_prefix}'")

NUM_LAYERS = model.config.num_hidden_layers      # 12 for bert-base

# For each encoder layer i (0..11), BERT has:
#   {prefix}{i}.attention.self.query.weight   → QKV_Proj
#   {prefix}{i}.attention.self.key.weight     → QKV_Proj
#   {prefix}{i}.attention.self.value.weight   → QKV_Proj
#   {prefix}{i}.attention.output.dense.weight → Attn_Out
#   {prefix}{i}.intermediate.dense.weight     → MLP_Expand
#   {prefix}{i}.output.dense.weight           → MLP_Reduce

LAYER_PATTERNS = {
    "QKV_Proj": [
        _prefix + "{i}.attention.self.query.weight",
        _prefix + "{i}.attention.self.key.weight",
        _prefix + "{i}.attention.self.value.weight",
    ],
    "Attn_Out": [
        _prefix + "{i}.attention.output.dense.weight",
    ],
    "MLP_Expand": [
        _prefix + "{i}.intermediate.dense.weight",
    ],
    "MLP_Reduce": [
        _prefix + "{i}.output.dense.weight",
    ],
}

records = []

for layer_idx in range(NUM_LAYERS):
    for layer_type, patterns in LAYER_PATTERNS.items():
        for pattern in patterns:
            key = pattern.format(i=layer_idx)
            if key not in state:
                print(f"  WARNING: key not found: {key}")
                continue
            w = state[key]
            # nn.Linear weight shape: (out_features, in_features)
            # GEMM: (M, K) × (K, N) where K = in_features, N = out_features
            N_dim, K_dim = w.shape
            records.append({
                "layer_type": layer_type,
                "layer_key": key,
                "layer_idx": layer_idx,
                "K": int(K_dim),
                "N": int(N_dim),
            })

with open(_OUT_JSON, "w") as f:
    json.dump(records, f, indent=2)

print(f"\nExtracted {len(records)} Linear weight shapes")
print(f"Saved to {_OUT_JSON}")

from collections import Counter
type_counts = Counter(r["layer_type"] for r in records)
print("\nPer layer type:")
for t, c in sorted(type_counts.items()):
    sample = next(r for r in records if r["layer_type"] == t)
    print(f"  {t:12s}  ×{c:3d}   (K={sample['K']}, N={sample['N']})")

