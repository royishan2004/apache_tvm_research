import json
import os

H = 768  # BERT hidden size

_HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_HERE, "bert_matmul_shapes_raw.json")) as f:
    data = json.load(f)

qkv_only = []

for item in data:
    inp = item["input"]     # e.g. [B, S, 768] or [B*S, 768]
    w = item["weight"]      # e.g. [768, 768]

    # Normalize input to 2D
    # (B, S, H) -> (B*S, H)
    if len(inp) == 3:
        M, K = inp[0] * inp[1], inp[2]
    elif len(inp) == 2:
        M, K = inp
    else:
        continue

    N, K_w = w

    # Q / K / V projection:
    # (M, 768) x (768, 768)
    if K == H and K_w == H and N == H:
        qkv_only.append({
            "M": M,
            "K": H,
            "N": H
        })

out_path = os.path.join(_HERE, "bert_matmul_shapes_qkv.json")
with open(out_path, "w") as f:
    json.dump(qkv_only, f, indent=2)

print(f"Filtered {len(qkv_only)} Q/K/V projection MatMul ops")
print(f"Saved to {out_path}")

# Print summary of unique shapes
unique = {(e["M"], e["K"], e["N"]) for e in qkv_only}
print(f"Unique shapes: {len(unique)}")
for s in sorted(unique):
    print(f"  M={s[0]}, K={s[1]}, N={s[2]}")
