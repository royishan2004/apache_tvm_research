import json

H = 768  # BERT hidden size

with open("bert_matmul_shapes_raw.json") as f:
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

with open("bert_matmul_shapes_qkv.json", "w") as f:
    json.dump(qkv_only, f, indent=2)

print(f"Filtered {len(qkv_only)} Q/K/V projection MatMul ops")
print("Saved to bert_matmul_shapes_qkv.json")
