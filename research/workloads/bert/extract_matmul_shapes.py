import torch
import json
from transformers import BertModel, BertTokenizer
import torch.nn as nn

# ---------------------------------
# Storage
# ---------------------------------

linear_records = []

# ---------------------------------
# Hook nn.Linear.forward
# ---------------------------------

orig_linear_forward = nn.Linear.forward

def logging_linear_forward(self, input):
    # input: (batch, seq, in_features) OR (N, in_features)
    # weight: (out_features, in_features)
    if input.dim() >= 2:
        linear_records.append({
            "input": list(input.shape),
            "weight": list(self.weight.shape),
            "bias": self.bias is not None
        })
    return orig_linear_forward(self, input)

nn.Linear.forward = logging_linear_forward

# ---------------------------------
# Load BERT
# ---------------------------------

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()

inputs = tokenizer(
    "This is a test sentence for matrix multiplication extraction.",
    return_tensors="pt"
)

with torch.no_grad():
    model(**inputs)

# ---------------------------------
# Save results
# ---------------------------------

with open("bert_matmul_shapes_raw.json", "w") as f:
    json.dump(linear_records, f, indent=2)

print(f"Captured {len(linear_records)} Linear layers")
print("Saved to bert_matmul_shapes_raw.json")
