import torch
from transformers import BertModel, BertTokenizer

print("Loading tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

print("Loading model...")
model = BertModel.from_pretrained("bert-base-uncased")

model.eval()

print("BERT loaded successfully")

# Simple sanity run
text = "TVM optimization of transformer matrix multiplication"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

print("Output last_hidden_state shape:", outputs.last_hidden_state.shape)
