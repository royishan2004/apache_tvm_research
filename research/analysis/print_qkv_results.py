import json
import pandas as pd
from tabulate import tabulate
from pathlib import Path

RESULTS_FILE = Path("research/results/bert_qkv_results.json")

if not RESULTS_FILE.exists():
    raise FileNotFoundError(f"No results found at {RESULTS_FILE}")

with open(RESULTS_FILE, "r") as f:
    data = json.load(f)

if not data:
    print("Results file is empty.")
    exit(0)

df = pd.DataFrame(data)

cols = [
    "variant",
    "latency_us",
    "B",
    "M",
    "N",
    "runs",
    "target",
    "timestamp",
]
df = df[cols]

#df = df.sort_values(by="latency_us")

pd.set_option("display.max_rows", None)
pd.set_option("display.float_format", "{:.2f}".format)

print("\n=== BERT QKV MatMul Results ===\n")
print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))
