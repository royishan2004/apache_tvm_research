import json
import sys
import pandas as pd
from tabulate import tabulate
from pathlib import Path
from research.workloads.bert.bert_shapes import (
    HIDDEN, FF, M_LIST,
    qkv_shape, mlp_expanded_shape, mlp_compressed_shape,
)
from research.workloads.common.data_aggregator_client import resolve_profile
import subprocess
import signal


def _fmt_latency_std(latency, std):
    """Format a latency cell as 'latency ± std' in microseconds.

    Falls back to latency-only when std is missing (older result files).
    """
    if pd.isna(latency):
        return ""
    if pd.isna(std):
        return f"{latency:.2f}"
    return f"{latency:.2f} ± {std:.2f}"

RESULTS_FILE = Path("research/results/bert_matmul_results.json")

# Fallback: try old filename for backward compatibility
if not RESULTS_FILE.exists():
    RESULTS_FILE = Path("research/results/bert_qkv_results.json")

if not RESULTS_FILE.exists():
    raise FileNotFoundError(f"No results found at {RESULTS_FILE}")

with open(RESULTS_FILE, "r") as f:
    data = json.load(f)

if not data:
    print("Results file is empty.")
    exit(0)

df = pd.DataFrame(data)

if "profile" not in df.columns:
    df["profile"] = "unknown"

# if "kernel" not in df.columns:
#   df["kernel"] = "qkv"

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.2f}".format)

# Optional CLI filter: python3 -m research.analysis.print_qkv_mlp_results [kernel]
if len(sys.argv) > 1:
    kernel_filter = sys.argv[1]
    df = df[df["kernel"] == kernel_filter]
    if df.empty:
        print(f"No results for kernel '{kernel_filter}'.")
        exit(0)

for kernel_name, group in df.groupby("kernel", sort=False):
    print(f"\n{'=' * 60}")
    print(f"  BERT {kernel_name.upper()} MatMul  —  latency (µs)")
    print(f"{'=' * 60}")

    shape_map = {
        "qkv": qkv_shape,
        "mlp_expand": mlp_expanded_shape,
        "mlp_reduce": mlp_compressed_shape,
    }
    example_M = M_LIST[0] if M_LIST else None
    if kernel_name in shape_map and example_M is not None:
        _, K_example, N_example = shape_map[kernel_name](example_M)
        print(f"  HIDDEN = {HIDDEN}    FF = {FF}")
        print(f"  Kernel shape example (M={example_M}): K={K_example}  N={N_example}")
        print(f"  M sweep = {M_LIST}")
        print()

    # Show which CPU profile(s) these rows belong to (group may contain multiple profiles)
    group_profiles = sorted({str(p) for p in group["profile"].dropna().unique() if str(p).strip() and str(p).lower() != "unknown"})
    if not group_profiles:
        group_profiles = [resolve_profile()]
    print(f"  Profile(s): {', '.join(group_profiles)}")

    latency_pivot = group.pivot_table(
        index="variant",
        columns="M",
        values="latency_us",
        aggfunc="median",       # keep best if duplicates
    )
    if "std_us" in group.columns:
        std_pivot = group.pivot_table(
            index="variant",
            columns="M",
            values="std_us",
            aggfunc="median",
        ).reindex_like(latency_pivot)
    else:
        std_pivot = pd.DataFrame(index=latency_pivot.index, columns=latency_pivot.columns)

    display_pivot = latency_pivot.copy()
    for col in display_pivot.columns:
        display_pivot[col] = [
            _fmt_latency_std(lat, std)
            for lat, std in zip(latency_pivot[col], std_pivot[col])
        ]

    display_pivot.columns = [f"M={int(c)}" for c in display_pivot.columns]
    display_pivot = display_pivot.reset_index()

    print("  Cell format: latency ± std (µs)")

    print(tabulate(display_pivot, headers="keys", tablefmt="grid", showindex=False))
    print()

# Prompt user whether to display plots (timeout after 30s)
def _input_timeout(seconds=30):
    def _handler(signum, frame):
        raise TimeoutError
    old_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        return input("Show plots for these results now? [y/N]: ").strip().lower()
    except (EOFError, KeyboardInterrupt, TimeoutError):
        return "n"
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

ans = _input_timeout(30)

if ans in ("y", "yes"):
    print("Launching plot viewer...")
    try:
        subprocess.run([sys.executable, "-m", "research.analysis.plot_qkv_mlp_results"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Plot script failed: {e}")
else:
    print("Skipping plots.")
