import json
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import os
from pathlib import Path
from research.workloads.bert.bert_shapes import (
    HIDDEN, FF, M_LIST,
    qkv_shape, mlp_expanded_shape, mlp_compressed_shape,
)

RESULTS_FILE = Path("research/results/bert_matmul_results.json")
if not RESULTS_FILE.exists():
    RESULTS_FILE = Path("research/results/bert_qkv_results.json")
if not RESULTS_FILE.exists():
    raise FileNotFoundError(f"No results found at {RESULTS_FILE}")

with open(RESULTS_FILE, "r") as f:
    data = json.load(f)

if not data:
    print("Results file is empty.")
    sys.exit(0)

df = pd.DataFrame(data)
if "kernel" not in df.columns:
    df["kernel"] = "qkv"

save_mode = "--save" in sys.argv
args = [a for a in sys.argv[1:] if a != "--save"]

if args:
    kernel_filter = args[0]
    df = df[df["kernel"] == kernel_filter]
    if df.empty:
        print(f"No results for kernel '{kernel_filter}'.")
        sys.exit(0)

OUTPUT_DIR = Path("research/results/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
backend = plt.get_backend().lower()
headless = (backend in ("agg", "pdf", "svg", "ps")) or (os.environ.get("DISPLAY") is None)
if headless and not save_mode:
    print(f"Matplotlib backend '{backend}' or missing DISPLAY detected; interactive plotting is not available.")
    print("To save plots instead, re-run with the --save flag. To view plots interactively, run in a graphical session with DISPLAY set.")
    sys.exit(1)

SHAPE_MAP = {
    "qkv": qkv_shape,
    "mlp_expand": mlp_expanded_shape,
    "mlp_reduce": mlp_compressed_shape,
}


figs = []
for kernel_name, group in df.groupby("kernel", sort=False):
    pivot = group.pivot_table(
        index="variant",
        columns="M",
        values="latency_us",
        aggfunc="min",
    )
    pivot = pivot.sort_index()  # alphabetical variants

    fig, ax = plt.subplots(figsize=(10, 6))

    m_values = sorted(pivot.columns)
    for variant in pivot.index:
        # pivot values are in microseconds; keep µs for plotting
        latencies_us = pivot.loc[variant, m_values].astype(float)
        latencies = latencies_us
        ax.plot(m_values, latencies, marker="o", linewidth=1.8, label=variant)

    subtitle_parts = [f"HIDDEN={HIDDEN}  FF={FF}"]
    if kernel_name in SHAPE_MAP:
        ex_M = M_LIST[0] if M_LIST else m_values[0]
        _, K_ex, N_ex = SHAPE_MAP[kernel_name](ex_M)
        subtitle_parts.append(f"K={K_ex}  N={N_ex}")

    ax.set_title(
        f"BERT {kernel_name.upper()} MatMul — Latency vs M\n"
        f"({', '.join(subtitle_parts)})",
        fontsize=13,
    )
    ax.set_xlabel("M (batch × seq_len)", fontsize=11)
    ax.set_ylabel("Latency (µs)", fontsize=11)
    ax.set_xticks(m_values)
    ax.legend(title="Schedule Variant", fontsize=8, title_fontsize=9,
              loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)
    # Avoid scientific offset on Y axis (so 1e6 multipliers aren't confusing)
    ax.yaxis.set_major_formatter(mtick.StrMethodFormatter("{x:,.1f}"))
    fig.tight_layout()

    if save_mode:
        out_path = OUTPUT_DIR / f"bert_{kernel_name}_matmul.png"
        fig.savefig(out_path, dpi=150)
        print(f"Saved {out_path}")
        plt.close(fig)
    else:
        figs.append(fig)

import numpy as np

heatmap_rows = []
kernel_groups = []   # track kernel name per row for separator lines
for kernel_name, group in df.groupby("kernel", sort=False):
    pivot = group.pivot_table(
        index="variant", columns="M", values="latency_us", aggfunc="min",
    ).sort_index()
    for variant in pivot.index:
        row_label = f"{kernel_name} / {variant}"
        heatmap_rows.append(pd.Series(pivot.loc[variant], name=row_label))
        kernel_groups.append(kernel_name)

heatmap_df = pd.DataFrame(heatmap_rows)
heatmap_df.columns = [int(c) for c in heatmap_df.columns]
heatmap_df = heatmap_df[sorted(heatmap_df.columns)]

n_rows, n_cols = heatmap_df.shape

# Size the figure so it fits a typical screen (~1920×1080) while staying readable
fig_w = min(18, max(10, n_cols * 1.8))
fig_h = min(10, max(5, n_rows * 0.55 + 2.5))
fig_all, ax_hm = plt.subplots(figsize=(fig_w, fig_h))

from matplotlib.colors import LogNorm
vmin = max(heatmap_df.min().min(), 1)
vmax = heatmap_df.max().max()
im = ax_hm.imshow(
    heatmap_df.values, aspect="auto",
    cmap="YlOrRd", norm=LogNorm(vmin=vmin, vmax=vmax),
)

cell_font = max(5.5, min(8, 120 / max(n_rows, n_cols)))
for i in range(n_rows):
    for j in range(n_cols):
        val = heatmap_df.iloc[i, j]
        if pd.notna(val):
            rel = (np.log(val) - np.log(vmin)) / (np.log(vmax) - np.log(vmin)) if vmax > vmin else 0
            txt_color = "white" if rel > 0.55 else "black"
            ax_hm.text(
                j, i, f"{val:,.0f}",
                ha="center", va="center",
                fontsize=cell_font, color=txt_color,
            )

prev_kernel = kernel_groups[0]
for ri in range(1, n_rows):
    if kernel_groups[ri] != prev_kernel:
        ax_hm.axhline(y=ri - 0.5, color="black", linewidth=1.8)
        prev_kernel = kernel_groups[ri]

groups = []
curr = kernel_groups[0]
start = 0
for i, k in enumerate(kernel_groups):
    if k != curr:
        groups.append((curr, start, i))
        curr = k
        start = i
groups.append((curr, start, n_rows))

fig_all.subplots_adjust(left=0.22)

for name, s, e in groups:
    mid = (s + e - 1) / 2.0
    # compute normalized y (axes coordinates, 0..1 top->bottom for transAxes)
    y_norm = 1.0 - (mid + 0.5) / float(n_rows)
    # Place subtitle to the right of the heatmap (just past rightmost column)
    x_pos = 1.01
    ax_hm.text(
        x_pos, y_norm, name.upper(), transform=ax_hm.transAxes,
        ha="left", va="center", fontsize=10, fontweight="bold",
        color="#333333", rotation=270,
    )


ax_hm.set_xticks(range(n_cols))
ax_hm.set_xticklabels([str(c) for c in heatmap_df.columns], fontsize=9)
ax_hm.set_yticks(range(n_rows))
short_labels = [lbl.split(" / ", 1)[-1] if " / " in lbl else lbl for lbl in heatmap_df.index]
ax_hm.set_yticklabels(short_labels, fontsize=max(6, min(9, 100 / n_rows)))
ax_hm.set_xlabel("M (batch × seq_len)", fontsize=11)
ax_hm.set_ylabel("Variant", fontsize=11)
ax_hm.set_title(
    f"BERT MatMul — All Kernels (Heatmap)\n"
    f"(HIDDEN={HIDDEN}  FF={FF})  ·  cell values in µs",
    fontsize=13,
)

cbar = fig_all.colorbar(im, ax=ax_hm, pad=0.08, shrink=0.85)
cbar.set_label("Latency (µs — log scale)", fontsize=10)
fig_all.tight_layout()

if save_mode:
    out_path = OUTPUT_DIR / "bert_all_kernels_matmul.png"
    fig_all.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig_all)
else:
    figs.append(fig_all)

if not save_mode and figs:
    plt.show()
