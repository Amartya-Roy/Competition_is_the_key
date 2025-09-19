#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import yaml

# Load figure paths (optional)
try:
    cfg = yaml.safe_load(open("config.yml"))
    FIG_PNG = cfg["outputs"]["figure_png"]
    FIG_PDF = cfg["outputs"]["figure_pdf"]
    FIG_TIES = cfg["outputs"]["figure_ties_png"]
except Exception:
    FIG_PNG = "composite_grouped_bars_large_fonts.png"
    FIG_PDF = "composite_grouped_bars_large_fonts.pdf"
    FIG_TIES = "composite_grouped_bars_ties_marked.png"

datasets = ["Asia", "Sachs", "Lucas", "Child", "Alarm", "Hepar2", "Dream", "Andes"]
methods = [
    "KCRL","NOTEARS","GOLEM","Gran-DAG","RL-BIC2","ICALiNGAM",
    "DirectLiNGAM","PC","GES","CORL","Ours (Using Gran-DAG)","Ours (Using GES)"
]

score = {
    "Asia":   [0.52, 0.13, 0.19, 0.42, 0.37, 0.26, 0.57, 0.54, 1.00, np.nan, 0.47, 1.00],
    "Sachs":  [0.32, 0.26, 0.13, 0.30, 0.21, 0.26, 0.23, 0.20, 0.39, 0.18, 0.30, 0.40],
    "Lucas":  [0.35, 0.33, 0.35, 0.23, 0.26, 0.20, 0.32, 0.72, 1.00, np.nan, 0.18, 1.00],
    "Child":  [0.13, 0.18, 0.12, 0.45, 0.30, 0.30, 0.11, 0.13, 0.17, np.nan, 0.47, 0.18],
    "Alarm":  [0.24, 0.26, 0.26, 0.18, 0.20, 0.43, 0.30, 0.36, 0.38, np.nan, 0.26, 0.43],
    "Hepar2": [np.nan, 0.01, np.nan, 0.30, np.nan, 0.24, 0.35, 0.20, 0.42, np.nan, 0.35, 0.43],
    "Dream":  [np.nan, 0.03, np.nan, 0.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.11, np.nan],
    "Andes":  [np.nan, 0.03, np.nan, 0.04, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.08, np.nan],
}
data = np.array([[score[d][i] for d in datasets] for i in range(len(methods))])

# Styling
TITLE_FONTSIZE = 20; LABEL_FONTSIZE = 18; TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 16; LEGEND_TITLE_FONTSIZE = 16

colors = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b",
    "#e377c2","#7f7f7f","#bcbd22","#4169E1","#17becf","#ff7f0e"
]
x = np.arange(len(datasets)); width = 0.065
fig, ax = plt.subplots(figsize=(16, 8))

for i, method in enumerate(methods):
    vals = data[i]
    ax.bar(x + i*width - (len(methods)/2)*width, vals, width,
           label=method, color=colors[i],
           hatch="//" if "Ours" in method else None,
           edgecolor="black" if "Ours" in method else None)

# Stars for ties
for j, d in enumerate(datasets):
    col = data[:, j]; max_val = np.nanmax(col)
    best_methods = np.where(col == max_val)[0]
    for bm in best_methods:
        ax.text(x[j] + bm*width - (len(methods)/2)*width, max_val + 0.025,
                "★", ha="center", va="bottom", fontsize=20, color="black")

ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=TICK_FONTSIZE)
ax.set_ylabel("Composite Score", fontsize=LABEL_FONTSIZE)
ax.set_xlabel("Datasets", fontsize=LABEL_FONTSIZE)
ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)
ax.set_title("Composite Score by Dataset and Method\n(Ours highlighted; ★ = best per dataset, ties marked)",
             fontsize=TITLE_FONTSIZE, pad=12)

ax.legend(title="Methods", fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE,
          bbox_to_anchor=(1.02, 1), loc="upper left", ncol=1, frameon=True)

plt.tight_layout()
plt.savefig(FIG_PNG, dpi=300, bbox_inches="tight")
plt.savefig(FIG_PDF, bbox_inches="tight")
plt.savefig(FIG_TIES, dpi=300, bbox_inches="tight")
print(f"Saved: {FIG_PNG}, {FIG_PDF}, {FIG_TIES}")
