"""Corrected copy of Figure 1 (methodology diagram) for the manuscript.

The layout, colours and numbers are taken unchanged from the approved figure (make_paper_figures.py in the
research repository). Only the wording is corrected to match the frozen implementation and the verification audit:
  * "discarded/removed" -> "excluded from generation" (original rows stay in the training set);
  * the quota no longer claims "at least 1 for every cluster" (the frozen rounding rule has no floor; every
    retained cluster received a positive quota in all 190 A-main folds, which is what the figure now states);
  * the partner is drawn from the 5 nearest *retained* minority points;
  * "removes starvation" -> observed reduction, with fold counts.
Output is written only to the manuscript folder.
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = str(PROJECT_ROOT / "work" / "generated_figures")
os.makedirs(OUT, exist_ok=True)

W = 6.9
INK, INK2, SURF = "#0b0b0b", "#52514e", "#ffffff"
COL = {"RCA-SMOTE": "#2a78d6", "AW-SMOTE": "#eb6834"}
TINT = {"RCA-SMOTE": "#e9f1fb", "AW-SMOTE": "#fdeee7"}
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "text.color": INK, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.03, "mathtext.fontset": "dejavusans"})

fig, ax = plt.subplots(figsize=(W, 4.15))
ax.set_xlim(0, 100)
ax.set_ylim(-2, 68)
ax.axis("off")
cols_x = [12.0, 37.3, 62.6, 87.9]
BW, BH = 22.6, 17.0
heads = ["1. Cluster minority points", "2. Allocate the budget", "3. Generate points",
         "Observed outcome"]
for x, h in zip(cols_x, heads):
    ax.text(x, 66.0, h, ha="center", va="center", fontsize=7.6, fontweight="bold", color=INK)


def box(x, y, text, method, key=False):
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=TINT[method], ec=COL[method], lw=1.8 if key else 0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.3, color=INK, linespacing=1.35)
    if key:
        ax.text(x + BW / 2 - 1.0, y + BH / 2 + 0.2, "transferable part", ha="right", va="center", fontsize=5.8,
                color=SURF, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.6", fc=COL[method], ec="none"))


def arrow(x0, x1, y, method):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, lw=1.0, color=COL[method]))


lanes = [("AW-SMOTE (released implementation)", 45.0,
          ["DBSCAN on minority points\n(eps = 65th percentile).\nEvery DBSCAN noise point is\n"
           "excluded from generation:\n34.9% of minority points",
           "Quota proportional to the\nboundary weight $W_j$.\nA cluster with no majority\nneighbour ($W_j = 0$) gets 0",
           "Interpolation between two\npoints of the same cluster,\n" r"$\lambda \sim U(0.2,\, 0.8)$",
           "Smallest sub-group gets\nno synthetic point in\n34.2% of folds (65/190)"], [False, False, False, False]),
         ("RCA-SMOTE", 16.5,
          ["DBSCAN (eps = 90th pct.).\nRescue: a noise point is\nexcluded only if its 5 nearest\n"
           "training neighbours are all\nmajority: 2.8% excluded",
           "Quota by cluster size:\n" r"$n_j \propto |C_j|^{1/2}$" "\n(rounded; positive for every\ncluster in all 190 folds)",
           "SMOTE-style interpolation:\nseed in the cluster, partner\namong the 5 nearest retained\n"
           r"minority points, $\lambda \sim U(0, 1)$",
           "Smallest sub-group gets\nno synthetic point in\n0.5% of folds (1/190)"], [True, True, False, False])]
for method, y, texts, keys in lanes:
    key = "RCA-SMOTE" if method.startswith("RCA") else "AW-SMOTE"
    ax.text(cols_x[0] - BW / 2, y + BH / 2 + 4.0, method, ha="left", va="center", fontsize=7.8,
            fontweight="bold", color=COL[key])
    for i, (x, t, k) in enumerate(zip(cols_x, texts, keys)):
        box(x, y, t, key, key=k)
        if i < 3:
            arrow(x + BW / 2 + 0.4, cols_x[i + 1] - BW / 2 - 0.4, y, key)
ax.text(50, 0.6, "Excluded points stay in the training set; they are only left out of the pool used to create "
        "synthetic points.\nAdding the two transferable parts to KMeans-SMOTE and DBSMOTE reduced starvation from "
        "60% and 34% of folds to 0.5%.",
        ha="center", va="center", fontsize=6.4, color=INK2, linespacing=1.45)
fig.savefig(os.path.join(OUT, "fig1_methodology.pdf"))
fig.savefig(os.path.join(OUT, "fig1_methodology.png"), dpi=300)
print("saved fig1_methodology")
