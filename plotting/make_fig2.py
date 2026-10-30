"""The 7 main figures of the paper, as one consistent set (PDF + 600-dpi PNG in paper_figures/).

Every number comes from the saved result files (results/stage3, results/stage4). The only computation is
Figure 2, the 2D example, which re-runs the fixed-seed demonstration of stage 1 (seed 0), exactly as
make_report_figures.py does. Nothing in the methods or results is changed.

  python make_paper_figures.py
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
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

OUT = str(PROJECT_ROOT / "work" / "generated_figures")
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ one style for all figures
W = 6.9                                  # full text width in inches (17.5 cm)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
GRAY = "#b4b2ab"
COL = {"RCA-SMOTE": "#2a78d6", "AW-SMOTE": "#eb6834", "SMOTE": "#1baf7a", "None": GRAY,
       "KMeans-SMOTE": "#6250d6", "DBSMOTE": "#e87ba4", "NearMiss-1": "#eda100",
       "ClusterCentroids": "#008300", "RandomOverSampler": "#e34948"}
TINT = {"RCA-SMOTE": "#e9f1fb", "AW-SMOTE": "#fdeee7"}
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})


def clean(ax, grid_axis="x"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=600)
    plt.close(fig)
    print("saved", name)


def fill_none(df, col="method"):
    df[col] = df[col].fillna("None")
    return df


DEV = ["ecoli", "led7", "shuttle", "texture", "wine_quality_white"]

# ================================================================== Figure 2: 2D example
from exp1_synthetic import draw, synth_per_sub
from rc_aw_smote import AWSMOTE
from rca_smote import RCASMOTE

rng = np.random.RandomState(0)
X_tr, y_tr, s_tr = draw(rng, "train")
Xs = StandardScaler().fit(X_tr).transform(X_tr)
fig, axes = plt.subplots(1, 2, figsize=(W, 3.15), sharex=True, sharey=True)
SUBS = [(0, "o", "S1: large"), (1, "^", "S2: rare, compact"), (2, "s", "S3: rare, sparse")]
counts = {}
for k, (ax, (m, smp)) in enumerate(zip(axes, [("AW-SMOTE", AWSMOTE(random_state=0)),
                                               ("RCA-SMOTE", RCASMOTE(random_state=0))])):
    Xr, yr = smp.fit_resample(Xs, y_tr)
    syn = Xr[len(Xs):]
    c = synth_per_sub(Xs, y_tr, s_tr, Xr, yr)
    counts[m] = c
    ax.scatter(*Xs[y_tr == 0].T, s=3, c="#d3d1cc", lw=0, zorder=1)
    ax.scatter(*syn.T, s=4, c=COL[m], alpha=0.6, lw=0, zorder=2)
    LBL = {0: (1.75, 1.05), 1: (-1.25, 2.35), 2: (-1.25, -1.25)}
    for s, mk, lab in SUBS:              # real minority points: hollow, so new points under them stay visible
        ax.scatter(*Xs[s_tr == s].T, s=15, marker=mk, facecolor="none", edgecolor=INK, linewidth=0.7, zorder=3)
        ax.text(*LBL[s], f"S{s + 1}", fontsize=7.5, fontweight="bold", color=INK, zorder=4)
    ax.set_title(f"new points: S1 = {c[0]},  S2 = {c[1]},  S3 = {c[2]}", fontsize=8, color=INK, loc="left", pad=3)
    ax.text(0.0, 1.075, m, transform=ax.transAxes, fontsize=8.5, color=COL[m], fontweight="bold", va="bottom")
    ax.set_xlabel("Feature 1 (standardised)")
    clean(ax, "both")
    panel(ax, "ab"[k], x=-0.01, y=1.075)
axes[0].set_ylabel("Feature 2 (standardised)")
handles = [Line2D([], [], marker="o", ls="none", ms=4, mfc="#d3d1cc", mec="none", label="Majority class"),
           Line2D([], [], marker="o", ls="none", ms=4, mfc=COL["AW-SMOTE"], mec="none", label="New points (AW-SMOTE)"),
           Line2D([], [], marker="o", ls="none", ms=4, mfc=COL["RCA-SMOTE"], mec="none", label="New points (RCA-SMOTE)")]
handles += [Line2D([], [], marker=mk, ls="none", ms=5, mfc="none", mec=INK, mew=0.8, label=lab) for _, mk, lab in SUBS]
fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.08),
           handletextpad=0.3, columnspacing=1.0)
save(fig, "fig2_synthetic_example")
print("Figure 2 counts:", counts)

