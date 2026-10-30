"""Redraw Figures 3, 4, 6, 7 and 8 of the manuscript for the updated RCA-SMOTE aggregate results.

Style follows rca-smote/make_paper_figures.py (same fonts, colours and sizes).

Data sources
- Table 2 / Table 3 means: the values printed in overleaf/main.tex (RCA-SMOTE: the researcher's updated
  aggregate measurements; all other methods: unchanged).
- Fig. 3: starvation rates and the smallest-subgroup recall of the other methods from
  rca-smote/results/stage4/audit_starvation.csv (earlier run); RCA-SMOTE's recall point uses the updated R_G.
- Fig. 6: transfer bars from rca-smote/results/stage4 (earlier run, = Table 6); reference lines from Table 2.
- Figs. 4, 7, 8: Table 2 means only (no per-dataset values, ranks or tests).

  python tools/make_updated_figures.py      (run from RCA_SMOTE_Final_Paper/)
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
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(PROJECT_ROOT / "work" / "generated_figures")
RES = str(PROJECT_ROOT / "results" / "supporting")
os.makedirs(OUT, exist_ok=True)

W = 6.9
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
GRAY = "#b4b2ab"
COL = {"RCA-SMOTE": "#2a78d6", "AW-SMOTE": "#eb6834", "SMOTE": "#1baf7a", "None": GRAY,
       "KMeans-SMOTE": "#6250d6", "DBSMOTE": "#e87ba4"}
plt.rcParams.update({
    "font.family": "Arial", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.6, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.03})

# ------------------------------------------------------------------ Table 2 (Group A-main, 19 datasets)
T2_COLS = ["wBA", "R_G", "G-mean", "F1", "MCC", "Prec", "Spec"]
T2 = pd.read_csv(PROJECT_ROOT / 'results/main_tables/table2_groupA_final.csv').set_index('Method')
T2 = T2.rename(columns={'R_G':'R_G','G_mean':'G-mean','Precision':'Prec','Specificity':'Spec'})[T2_COLS]
assert len(T2) == 16
HL = ("RCA-SMOTE", "AW-SMOTE", "SMOTE")


def clean(ax, grid_axis="x"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def panel(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    plt.close(fig)
    print("saved", name)


def mcolor(m):
    return COL.get(m, GRAY) if m in HL else GRAY


# ================================================================== Figure 3: starvation audit
A = pd.read_csv(os.path.join(RES, "stage4", "audit_starvation.csv"), index_col=0)
NAME = {"KMeans-SMOTE (sv)": "KMeans-SMOTE (smote-variants)", "KMeans-SMOTE (imblearn) [stage 3]":
        "KMeans-SMOTE (imbalanced-learn)", "AW-SMOTE [stage 3]": "AW-SMOTE", "SMOTE [stage 3]": "SMOTE",
        "RCA-SMOTE [stage 3]": "RCA-SMOTE"}
A.index = [NAME.get(i, i) for i in A.index]
# the only change to the data: RCA-SMOTE's recall point takes its updated Table 2 value
A.loc["RCA-SMOTE", "recall_smallest_sub"] = T2.loc["RCA-SMOTE", "R_G"]
# consistency with Table 2 for the methods that appear in both
for m in ["SMOTE", "AW-SMOTE", "KMeans-SMOTE (imbalanced-learn)"]:
    t2name = "KMeans-SMOTE" if m.startswith("KMeans") else m
    assert round(A.loc[m, "recall_smallest_sub"], 3) == T2.loc[t2name, "R_G"], m


def scol(name):
    for key in ["RCA-SMOTE", "AW-SMOTE", "DBSMOTE", "KMeans-SMOTE"]:
        if name.startswith(key):
            return COL[key]
    return COL["SMOTE"] if name == "SMOTE" else GRAY


A = A.sort_values("runs_smallest_zero_pct", ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw=dict(width_ratios=[1.05, 1]))
ax = axes[0]
v = A.runs_smallest_zero_pct
ax.barh(range(len(v)), v.values, color=[scol(m) for m in v.index], height=0.68, edgecolor=SURF, lw=0.6)
for i, x in enumerate(v.values):
    ax.text(x + 0.8, i, f"{x:.1f}", va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(v)), v.index)
ax.set_xlim(0, 68)
ax.set_xlabel("Runs where the smallest sub-group\ngot no new points (%)")
clean(ax, "x")
panel(ax, "a", x=-0.62)
ax = axes[1]
for m, r in A.iterrows():
    ax.scatter(r.runs_smallest_zero_pct, r.recall_smallest_sub, s=30, c=scol(m), edgecolor=SURF, lw=0.8, zorder=3)
LAB = {"RCA-SMOTE": (13, 0.676), "MWMOTE": (13, 0.662), "CBSO": (13, 0.651), "SMOTE": (13, 0.640),
       "Cluster-SMOTE": (13, 0.624), "A-SUWO": (10, 0.580), "CURE-SMOTE": (10, 0.555),
       "AW-SMOTE": (38, 0.567), "KMeans-SMOTE (imbalanced-learn)": (28.5, 0.526), "DBSMOTE": (38, 0.531),
       "SOMO": (24, 0.510), "KMeans-SMOTE (smote-variants)": (57, 0.513)}
for m, r in A.iterrows():
    tx, ty = LAB[m]
    right = tx < r.runs_smallest_zero_pct
    txt = m.replace(" (imbalanced-learn)", "\n(imbalanced-learn)").replace(" (smote-variants)", "\n(smote-variants)")
    ax.annotate(txt, (r.runs_smallest_zero_pct, r.recall_smallest_sub), xytext=(tx, ty), textcoords="data",
                linespacing=1.0, fontsize=6.5, color=COL[m] if m in HL else INK2,
                fontweight="bold" if m in HL else "normal", ha="right" if right else "left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=3))
ax.set_xlabel("Runs where the smallest sub-group\ngot no new points (%)")
ax.set_ylabel("Recall of the smallest sub-group")
ax.set_xlim(-3, 66)
ax.set_ylim(0.48, 0.69)
clean(ax, "both")
panel(ax, "b", x=-0.12)
fig.tight_layout(w_pad=1.2)
save(fig, "fig3_starvation")

# ================================================================== Figure 4: aggregate Group A comparison
M4 = ["RCA-SMOTE", "SMOTE", "AW-SMOTE"]
fig, axes = plt.subplots(1, 2, figsize=(W * 0.72, 1.55))
for k, (ax, (col, lab, lo, hi)) in enumerate(zip(axes, [("wBA", "wBA (axis starts at 0.80)", 0.80, 0.86),
                                                        ("R_G", "Smallest-subgroup recall (axis starts at 0.50)",
                                                         0.50, 0.70)])):
    vals = T2.loc[M4, col].values
    ys = np.arange(len(M4))[::-1]
    ax.barh(ys, vals - lo, left=lo, height=0.62, color=[COL[m] for m in M4], edgecolor=SURF, lw=0.6)
    for y, val in zip(ys, vals):
        ax.text(val + (hi - lo) * 0.012, y, f"{val:.3f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(ys, M4 if k == 0 else [""] * len(M4))
    for t, m in zip(ax.get_yticklabels(), M4):
        t.set_color(COL[m]); t.set_fontweight("bold")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(lab)
    clean(ax, "x")
    panel(ax, "ab"[k], x=-0.03 if k else -0.30)
fig.tight_layout(w_pad=1.0)
save(fig, "fig4_dataset_comparison")


# ================================================================== Figures 7 and 8: all 16 methods
def method_bars(col, lo, hi, xlabel, name, tick):
    # descending with ties kept in Table 2 order, then reversed so the largest is at the top
    s = T2[col].sort_values(ascending=False, kind="stable")[::-1]
    fig, ax = plt.subplots(figsize=(W * 0.58, 3.05))
    ys = np.arange(len(s))
    ax.barh(ys, s.values - lo, left=lo, height=0.68, color=[mcolor(m) for m in s.index], edgecolor=SURF, lw=0.6)
    for y, (m, val) in zip(ys, s.items()):
        ax.text(val + (hi - lo) * 0.008, y, f"{val:.3f}", va="center", fontsize=6.6,
                color=COL[m] if m in HL else INK2, fontweight="bold" if m in HL else "normal")
    ax.set_yticks(ys, s.index)
    for t in ax.get_yticklabels():
        m = t.get_text()
        if m in HL:
            t.set_color(COL[m]); t.set_fontweight("bold")
    ax.set_xlim(lo, hi)
    ax.set_xticks(tick)
    ax.set_ylim(-0.6, len(s) - 0.4)
    ax.set_xlabel(xlabel)
    clean(ax, "x")
    fig.tight_layout()
    save(fig, name)


method_bars("R_G", 0.0, 0.80, "Mean smallest-subgroup recall $R_G$ (19 datasets)", "fig7_subgroup_recall",
            np.arange(0, 0.81, 0.1))
method_bars("wBA", 0.70, 0.87, "Mean wBA (19 datasets; axis starts at 0.70)", "fig8_rankings",
            np.arange(0.70, 0.871, 0.02))

# ================================================================== Figure 6: transfer of the corrections
DEV = ["ecoli", "led7", "shuttle", "texture", "wine_quality_white"]
C = pd.read_csv(os.path.join(RES, "stage4", "cluster_all_folds.csv.gz"))
C = C[~C.dataset.isin(DEV)]
CM = C.groupby(["dataset", "classifier", "method"])[["wBA", "Precision"]].mean().groupby("method").mean()
SC = pd.read_csv(os.path.join(RES, "stage4", "cluster_starvation_folds.csv.gz"))
ST = SC[~SC.dataset.isin(DEV)].groupby("method").smallest_sub_got_zero.mean() * 100
VAR = ["", " + rescue", " + quota", " + rescue + quota"]
VLAB = ["none", "rescue", "quota", "both"]
# Table 6 values (earlier run) that the bars must reproduce
T6 = {"KMeans-SMOTE (sv)": dict(starv=[60.0, 35.3, 51.6, 0.5], wBA=[0.806, 0.813, 0.810, 0.842],
                                Precision=[0.798, 0.787, 0.789, 0.740]),
      "DBSMOTE": dict(starv=[34.2, 0.5, 34.2, 0.5], wBA=[0.815, 0.837, 0.816, 0.838],
                      Precision=[0.781, 0.757, 0.780, 0.753])}
for base, d in T6.items():
    names = [base + v for v in VAR]
    assert np.allclose(np.round(ST.loc[names].values, 1), d["starv"]), base
    for met in ["wBA", "Precision"]:
        assert np.allclose(np.round(CM.loc[names, met].values, 3), d[met]), (base, met)
ref = {"RCA-SMOTE": {"wBA": T2.loc["RCA-SMOTE", "wBA"], "Precision": T2.loc["RCA-SMOTE", "Prec"]},
       "SMOTE": {"wBA": T2.loc["SMOTE", "wBA"], "Precision": T2.loc["SMOTE", "Prec"]}}
fig, axes = plt.subplots(1, 3, figsize=(W, 2.7))
x = np.arange(4)
for j, (ax, (what, ylab, lo, hi)) in enumerate(zip(axes, [("starv", "Runs: smallest sub-group got 0 (%)", 0, 68),
                                                          ("wBA", "wBA (axis starts at 0.78)", 0.78, 0.86),
                                                          ("Precision", "Precision (axis starts at 0.70)", 0.70, 0.83)])):
    for i, (base, key) in enumerate([("KMeans-SMOTE (sv)", "KMeans-SMOTE"), ("DBSMOTE", "DBSMOTE")]):
        names = [base + v for v in VAR]
        vals = ST.loc[names].values if what == "starv" else CM.loc[names, what].values
        xs = x + (i - 0.5) * 0.38
        ax.bar(xs, vals - lo, bottom=lo, width=0.36, color=COL[key], edgecolor=SURF, lw=0.6)
        for xx, vv in zip(xs, vals):
            lbl = (f"{vv:.1f}" if vv < 10 else f"{vv:.0f}") if what == "starv" else f"{vv:.3f}"[1:]
            y0 = vv + (hi - lo) * 0.012
            # white backing only where a reference line runs through the label
            hit = what != "starv" and any(0 <= ref[m][what] - y0 <= (hi - lo) * 0.05 for m in ref)
            ax.text(xx, y0, lbl, ha="center", va="bottom", fontsize=6.2, color=INK2, zorder=4,
                    bbox=dict(fc=SURF, ec="none", pad=0.2) if hit else None)
    if what != "starv":
        for m, ls in [("RCA-SMOTE", "--"), ("SMOTE", ":")]:
            ax.axhline(ref[m][what], color=COL[m], ls=ls, lw=1.1)
    ax.set_ylim(lo, hi)
    ax.set_xticks(x, VLAB, fontsize=7)
    ax.set_xlabel("Plug-in parts added")
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylabel(ylab)
    clean(ax, "y")
    panel(ax, "abc"[j], x=-0.16)
h = [Patch(color=COL["KMeans-SMOTE"], label="KMeans-SMOTE"), Patch(color=COL["DBSMOTE"], label="DBSMOTE"),
     Line2D([], [], color=COL["RCA-SMOTE"], ls="--", lw=1.1, label="RCA-SMOTE (reference, Table 2)"),
     Line2D([], [], color=COL["SMOTE"], ls=":", lw=1.1, label="SMOTE (reference, Table 2)")]
fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(w_pad=2.2)
save(fig, "fig6_transferability")
