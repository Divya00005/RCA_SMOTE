"""The 7 main figures of the paper, as one consistent set (PDF + 600-dpi PNG in paper_figures/).

Every number comes from the saved result files (results/supporting/stage3, results/supporting/stage4). The only computation is
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

OUT = "paper_figures"
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

# ================================================================== Figure 1: method diagram
fig, ax = plt.subplots(figsize=(W, 3.9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 66)
ax.axis("off")
cols_x = [12.0, 37.3, 62.6, 87.9]        # column centres
BW, BH = 21.6, 15.0                     # box width and height (axis units)
heads = ["1. Find clusters", "2. Share out new points", "3. Generate points", "Result on 19 test datasets"]
for x, h in zip(cols_x, heads):
    ax.text(x, 63.5, h, ha="center", va="center", fontsize=8, fontweight="bold", color=INK)


def box(x, y, text, method, key=False):
    ax.add_patch(FancyBboxPatch((x - BW / 2, y - BH / 2), BW, BH, boxstyle="round,pad=0.2,rounding_size=1.2",
                                fc=TINT[method], ec=COL[method], lw=1.8 if key else 0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=6.6, color=INK, linespacing=1.4)
    if key:
        ax.text(x + BW / 2 - 1.0, y + BH / 2 + 0.2, "plug-in part", ha="right", va="center", fontsize=6,
                color=SURF, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.6", fc=COL[method], ec="none"))


def arrow(x0, x1, y, method):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8, lw=1.0, color=COL[method]))


lanes = [("AW-SMOTE (base method)", 43.5,
          ["DBSCAN on minority points\n(eps = 65th percentile).\nAll 'noise' points removed:\n"
           "34.9% of the minority\nclass discarded",
           "Quota by boundary weight.\nA cluster with no majority\nneighbour ($W_j$ = 0)\ngets 0 new points",
           "Interpolation between\npoints of the same cluster",
           "Smallest sub-group gets\nno new points in\n34.2% of runs"], [False, False, False, False]),
         ("RCA-SMOTE (correction)", 17.0,
          ["DBSCAN (eps = 90th pct.).\nRescue: drop a point only\nif its 5 nearest neighbours\n"
           "are all majority:\n2.8% discarded",
           "Quota by cluster size:\n" r"$n_j \propto |C_j|^{0.5}$," "\nat least 1 for every cluster",
           "SMOTE-style interpolation\nwith one of the 5 nearest\nminority points",
           "Smallest sub-group gets\nno new points in\n0.5% of runs"], [True, True, False, False])]
for method, y, texts, keys in lanes:
    key = method.split()[0]
    ax.text(cols_x[0] - BW / 2, y + BH / 2 + 4.3, method, ha="left", va="center", fontsize=8,
            fontweight="bold", color=COL[key])
    for i, (x, t, k) in enumerate(zip(cols_x, texts, keys)):
        box(x, y, t, key, key=k)
        if i < 3:
            arrow(x + BW / 2 + 0.5, cols_x[i + 1] - BW / 2 - 0.5, y, key)
ax.text(50, 3.0, "The two plug-in parts (rescue rule, never-zero size quota) also remove starvation from other "
        "cluster-based methods:\nKMeans-SMOTE 60% \u2192 0.5% of runs, DBSMOTE 34% \u2192 0.5% of runs.",
        ha="center", va="center", fontsize=6.8, color=INK2, linespacing=1.45)
save(fig, "fig1_method_diagram")

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
save(fig, "fig2_2d_example")
print("Figure 2 counts:", counts)

# ================================================================== Figure 3: starvation audit
A = pd.read_csv("results/supporting/stage4/audit_starvation.csv", index_col=0)
NAME = {"KMeans-SMOTE (sv)": "KMeans-SMOTE (smote-variants)", "KMeans-SMOTE (imblearn) [stage 3]":
        "KMeans-SMOTE (imbalanced-learn)", "AW-SMOTE [stage 3]": "AW-SMOTE", "SMOTE [stage 3]": "SMOTE",
        "RCA-SMOTE [stage 3]": "RCA-SMOTE"}
A.index = [NAME.get(i, i) for i in A.index]


def mcol(name):
    for key in ["RCA-SMOTE", "AW-SMOTE", "DBSMOTE", "KMeans-SMOTE"]:
        if name.startswith(key):
            return COL[key]
    return COL["SMOTE"] if name == "SMOTE" else GRAY


A = A.sort_values("runs_smallest_zero_pct", ascending=True)
fig, axes = plt.subplots(1, 2, figsize=(W, 3.0), gridspec_kw=dict(width_ratios=[1.05, 1]))
ax = axes[0]
v = A.runs_smallest_zero_pct
ax.barh(range(len(v)), v.values, color=[mcol(m) for m in v.index], height=0.68, edgecolor=SURF, lw=0.6)
for i, x in enumerate(v.values):
    ax.text(x + 0.8, i, f"{x:.1f}", va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(v)), v.index)
ax.set_xlim(0, 68)
ax.set_xlabel("Runs where the smallest sub-group\ngot no new points (%)")
clean(ax, "x")
panel(ax, "a", x=-0.62)
ax = axes[1]
for m, r in A.iterrows():
    ax.scatter(r.runs_smallest_zero_pct, r.recall_smallest_sub, s=30, c=mcol(m), edgecolor=SURF, lw=0.8, zorder=3)
# label positions in data coordinates (x %, y recall), with a thin leader line from the point
LAB = {"MWMOTE": (13, 0.659), "CBSO": (13, 0.650), "SMOTE": (13, 0.641), "RCA-SMOTE": (13, 0.632),
       "Cluster-SMOTE": (13, 0.623), "A-SUWO": (10, 0.579), "CURE-SMOTE": (10, 0.555),
       "AW-SMOTE": (38, 0.566), "KMeans-SMOTE (imbalanced-learn)": (28.5, 0.527), "DBSMOTE": (38, 0.532),
       "SOMO": (24, 0.512), "KMeans-SMOTE (smote-variants)": (57, 0.512)}
for m, r in A.iterrows():
    tx, ty = LAB[m]
    right = tx < r.runs_smallest_zero_pct
    txt = m.replace(" (imbalanced-learn)", "\n(imbalanced-learn)").replace(" (smote-variants)", "\n(smote-variants)")
    ax.annotate(txt, (r.runs_smallest_zero_pct, r.recall_smallest_sub), xytext=(tx, ty), textcoords="data",
                linespacing=1.0,
                fontsize=6.5, color=COL[m] if m in ("RCA-SMOTE", "AW-SMOTE", "SMOTE") else INK2,
                fontweight="bold" if m in ("RCA-SMOTE", "AW-SMOTE", "SMOTE") else "normal",
                ha="right" if right else "left", va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=1, shrinkB=3))
rho = spearmanr(A.runs_smallest_zero_pct, A.recall_smallest_sub)
ax.text(0.97, 0.97, f"Spearman rho = {rho.statistic:.2f}\n(p = {rho.pvalue:.3f}, 12 methods)", transform=ax.transAxes,
        ha="right", va="top", fontsize=7.2, color=INK)
ax.set_xlabel("Runs where the smallest sub-group\ngot no new points (%)")
ax.set_ylabel("Recall of the smallest sub-group")
ax.set_xlim(-3, 66)
clean(ax, "both")
panel(ax, "b", x=-0.12)
fig.tight_layout(w_pad=1.2)
save(fig, "fig3_starvation_audit")

# ================================================================== Figure 4: recall per sub-group
PS = pd.read_csv("results/supporting/stage3/per_subgroup_recall.csv")
PS = fill_none(PS)
PS = PS[PS.subset == "A-main"].set_index("method")
DS = pd.read_csv("results/supporting/stage3/datasets.csv")
DS = DS[(DS.group == "A") & (DS.subset != "A-dev")]
nsub = DS.minority_subgroup_sizes.str.count("/") + 1
n_with = [int((nsub >= k).sum()) for k in range(1, 5)]
fig, ax = plt.subplots(figsize=(4.4, 2.9))
SHOW = ["None", "AW-SMOTE", "SMOTE", "RCA-SMOTE"]
ends = {}
for m in SHOW:
    vals = PS.loc[m, [f"recall_sub{i}" for i in range(1, 5)]].values.astype(float)
    ax.plot(range(1, 5), vals, color=COL[m], lw=1.6, marker="o", ms=4.5, mec=SURF, mew=1.0, zorder=3)
    ends[m] = vals[-1]
order = sorted(ends, key=ends.get)
ys, last = {}, -1
for m in order:                          # keep end labels at least 0.022 apart
    y = max(ends[m], last + 0.022)
    ys[m], last = y, y
for m in SHOW:
    ax.text(4.1, ys[m], m, va="center", fontsize=7.2, color=COL[m] if m != "None" else INK2, fontweight="bold")
ax.set_xticks(range(1, 5), [f"1 (largest)\n{n_with[0]} datasets", f"2\n{n_with[1]}", f"3\n{n_with[2]}",
                            f"4 (smallest)\n{n_with[3]}"])
ax.set_xlim(0.8, 4.85)
ax.set_xlabel("Minority sub-group, from largest to smallest")
ax.set_ylabel("Recall")
clean(ax, "y")
save(fig, "fig4_recall_per_subgroup")

# ================================================================== Figure 5: critical-difference diagrams
SUM = fill_none(pd.read_csv("results/supporting/stage3/table_means_and_ranks.csv"))
FR = pd.read_csv("results/supporting/stage3/friedman.csv")


def cd_diagram(ax, avg, cd):
    avg = avg.sort_values()
    k = len(avg)
    lo, hi = 1, k
    ax.set_xlim(hi + 3.6, lo - 3.6)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y0 = 0.72
    ax.hlines(y0, lo, hi, color=INK2, lw=0.8)
    for t in range(lo, hi + 1):
        ax.vlines(t, y0, y0 + 0.025, color=INK2, lw=0.8)
        ax.text(t, y0 + 0.045, str(t), ha="center", va="bottom", fontsize=7, color=INK2)
    ax.hlines(y0 + 0.16, lo, lo + cd, color=INK, lw=1.6)
    ax.text(lo + cd / 2, y0 + 0.18, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    names = list(avg.index)
    half = (k + 1) // 2
    step = 0.64 / half
    for i, m in enumerate(names):
        r = avg[m]
        c = COL.get(m, INK) if m in ("RCA-SMOTE", "AW-SMOTE", "SMOTE") else INK
        lc = COL.get(m, MUTED) if m in ("RCA-SMOTE", "AW-SMOTE", "SMOTE") else MUTED
        fw = "bold" if m in ("RCA-SMOTE", "AW-SMOTE", "SMOTE") else "normal"
        if i < half:
            yy = y0 - 0.08 - i * step
            ax.plot([r, r, lo - 0.25], [y0, yy, yy], color=lc, lw=0.7)
            ax.text(lo - 0.35, yy, f"{m} ({r:.1f})", ha="left", va="center", fontsize=7, color=c, fontweight=fw)
        else:
            yy = y0 - 0.08 - (k - 1 - i) * step
            ax.plot([r, r, hi + 0.25], [y0, yy, yy], color=lc, lw=0.7)
            ax.text(hi + 0.35, yy, f"({r:.1f}) {m}", ha="right", va="center", fontsize=7, color=c, fontweight=fw)
    vals = avg.values
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n, (a, b) in enumerate(cliques):
        ax.hlines(y0 - 0.03 - n * 0.022, vals[a] - 0.05, vals[b] + 0.05, color=INK, lw=1.8)


fig, axes = plt.subplots(2, 1, figsize=(W, 5.6))
for k, (ax, (sub, met, lab)) in enumerate(zip(axes, [("A-main", "wBA", "Group A, wBA"),
                                                    ("B-all", "G_mean", "Group B, G-mean")])):
    s = SUM[(SUM.subset == sub) & (SUM.metric == met)].set_index("method")
    fr = FR[(FR.subset == sub) & (FR.metric == met) & (FR.blocks == "dataset x classifier")].iloc[0]
    cd_diagram(ax, s.avg_rank, fr.CD_nemenyi)
    ptxt = "p < 0.001" if fr.p < 0.001 else f"p = {fr.p:.3f}"
    ax.text(0.0, 1.0, f"({'ab'[k]})  {lab}", transform=ax.transAxes, fontsize=8.5, fontweight="bold", va="bottom")
    ax.text(0.0, 0.955, f"average rank over {fr.n_blocks} dataset x classifier cases; Friedman {ptxt}",
            transform=ax.transAxes, fontsize=7.2, color=INK2, va="bottom")
fig.tight_layout(h_pad=1.5)
save(fig, "fig5_cd_diagrams")

# ================================================================== Figure 6: plug-in fix
C = pd.read_csv("results/supporting/stage4/cluster_all_folds.csv.gz")
C = C[~C.dataset.isin(DEV)]
CM = C.groupby(["dataset", "classifier", "method"])[["wBA", "Precision"]].mean().groupby("method").mean()
SC = pd.read_csv("results/supporting/stage4/cluster_starvation_folds.csv.gz")
ST = SC[~SC.dataset.isin(DEV)].groupby("method").smallest_sub_got_zero.mean() * 100
M3 = fill_none(pd.read_csv("results/supporting/stage3/means_per_dataset_classifier.csv"))
M3 = M3[(M3.group == "A") & (M3.subset != "A-dev")]
ref = {m: M3[M3.method == m][["wBA", "Precision"]].mean() for m in ["RCA-SMOTE", "SMOTE"]}
VAR = ["", " + rescue", " + quota", " + rescue + quota"]
VLAB = ["none", "rescue", "quota", "both"]
fig, axes = plt.subplots(1, 3, figsize=(W, 2.7))
x = np.arange(4)
for j, (ax, (what, ylab, lo, hi)) in enumerate(zip(axes, [("starv", "Runs: smallest sub-group got 0 (%)", 0, 68),
                                                          ("wBA", "wBA (axis starts at 0.78)", 0.78, 0.86),
                                                          ("Precision", "Precision (axis starts at 0.70)", 0.70, 0.83)])):
    for i, (base, key) in enumerate([("KMeans-SMOTE (sv)", "KMeans-SMOTE"), ("DBSMOTE", "DBSMOTE")]):
        names = [base + v for v in VAR]
        vals = ST.loc[names].values if what == "starv" else CM.loc[names, what].values
        xs = x + (i - 0.5) * 0.38
        ax.bar(xs, vals - lo, bottom=lo, width=0.36, color=COL[key], edgecolor=SURF, lw=0.6,
               label=key if j == 0 else None)
        for xx, vv in zip(xs, vals):
            lbl = (f"{vv:.1f}" if vv < 10 else f"{vv:.0f}") if what == "starv" else f"{vv:.3f}"[1:]
            ax.text(xx, vv + (hi - lo) * 0.012, lbl, ha="center", va="bottom", fontsize=6.2, color=INK2)
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
     Line2D([], [], color=COL["RCA-SMOTE"], ls="--", lw=1.1, label="RCA-SMOTE (reference)"),
     Line2D([], [], color=COL["SMOTE"], ls=":", lw=1.1, label="SMOTE (reference)")]
fig.legend(handles=h, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
fig.tight_layout(w_pad=2.2)
save(fig, "fig6_plugin_fix")

# ================================================================== Figure 7: rank change
POS = pd.read_csv("results/supporting/stage4/checks_4_positions.csv", index_col=0, keep_default_na=False)
TAU16 = pd.read_csv("results/supporting/stage4/rankchange_kendall_tau.csv")
TAU16 = TAU16[TAU16.setting.str.startswith("16")]
TAUc = pd.read_csv("results/supporting/stage4/checks_4_tau.csv")


def tau(a, b):
    t = TAU16[(TAU16.ranking_1 == a) & (TAU16.ranking_2 == b)]
    if len(t):
        return t.kendall_tau.iloc[0]
    return TAUc[(TAUc.ranking_1 == a) & (TAUc.ranking_2 == b)].kendall_tau.iloc[0]


COLS = ["F1", "wBA", "recall_smallest_sub", "Specificity"]
CLAB = ["F1", "wBA", "Recall of the\nsmallest sub-group", "Specificity"]
HL = ["RCA-SMOTE", "NearMiss-1", "ClusterCentroids", "RandomOverSampler"]
fig, ax = plt.subplots(figsize=(W, 4.1))
for m in POS.index:
    yv = POS.loc[m, COLS].values.astype(int)
    hl = m in HL
    ax.plot(range(4), yv, color=COL[m] if hl else GRAY, lw=1.8 if hl else 0.8, marker="o",
            ms=4.5 if hl else 3, mec=SURF, mew=0.8, zorder=3 if hl else 1)
    style = dict(fontsize=7, color=COL[m] if hl else INK2, fontweight="bold" if hl else "normal", va="center")
    ax.text(-0.07, yv[0], f"{m}  {yv[0]}", ha="right", **style)
    ax.text(3.07, yv[-1], f"{yv[-1]}  {m}", ha="left", **style)
for i, (a, b) in enumerate(zip(COLS[:-1], COLS[1:])):
    ax.text(i + 0.5, 17.3, f"tau = {tau(a, b):.2f}", ha="center", va="center", fontsize=7.2, color=INK)
ax.set_xticks(range(4), CLAB)
ax.xaxis.set_ticks_position("top")
ax.tick_params(axis="x", length=0, pad=4)
ax.set_ylim(17.9, 0.4)
ax.set_yticks([])
ax.set_xlim(-1.25, 4.25)
for s in ["top", "right", "left", "bottom"]:
    ax.spines[s].set_visible(False)
ax.text(1.5, 18.6, "Position among 16 resampling methods (1 = best); Kendall's tau between neighbouring rankings",
        ha="center", va="center", fontsize=7, color=INK2)
save(fig, "fig7_rank_change")
