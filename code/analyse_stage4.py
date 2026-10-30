"""Stage 4 analysis: ablation + sensitivity, starvation audit, plug-in fix, rank-change analysis.

Unit = one dataset x classifier case (metric averaged over 10 folds), as in stage 3. Main subset = A-main
(19 Group A datasets without the 5 development datasets); the development datasets are reported separately.
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
from scipy.stats import wilcoxon, kendalltau

from stage4_common import load_results


def holm(pv):
    """Holm-corrected p-values (same as analyse_stage3.holm)."""
    pv = np.asarray(pv, float)
    out = np.full_like(pv, np.nan)
    ok = ~np.isnan(pv)
    idx = np.argsort(pv[ok])
    m = ok.sum()
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * pv[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out

OUT, FIG = "results/supporting/stage4", "figures"
os.makedirs(OUT, exist_ok=True)
log = []


def p(s=""):
    log.append(str(s))
    print(s)


def case_means(F, metrics):
    return F.groupby(["dataset", "subset", "classifier", "method"])[metrics].mean().reset_index()


def compare(M, ref, others, metric, subset_mask):
    d = M[subset_mask(M)].pivot_table(index=["dataset", "classifier"], columns="method", values=metric)
    rows = []
    for o in others:
        a, b = d[o], d[ref]
        diff = a - b
        try:
            pv = wilcoxon(a, b).pvalue if (diff.abs() > 1e-12).any() else 1.0
        except ValueError:
            pv = np.nan
        rows.append(dict(metric=metric, method=o, mean=a.mean(), mean_ref=b.mean(), diff=diff.mean(),
                         better=int((diff > 1e-9).sum()), worse=int((diff < -1e-9).sum()), p=pv))
    R = pd.DataFrame(rows)
    R["p_holm"] = holm(R.p.values)
    return R


def starv_summary(S, mask):
    s = S[mask(S)].copy()
    for c in ["zero_quota_clusters", "pct_minority_discarded", "pct_minority_removed_from_data"]:
        if c not in s:
            s[c] = np.nan
    return s.groupby("method").agg(
        runs_smallest_zero_pct=("smallest_sub_got_zero", lambda x: 100 * x.mean()),
        smallest_share_of_synthetic_pct=("smallest_sub_share_pct", "mean"),
        smallest_share_of_real_pct=("smallest_sub_real_share_pct", "mean"),
        minority_discarded_pct=("pct_minority_discarded", "mean"),
        minority_removed_from_data_pct=("pct_minority_removed_from_data", "mean"),
        real_minority_used_pct=("pct_real_minority_used", "mean"),
        synthetic_in_majority_area_pct=("pct_syn_in_majority_area", "mean"),
        zero_quota_clusters=("zero_quota_clusters", "mean"),
        failed_folds=("sampler_status", lambda x: int((x != "ok").sum())))


MAIN = lambda d: d.subset != "A-dev"
DEV = lambda d: d.subset == "A-dev"
MET = ["wBA", "recall_smallest_sub", "G_mean", "MCC", "F1", "Specificity"]
SURF, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
GRAY = "#b4b2ab"
plt.rcParams.update({"font.size": 9.5, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "text.color": INK})


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


# stage-3 references (same datasets, folds and seeds)
S3 = pd.read_csv("results/supporting/stage3/means_per_dataset_classifier.csv")
S3["method"] = S3["method"].fillna("None")
S3A = S3[S3.group == "A"]
ST3 = pd.read_csv("results/supporting/stage3/starvation_folds.csv.gz")
ref_mean = lambda m, met, mask=MAIN: S3A[mask(S3A) & (S3A.method == m)][met].mean()

# =============================================================== 1. ablation and sensitivity
F, S = load_results("results/supporting/stage4/ablation_folds")
F.to_csv(f"{OUT}/ablation_all_folds.csv.gz", index=False)
S.to_csv(f"{OUT}/ablation_starvation_folds.csv.gz", index=False)
M = case_means(F, MET)
DEF = "RCA-SMOTE (default)"
# reproduction check: default config must equal stage-3 RCA-SMOTE
chk = M[M.method == DEF].set_index(["dataset", "classifier"]).wBA
chk3 = S3A[S3A.method == "RCA-SMOTE"].set_index(["dataset", "classifier"]).wBA.reindex(chk.index)
p(f"Reproduction: default config vs stage-3 RCA-SMOTE, max |diff| wBA = {np.max(np.abs(chk - chk3)):.2e}")
ABL = [m for m in M.method.unique() if m.startswith("ablation")]
SENS = [m for m in M.method.unique() if m not in ABL and m != DEF]
abl_rows = []
for subset_name, mask in [("A-main", MAIN), ("A-dev", DEV)]:
    for met in ["wBA", "recall_smallest_sub", "G_mean", "MCC"]:
        R = compare(M, DEF, ABL, met, mask)
        R.insert(0, "subset", subset_name)
        abl_rows.append(R)
        Rs = compare(M, DEF, SENS, met, mask)
        Rs.insert(0, "subset", subset_name)
        abl_rows.append(Rs.assign(kind="sensitivity"))
AB = pd.concat(abl_rows, ignore_index=True)
AB["kind"] = AB["kind"].fillna("ablation")
AB.to_csv(f"{OUT}/ablation_sensitivity_tests.csv", index=False)
STA = pd.concat([starv_summary(S, MAIN).assign(subset="A-main"), starv_summary(S, DEV).assign(subset="A-dev")])
STA.to_csv(f"{OUT}/ablation_sensitivity_starvation.csv")

p("\n=== Ablation (A-main, 19 datasets x 4 classifiers; vs RCA-SMOTE default; Holm over the 6 ablations) ===")
for met in ["wBA", "recall_smallest_sub", "G_mean", "MCC"]:
    t = AB[(AB.subset == "A-main") & (AB.metric == met) & (AB.kind == "ablation")]
    p(f"-- {met}: default = {t.mean_ref.iloc[0]:.4f}")
    p(t[["method", "mean", "diff", "better", "worse", "p_holm"]].round(4).to_string(index=False))
p("\nStarvation (A-main):")
p(STA[STA.subset == "A-main"].loc[[DEF] + ABL].round(1).to_string())
p(f"References (stage 3, A-main): AW-SMOTE wBA {ref_mean('AW-SMOTE', 'wBA'):.4f}, smallest {ref_mean('AW-SMOTE', 'recall_smallest_sub'):.4f}; "
  f"SMOTE wBA {ref_mean('SMOTE', 'wBA'):.4f}, smallest {ref_mean('SMOTE', 'recall_smallest_sub'):.4f}")

p("\n=== Sensitivity (A-main; Holm over the 17 settings) ===")
for met in ["wBA", "recall_smallest_sub"]:
    t = AB[(AB.subset == "A-main") & (AB.metric == met) & (AB.kind == "sensitivity")]
    p(f"-- {met}")
    p(t[["method", "mean", "diff", "better", "worse", "p_holm"]].round(4).to_string(index=False))

# sensitivity figure: 2 rows (wBA, smallest recall) x 4 parameters
fams = [("eps percentile", "eps percentile", 90, [50, 65, 75, 80, 85, 90, 95, 99]),
        ("beta", "quota exponent beta (n_j ~ |C|^(1-beta))", 0.5, [0, 0.25, 0.5, 0.75, 1.0]),
        ("k partners", "k partners", 5, [3, 5, 7, 10]),
        ("k rescue test", "k for the rescue test", 5, [3, 5, 7, 10])]
mm = M[MAIN(M)].groupby("method")[["wBA", "recall_smallest_sub"]].mean()


def label_of(fam, v):
    if fam == "eps percentile" and v == 65:
        return "ablation: no coarse clusters (eps 65)"
    return f"{fam} {v:g}"


fig, axes = plt.subplots(2, 4, figsize=(15, 6.6), sharey="row")
for c, (fam, xl, dv, vals) in enumerate(fams):
    for r, met in enumerate(["wBA", "recall_smallest_sub"]):
        ax = axes[r, c]
        ys = [mm.loc[DEF if v == dv else label_of(fam, v), met] for v in vals]
        ax.plot(vals, ys, color=BLUE, lw=2, marker="o", ms=6, markeredgecolor=SURF, markeredgewidth=1.5,
                label="RCA-SMOTE with this setting")
        ax.plot([dv], [mm.loc[DEF, met]], marker="o", ms=10, color=BLUE, markeredgecolor=INK, markeredgewidth=1.2,
                linestyle="none", label="default")
        for ref, col in [("SMOTE", AQUA), ("AW-SMOTE", ORANGE)]:
            ax.axhline(ref_mean(ref, met), color=col, lw=1.5, ls="--", label=ref)
        ax.grid(color=GRID)
        clean(ax)
        if r == 1:
            ax.set_xlabel(xl)
        if c == 0:
            ax.set_ylabel("wBA" if met == "wBA" else "Recall, smallest sub-group")
axes[0, 0].legend(fontsize=7.5, frameon=False, loc="lower right")
fig.suptitle("Sensitivity of RCA-SMOTE (A-main, mean over 76 dataset x classifier cases); dashed = stage-3 references",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{FIG}/stage4_fig1_sensitivity.png", dpi=170)
plt.close(fig)

# =============================================================== 2 + 3. starvation audit and plug-in fix
F2, S2 = load_results("results/supporting/stage4/cluster_folds")
F2.to_csv(f"{OUT}/cluster_all_folds.csv.gz", index=False)
S2.to_csv(f"{OUT}/cluster_starvation_folds.csv.gz", index=False)
M2 = case_means(F2, MET)
AUD = ["KMeans-SMOTE (sv)", "DBSMOTE", "A-SUWO", "CBSO", "CURE-SMOTE", "SOMO", "MWMOTE", "Cluster-SMOTE"]
st2 = pd.concat([starv_summary(S2, MAIN).assign(subset="A-main"), starv_summary(S2, DEV).assign(subset="A-dev")])
# stage-3 references
ST3["subset"] = np.where(ST3.subset == "A-dev", "A-dev", "A-main")
ST3["smallest_sub_share_pct"] = np.nan
ST3["smallest_sub_real_share_pct"] = np.nan
for i in ST3.index:
    k = int(ST3.loc[i, ["train_sub1", "train_sub2", "train_sub3", "train_sub4"]].notna().sum())
    if ST3.loc[i, "syn_total"] > 0:
        ST3.loc[i, "smallest_sub_share_pct"] = 100 * ST3.loc[i, f"syn_sub{k}"] / ST3.loc[i, "syn_total"]
    ST3.loc[i, "smallest_sub_real_share_pct"] = 100 * ST3.loc[i, f"train_sub{k}"] / ST3.loc[i, [f"train_sub{j}" for j in range(1, k + 1)]].sum()
ST3["pct_minority_removed_from_data"] = 0.0
ref3 = ST3[ST3.method.isin(["SMOTE", "AW-SMOTE", "RCA-SMOTE", "KMeans-SMOTE"])].copy()
ref3["method"] = ref3.method.map({"SMOTE": "SMOTE [stage 3]", "AW-SMOTE": "AW-SMOTE [stage 3]",
                                  "RCA-SMOTE": "RCA-SMOTE [stage 3]",
                                  "KMeans-SMOTE": "KMeans-SMOTE (imblearn) [stage 3]"})
st_ref = starv_summary(ref3, MAIN).assign(subset="A-main")
AUDT = pd.concat([st2.loc[st2.subset == "A-main"].loc[AUD], st_ref])
perf = M2[MAIN(M2)].groupby("method")[["wBA", "recall_smallest_sub", "MCC"]].mean()
for r in ["SMOTE", "AW-SMOTE", "RCA-SMOTE"]:
    perf.loc[f"{r} [stage 3]"] = [ref_mean(r, m) for m in ["wBA", "recall_smallest_sub", "MCC"]]
perf.loc["KMeans-SMOTE (imblearn) [stage 3]"] = [ref_mean("KMeans-SMOTE", m) for m in ["wBA", "recall_smallest_sub", "MCC"]]
AUDT = AUDT.join(perf)
AUDT.to_csv(f"{OUT}/audit_starvation.csv")
p("\n=== Starvation audit (A-main, 19 datasets x 10 folds; smote-variants defaults) ===")
p(AUDT.round(3).to_string())
p("\nAudit, A-dev:")
p(st2.loc[st2.subset == "A-dev"].loc[AUD].round(1).to_string())

PLUG = []
for base in ["KMeans-SMOTE (sv)", "DBSMOTE"]:
    vs = [f"{base} + rescue", f"{base} + quota", f"{base} + rescue + quota"]
    for met in ["wBA", "recall_smallest_sub", "MCC", "G_mean"]:
        R = compare(M2, base, vs, met, MAIN)
        R.insert(0, "base", base)
        PLUG.append(R)
PLUG = pd.concat(PLUG, ignore_index=True)
PLUG.to_csv(f"{OUT}/plugin_tests.csv", index=False)
p("\n=== Plug-in fix (A-main; vs the original method; Holm over the 3 variants) ===")
for base in ["KMeans-SMOTE (sv)", "DBSMOTE"]:
    names = [base, f"{base} + rescue", f"{base} + quota", f"{base} + rescue + quota"]
    t = st2.loc[st2.subset == "A-main"].loc[names, ["runs_smallest_zero_pct", "minority_discarded_pct",
                                                   "real_minority_used_pct", "synthetic_in_majority_area_pct"]]
    t = t.join(perf.loc[names])
    p(t.round(3).to_string())
    p(PLUG[PLUG.base == base][["metric", "method", "mean", "mean_ref", "diff", "better", "worse", "p_holm"]].round(4).to_string(index=False))

# figure: audit starvation + plug-in
fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
order = AUD + ["KMeans-SMOTE (imblearn) [stage 3]", "AW-SMOTE [stage 3]", "SMOTE [stage 3]", "RCA-SMOTE [stage 3]"]
v = AUDT.loc[order, "runs_smallest_zero_pct"]
cols = [BLUE if "RCA" in m else (GRAY if "[stage 3]" in m else ORANGE) for m in order]
axes[0].barh(range(len(order)), v.values, color=cols, height=0.66)
for i, x in enumerate(v.values):
    axes[0].text(x + 0.8, i, f"{x:.1f}", va="center", fontsize=8, color=INK2)
axes[0].set_yticks(range(len(order)), order, fontsize=8.5)
axes[0].invert_yaxis()
axes[0].set_xlim(0, max(10, v.max() * 1.15))
axes[0].set_title("% runs: smallest sub-group got 0 synthetic points\n(orange = smote-variants defaults; grey = stage-3 references)", fontsize=9)
axes[0].grid(axis="x", color=GRID)
clean(axes[0])
labels = ["original", "+ rescue", "+ quota", "+ rescue + quota"]
x = np.arange(4)
for j, (met, ttl) in enumerate([("runs_smallest_zero_pct", "% runs: smallest sub-group got 0"), ("wBA", "wBA")]):
    ax = axes[j + 1]
    for i, (base, col) in enumerate([("KMeans-SMOTE (sv)", VIOLET), ("DBSMOTE", MAGENTA)]):
        names = [base, f"{base} + rescue", f"{base} + quota", f"{base} + rescue + quota"]
        src = st2.loc[st2.subset == "A-main"] if met != "wBA" else perf
        vals = src.loc[names, met].values
        ax.bar(x + (i - 0.5) * 0.38, vals, width=0.36, color=col, label=base, edgecolor=SURF)
        for xx, vv in zip(x + (i - 0.5) * 0.38, vals):
            ax.text(xx, vv, f"{vv:.1f}" if met != "wBA" else f"{vv:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    if met == "wBA":
        ax.axhline(ref_mean("RCA-SMOTE", "wBA"), color=BLUE, ls="--", lw=1.5, label="RCA-SMOTE (stage 3)")
        ax.axhline(ref_mean("SMOTE", "wBA"), color=AQUA, ls="--", lw=1.5, label="SMOTE (stage 3)")
        lo = np.floor(min(perf.loc[[n for n in perf.index if "KMeans-SMOTE (sv)" in n or "DBSMOTE" in n], "wBA"].min(), 0.8) * 20) / 20
        ax.set_ylim(lo, 0.87)
    ax.set_xticks(x, labels)
    ax.set_title(f"Plug-in fix, A-main: {ttl}" + (" (y-axis does not start at 0)" if met == "wBA" else ""), fontsize=9)
    ax.grid(axis="y", color=GRID)
    ax.legend(fontsize=7.5, frameon=False)
    clean(ax)
fig.tight_layout()
fig.savefig(f"{FIG}/stage4_fig2_audit_plugin.png", dpi=170)
plt.close(fig)

# =============================================================== 4. rank-change analysis (stage-3 data)
SAMPLERS = ["None", "RandomOverSampler", "SMOTE", "ADASYN", "Borderline-SMOTE", "SVM-SMOTE", "KMeans-SMOTE",
            "AW-SMOTE", "RCA-SMOTE", "RandomUnderSampler", "NearMiss-1", "ClusterCentroids", "TomekLinks", "ENN",
            "SMOTEENN", "SMOTETomek"]
ENS = ["BalancedRandomForest", "BalancedBagging", "EasyEnsemble", "RUSBoost"]
RMET = ["F1", "AUC", "wBA", "recall_smallest_sub"]
A = S3A[MAIN(S3A)]


def avg_ranks(df, metric, methods, block):
    piv = df.pivot_table(index=block, columns="method", values=metric)[methods].dropna()
    return piv.rank(axis=1, ascending=False).mean()


def tau_table(df, methods, block, nboot=2000, seed=0):
    rng = np.random.RandomState(seed)
    R = {m: avg_ranks(df, m, methods, block) for m in RMET}
    dsets = df.dataset.unique()
    rows = []
    pairs = [("F1", "wBA"), ("F1", "recall_smallest_sub"), ("AUC", "wBA"), ("AUC", "recall_smallest_sub"),
             ("F1", "AUC"), ("wBA", "recall_smallest_sub")]
    boots = {pr: [] for pr in pairs}
    for _ in range(nboot):
        pick = rng.choice(dsets, len(dsets), replace=True)
        bd = pd.concat([df[df.dataset == d].assign(dataset=f"{d}#{i}") for i, d in enumerate(pick)])
        Rb = {m: avg_ranks(bd, m, methods, block) for m in RMET}
        for a, b in pairs:
            boots[(a, b)].append(kendalltau(Rb[a][methods], Rb[b][methods]).statistic)
    for a, b in pairs:
        t = kendalltau(R[a][methods], R[b][methods])
        lo, hi = np.percentile(boots[(a, b)], [2.5, 97.5])
        rows.append(dict(ranking_1=a, ranking_2=b, kendall_tau=t.statistic, p=t.pvalue, ci95_low=lo, ci95_high=hi))
    pos = pd.DataFrame({m: R[m][methods].rank(method="min").astype(int) for m in RMET})
    return pd.DataFrame(rows), pos, pd.DataFrame(R)


p("\n=== Rank-change analysis (stage-3 results, A-main) ===")
T16, POS16, AR16 = tau_table(A, SAMPLERS, ["dataset", "classifier"])
T16.insert(0, "setting", "16 resampling methods x 4 classifiers (76 cases)")
# per-case tau between F1 and wBA scores
pc = []
for (d, c), g in A[A.method.isin(SAMPLERS)].groupby(["dataset", "classifier"]):
    g = g.set_index("method").loc[SAMPLERS]
    for a, b in [("F1", "wBA"), ("F1", "recall_smallest_sub"), ("AUC", "wBA"), ("AUC", "recall_smallest_sub")]:
        pc.append(dict(dataset=d, classifier=c, pair=f"{a} vs {b}", tau=kendalltau(g[a], g[b]).statistic))
PC = pd.DataFrame(pc)
PC.to_csv(f"{OUT}/rankchange_per_case_tau.csv", index=False)
# 20 methods with random forest (ensembles are random-forest-like classifiers), blocks = datasets
A20 = A[((A.classifier == "RF") & A.method.isin(SAMPLERS)) | A.method.isin(ENS)].copy()
A20["method"] = np.where(A20.method.isin(ENS), A20.method, "RF + " + A20.method)
M20 = ["RF + " + m for m in SAMPLERS] + ENS
T20, POS20, AR20 = tau_table(A20, M20, "dataset")
T20.insert(0, "setting", "20 methods with random forest (19 datasets)")
TT = pd.concat([T16, T20])
TT.to_csv(f"{OUT}/rankchange_kendall_tau.csv", index=False)
p(TT.round(3).to_string(index=False))
p("\nPer-case Kendall tau (16 methods within one dataset x classifier case): median and % of cases with tau < 0")
p(PC.groupby("pair").tau.agg(median="median", pct_negative=lambda x: 100 * (x < 0).mean()).round(2).to_string())


def moves(POS):
    D = POS.copy()
    D["F1 -> wBA"] = D["F1"] - D["wBA"]
    D["F1 -> smallest"] = D["F1"] - D["recall_smallest_sub"]
    D["AUC -> wBA"] = D["AUC"] - D["wBA"]
    D["AUC -> smallest"] = D["AUC"] - D["recall_smallest_sub"]
    D["max |move|"] = D[["F1 -> wBA", "F1 -> smallest", "AUC -> wBA", "AUC -> smallest"]].abs().max(axis=1)
    return D.sort_values("max |move|", ascending=False)


MV16, MV20 = moves(POS16), moves(POS20)
MV16.to_csv(f"{OUT}/rankchange_positions_16.csv")
MV20.to_csv(f"{OUT}/rankchange_positions_20_rf.csv")
p("\nPositions (1 = best) and moves (positive = moves UP when switching to the sub-group metric), 16 methods:")
p(MV16.to_string())
p("\n20 methods with random forest:")
p(MV20.to_string())

# figure: bump charts F1 -> wBA -> smallest, AUC -> wBA -> smallest (16 methods)
fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
top_movers = list(MV16.index[:4])
HL = {"RCA-SMOTE": BLUE}
for m, col in zip([t for t in top_movers if t != "RCA-SMOTE"][:3], [ORANGE, AQUA, VIOLET]):
    HL[m] = col
for ax, first in zip(axes, ["F1", "AUC"]):
    cols_ = [first, "wBA", "recall_smallest_sub"]
    for m in SAMPLERS:
        ys = POS16.loc[m, cols_].values
        hl = m in HL
        ax.plot(range(3), ys, color=HL.get(m, GRAY), lw=2.2 if hl else 1.0, marker="o", ms=6 if hl else 4,
                markeredgecolor=SURF, markeredgewidth=1.2, zorder=3 if hl else 1)
        ax.text(-0.08, ys[0], f"{m}  {ys[0]}", ha="right", va="center", fontsize=7.5,
                color=HL.get(m, INK2), fontweight="bold" if hl else "normal")
        ax.text(2.08, ys[2], f"{ys[2]}  {m}", ha="left", va="center", fontsize=7.5,
                color=HL.get(m, INK2), fontweight="bold" if hl else "normal")
    ax.set_xticks(range(3), [f"rank by {first}", "rank by wBA", "rank by smallest-\nsub-group recall"])
    ax.set_ylim(16.6, 0.4)
    ax.set_yticks([])
    ax.set_xlim(-1.1, 3.1)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    t1 = TT[(TT.setting.str.startswith("16")) & (TT.ranking_1 == first) & (TT.ranking_2 == "wBA")].iloc[0]
    t2 = TT[(TT.setting.str.startswith("16")) & (TT.ranking_1 == first) & (TT.ranking_2 == "recall_smallest_sub")].iloc[0]
    ax.set_title(f"{first} vs wBA: tau = {t1.kendall_tau:.2f};  {first} vs smallest: tau = {t2.kendall_tau:.2f}", fontsize=9.5)
fig.suptitle("Ranking of the 16 resampling methods under different metrics (A-main, average rank over 76 cases; 1 = best)",
             fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{FIG}/stage4_fig3_rank_change.png", dpi=170)
plt.close(fig)

open(f"{OUT}/summary.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
print("\nsaved to results/supporting/stage4/ and figures/stage4_*.png")
