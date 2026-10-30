"""Stage 3 analysis: statistics, tables and figures for exp4_all_methods.py.

Unit of analysis = one dataset x classifier case, with the metric averaged over the 10 CV folds
(as in stages 1-2). Ensembles are their own classifier, so they are compared with random forest + resampler.

Reporting subsets (fixed in results/supporting/stage3/protocol_stage3.json before running):
  A-main = Group A without the 5 development datasets (stage-2 held-out + new), A-new, A-dev,
  B-all, B-no-overlap (without the 6 KEEL datasets made from the same raw data as Group A datasets).
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import glob
import json
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, friedmanchisquare, studentized_range

OUT = os.environ.get("STAGE3_OUT", "results/supporting/stage3")
FIG = os.environ.get("STAGE3_FIG", "figures")

P = json.load(open("results/supporting/stage3/protocol_stage3.json"))
OURS = "RCA-SMOTE"
SAMPLERS = ["None", "RandomOverSampler", "SMOTE", "ADASYN", "Borderline-SMOTE", "SVM-SMOTE", "KMeans-SMOTE",
            "AW-SMOTE", "RCA-SMOTE", "RandomUnderSampler", "NearMiss-1", "ClusterCentroids", "TomekLinks", "ENN",
            "SMOTEENN", "SMOTETomek"]
ENSEMBLES = ["BalancedRandomForest", "BalancedBagging", "EasyEnsemble", "RUSBoost"]
FAMILY = {m: "baseline" if m == "None" else "over" for m in SAMPLERS[:9]}
FAMILY.update({m: "under" for m in SAMPLERS[9:14]})
FAMILY.update({"SMOTEENN": "hybrid", "SMOTETomek": "hybrid"})
FAMILY.update({m: "ensemble" for m in ENSEMBLES})
CLFS = ["SVM", "kNN", "DT", "RF"]
MET_A = ["wBA", "recall_smallest_sub", "min_sub_recall", "BA", "G_mean", "Recall", "Precision", "Specificity",
         "F1", "MCC", "AUC", "PR_AUC", "Accuracy"]
MET_B = ["BA", "G_mean", "Recall", "Precision", "Specificity", "F1", "MCC", "AUC", "PR_AUC", "Accuracy"]
LABEL = {"wBA": "wBA", "recall_smallest_sub": "Recall, smallest sub-group", "min_sub_recall": "Worst sub-group recall",
         "G_mean": "G-mean", "MCC": "MCC", "F1": "F1", "AUC": "ROC-AUC", "PR_AUC": "PR-AUC", "BA": "BA"}
SUBSETS = {
    "A-main": lambda d: (d.group == "A") & (d.subset != "A-dev"),
    "A-new": lambda d: d.subset == "A-new",
    "A-dev": lambda d: d.subset == "A-dev",
    "B-all": lambda d: d.group == "B",
    "B-no-overlap": lambda d: (d.group == "B") & ~d.dataset.isin(list(P["keel_sharing_raw_data_with_group_A"])),
}
PRIMARY = {"A": ["wBA", "recall_smallest_sub"], "B": ["G_mean", "MCC"]}
log = []


def p(s=""):
    log.append(str(s))
    print(s)


# ------------------------------------------------------------------ load
files = [f for f in glob.glob(f"{OUT}/folds/*.csv") if not f.endswith("_starvation.csv") and "_timing" not in f]
F = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
F["method"] = F["method"].fillna("None")
F.to_csv(f"{OUT}/all_folds.csv.gz", index=False)
S = pd.concat([pd.read_csv(f) for f in glob.glob(f"{OUT}/folds/*_starvation.csv")], ignore_index=True)
S.to_csv(f"{OUT}/starvation_folds.csv.gz", index=False)
n_folds = F.groupby("dataset").fold.nunique()
p(f"datasets: {F.dataset.nunique()} (A: {F[F.group == 'A'].dataset.nunique()}, B: {F[F.group == 'B'].dataset.nunique()}); "
  f"folds per dataset: {n_folds.min()}-{n_folds.max()}; rows: {len(F)}")
assert n_folds.min() == 10, "some datasets are incomplete"

ALLMET = sorted(set(MET_A) | set(MET_B))
M = F.groupby(["dataset", "group", "subset", "classifier", "method"])[ALLMET].mean().reset_index()
M.to_csv(f"{OUT}/means_per_dataset_classifier.csv", index=False)

# ------------------------------------------------------------------ sanity check against stage 2
old = pd.read_csv("results/stage2_heldout_all.csv")
old["method"] = old["method"].fillna("None")
old["method"] = old["method"].replace({m: OURS for m in old.method.unique() if str(m).startswith("FINAL")})
chk = []
for m in [OURS, "SMOTE", "AW-SMOTE"]:
    a = old[old.method == m].groupby(["dataset", "classifier"]).wBA.mean()
    b = M[M.method == m].set_index(["dataset", "classifier"]).wBA.reindex(a.index)
    chk.append((m, float(np.nanmax(np.abs(a - b)))))
p("Reproduction check vs stage-2 held-out results (max |difference| in wBA): " +
  ", ".join(f"{m}: {d:.2e}" for m, d in chk))

# ------------------------------------------------------------------ failures / fallbacks
st = F[F.classifier.isin(["SVM", "(ensemble)"])].groupby(["method", "sampler_status"]).size().reset_index(name="dataset_folds")
st = st[st.sampler_status != "ok"]
st.to_csv(f"{OUT}/sampler_status.csv", index=False)
p("\nSampler failures / fallbacks (dataset-folds, out of %d):" % (F.dataset.nunique() * 10))
p(st.to_string(index=False) if len(st) else "none")


# ------------------------------------------------------------------ statistics helpers
def holm(pv):
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


def pivot(sub, metric, methods, clfs=CLFS):
    d = sub[sub.classifier.isin(clfs) & sub.method.isin(methods)]
    return d.pivot_table(index=["dataset", "classifier"], columns="method", values=metric)[methods].dropna()


def wilcoxon_vs(piv, ours, others):
    rows = []
    for o in others:
        d = piv[ours] - piv[o]
        try:
            pv = wilcoxon(piv[ours], piv[o]).pvalue if (d.abs() > 1e-12).any() else 1.0
        except ValueError:
            pv = np.nan
        rows.append(dict(vs=o, family=FAMILY[o], mean_ours=piv[ours].mean(), mean_other=piv[o].mean(),
                         diff=d.mean(), wins=int((d > 1e-9).sum()), ties=int((d.abs() <= 1e-9).sum()),
                         losses=int((d < -1e-9).sum()), p=pv))
    W = pd.DataFrame(rows)
    W["p_holm"] = holm(W.p.values)
    return W


def cd_value(k, n, alpha=0.05):
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


# ------------------------------------------------------------------ main loop over subsets
summary, ranks_all, fried_all, wil_all, ens_all = [], [], [], [], []
for sname, fn in SUBSETS.items():
    sub = M[fn(M)]
    grp = "A" if sname.startswith("A") else "B"
    mets = MET_A if grp == "A" else MET_B
    nd = sub.dataset.nunique()
    # means and average ranks (blocks = dataset x classifier)
    for met in mets:
        piv = pivot(sub, met, SAMPLERS)
        rk = piv.rank(axis=1, ascending=False).mean()
        for m in SAMPLERS:
            summary.append(dict(subset=sname, metric=met, method=m, family=FAMILY[m], mean=piv[m].mean(),
                                avg_rank=rk[m], n_cases=len(piv)))
        st_, pv = friedmanchisquare(*[piv[m].values for m in SAMPLERS])
        fried_all.append(dict(subset=sname, metric=met, blocks="dataset x classifier", n_blocks=len(piv),
                              k=len(SAMPLERS), chi2=st_, p=pv, CD_nemenyi=cd_value(len(SAMPLERS), len(piv))))
        for c in CLFS:
            pc = pivot(sub, met, SAMPLERS, [c])
            if len(pc) >= 3:
                st_, pv = friedmanchisquare(*[pc[m].values for m in SAMPLERS])
                fried_all.append(dict(subset=sname, metric=met, blocks=f"datasets ({c})", n_blocks=len(pc),
                                      k=len(SAMPLERS), chi2=st_, p=pv, CD_nemenyi=cd_value(len(SAMPLERS), len(pc))))
        W = wilcoxon_vs(piv, OURS, [m for m in SAMPLERS if m != OURS])
        W.insert(0, "metric", met)
        W.insert(0, "classifier", "all 4 pooled")
        W.insert(0, "subset", sname)
        wil_all.append(W)
        if met in PRIMARY[grp]:
            for c in CLFS:
                pc = pivot(sub, met, SAMPLERS, [c])
                Wc = wilcoxon_vs(pc, OURS, [m for m in SAMPLERS if m != OURS])
                Wc.insert(0, "metric", met)
                Wc.insert(0, "classifier", c)
                Wc.insert(0, "subset", sname)
                wil_all.append(Wc)
    # ensembles vs RF + resampler (blocks = datasets)
    rf = sub[(sub.classifier == "RF") & sub.method.isin(["None", "SMOTE", OURS])].copy()
    rf["method"] = "RF + " + rf.method
    ens = pd.concat([rf, sub[sub.method.isin(ENSEMBLES)]])
    for met in mets:
        pv_ = ens.pivot_table(index="dataset", columns="method", values=met)
        for ref in ["RF + RCA-SMOTE", "RF + SMOTE"]:
            rows = []
            for e in ENSEMBLES:
                d = pv_[ref] - pv_[e]
                try:
                    pw = wilcoxon(pv_[ref], pv_[e]).pvalue
                except ValueError:
                    pw = np.nan
                rows.append(dict(subset=sname, metric=met, reference=ref, ensemble=e, mean_reference=pv_[ref].mean(),
                                 mean_ensemble=pv_[e].mean(), diff=d.mean(), reference_wins=int((d > 1e-9).sum()),
                                 ties=int((d.abs() <= 1e-9).sum()), reference_losses=int((d < -1e-9).sum()), p=pw))
            E = pd.DataFrame(rows)
            E["p_holm"] = holm(E.p.values)
            ens_all.append(E)
        for m in pv_.columns:
            ens_all.append(pd.DataFrame([dict(subset=sname, metric=met, reference="(mean only)", ensemble=m,
                                              mean_ensemble=pv_[m].mean())]))

SUM = pd.DataFrame(summary)
FR = pd.DataFrame(fried_all)
WIL = pd.concat(wil_all, ignore_index=True)
ENS = pd.concat(ens_all, ignore_index=True)
SUM.to_csv(f"{OUT}/table_means_and_ranks.csv", index=False)
FR.to_csv(f"{OUT}/friedman.csv", index=False)
WIL.to_csv(f"{OUT}/wilcoxon_holm_rca_vs_each.csv", index=False)
ENS.to_csv(f"{OUT}/ensembles_vs_rf.csv", index=False)

# family tables (wide): one row per method, primary + key metrics, mean (rank)
fam_rows = []
for sname in SUBSETS:
    grp = "A" if sname.startswith("A") else "B"
    keys = (["wBA", "recall_smallest_sub", "G_mean", "F1", "MCC", "PR_AUC"] if grp == "A"
            else ["G_mean", "MCC", "F1", "BA", "AUC", "PR_AUC"])
    s = SUM[(SUM.subset == sname) & SUM.metric.isin(keys)]
    w = s.pivot(index="method", columns="metric", values="mean")[keys]
    r = s.pivot(index="method", columns="metric", values="avg_rank")[keys]
    for m in SAMPLERS:
        fam_rows.append(dict(subset=sname, family=FAMILY[m], method=m,
                             **{f"{k}": round(w.loc[m, k], 3) for k in keys},
                             **{f"rank_{k}": round(r.loc[m, k], 2) for k in keys}))
FAM = pd.DataFrame(fam_rows)
FAM.to_csv(f"{OUT}/table_per_family.csv", index=False)

for sname in SUBSETS:
    grp = "A" if sname.startswith("A") else "B"
    p(f"\n================ {sname} ({M[SUBSETS[sname](M)].dataset.nunique()} datasets) ================")
    t = FAM[FAM.subset == sname].drop(columns="subset").set_index(["family", "method"])
    p(t.to_string())
    for met in PRIMARY[grp]:
        fr = FR[(FR.subset == sname) & (FR.metric == met)]
        p(f"\nFriedman {met}: " + "; ".join(f"{r.blocks}: chi2={r.chi2:.1f}, p={r.p:.2g}, CD={r.CD_nemenyi:.2f}"
                                             for r in fr.itertuples()))
        w = WIL[(WIL.subset == sname) & (WIL.metric == met) & (WIL.classifier == "all 4 pooled")]
        p(f"Wilcoxon RCA-SMOTE vs each ({met}, all classifiers pooled, Holm-corrected):")
        p(w[["vs", "mean_ours", "mean_other", "diff", "wins", "ties", "losses", "p", "p_holm"]]
          .round(4).to_string(index=False))

# ------------------------------------------------------------------ per-sub-group recall (Group A)
PS = []
for sname in ["A-main", "A-new", "A-dev"]:
    sub = F[SUBSETS[sname](F) & F.method.isin(SAMPLERS + ENSEMBLES)]
    d = sub.groupby(["dataset", "classifier", "method"])[[f"recall_sub{i}" for i in range(1, 5)] +
                                                         ["recall_smallest_sub", "Specificity"]].mean()
    t = d.groupby("method").mean()
    t.insert(0, "subset", sname)
    PS.append(t.reset_index())
PS = pd.concat(PS)
PS.to_csv(f"{OUT}/per_subgroup_recall.csv", index=False)
p("\nPer-sub-group recall, A-main (sub1 = largest; mean over dataset x classifier; ensembles: own classifier)")
p(PS[PS.subset == "A-main"].drop(columns="subset").set_index("method").round(3).to_string())

# ------------------------------------------------------------------ starvation (oversamplers, Group A)
S["subset"] = np.where(S.subset == "A-dev", "A-dev", "A-main")
S["smallest_share_pct"] = np.nan
for r in S.index:
    k = int(S.loc[r, ["train_sub1", "train_sub2", "train_sub3", "train_sub4"]].notna().sum())
    if S.loc[r, "syn_total"] > 0:
        S.loc[r, "smallest_share_pct"] = 100 * S.loc[r, f"syn_sub{k}"] / S.loc[r, "syn_total"]
    S.loc[r, "smallest_real_share_pct"] = 100 * S.loc[r, f"train_sub{k}"] / S.loc[r, [f"train_sub{i}" for i in range(1, k + 1)]].sum()
STV = S.groupby(["subset", "method"]).agg(
    runs_smallest_sub_zero_pct=("smallest_sub_got_zero", lambda x: 100 * x.mean()),
    smallest_sub_share_of_synthetic_pct=("smallest_share_pct", "mean"),
    smallest_sub_share_of_real_pct=("smallest_real_share_pct", "mean"),
    synthetic_in_majority_area_pct=("pct_syn_in_majority_area", "mean"),
    real_minority_used_pct=("pct_real_minority_used", "mean"),
    minority_discarded_pct=("pct_minority_discarded", "mean"),
    zero_quota_clusters=("zero_quota_clusters", "mean")).round(1)
STV.to_csv(f"{OUT}/starvation_summary.csv")
p("\nStarvation measures (oversamplers, training folds, Group A)")
p(STV.to_string())

# ------------------------------------------------------------------ run time
RT = F[F.method.isin(SAMPLERS)].groupby(["dataset", "fold", "method"]).sec_resample.first().groupby("method").mean()
RT = pd.DataFrame({"mean_sec_resample_per_fold": RT.round(3)})
RT["mean_sec_ensemble_fit_per_fold"] = np.nan
for e in ENSEMBLES:
    RT.loc[e, "mean_sec_ensemble_fit_per_fold"] = round(F[F.method == e].sec_fit.mean(), 3)
RT.to_csv(f"{OUT}/runtime.csv")
p("\nRun time (seconds per training fold)")
p(RT.to_string())

# ------------------------------------------------------------------ figures
SURF, INK, INK2, MUTED, GRID = "#ffffff", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
FAMCOL = {"baseline": "#b4b2ab", "over": BLUE, "under": ORANGE, "hybrid": AQUA, "ensemble": YELLOW}
plt.rcParams.update({"font.size": 9.5, "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK2,
                     "text.color": INK})


def clean(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)


def cd_diagram(ax, avg, cd, title):
    """Demsar-style critical-difference diagram; bars join methods not significantly different (Nemenyi)."""
    avg = avg.sort_values()
    k = len(avg)
    lo, hi = 1, k
    ax.set_xlim(hi + 3.2, lo - 3.2)
    ax.set_ylim(0, 1)
    ax.axis("off")
    y0 = 0.78
    ax.hlines(y0, lo, hi, color=INK2, lw=1)
    for t in range(lo, hi + 1):
        ax.vlines(t, y0, y0 + 0.025, color=INK2, lw=1)
        ax.text(t, y0 + 0.04, str(t), ha="center", va="bottom", fontsize=8, color=INK2)
    ax.hlines(y0 + 0.13, lo, lo + cd, color=INK, lw=2)
    ax.text(lo + cd / 2, y0 + 0.15, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    names = list(avg.index)
    half = (k + 1) // 2
    step = 0.70 / half
    for i, m in enumerate(names):
        r = avg[m]
        col = BLUE if m == OURS else INK
        fw = "bold" if m == OURS else "normal"
        if i < half:
            yy = y0 - 0.08 - i * step
            ax.plot([r, r, lo - 0.25], [y0, yy, yy], color=MUTED if m != OURS else BLUE, lw=0.8)
            ax.text(lo - 0.3, yy, f"{m} ({r:.1f})", ha="left", va="center", fontsize=8, color=col, fontweight=fw)
        else:
            yy = y0 - 0.08 - (k - 1 - i) * step
            ax.plot([r, r, hi + 0.25], [y0, yy, yy], color=MUTED if m != OURS else BLUE, lw=0.8)
            ax.text(hi + 0.3, yy, f"({r:.1f}) {m}", ha="right", va="center", fontsize=8, color=col, fontweight=fw)
    # cliques: maximal groups whose rank range < cd
    vals = avg.values
    cliques = []
    for i in range(k):
        j = i
        while j + 1 < k and vals[j + 1] - vals[i] < cd:
            j += 1
        if j > i and not any(a <= i and j <= b for a, b in cliques):
            cliques.append((i, j))
    for n, (a, b) in enumerate(cliques):
        yy = y0 - 0.03 - n * 0.022
        ax.hlines(yy, vals[a] - 0.05, vals[b] + 0.05, color=INK, lw=2.2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")


panels = [("A-main", "wBA"), ("A-main", "recall_smallest_sub"), ("B-all", "G_mean"), ("B-all", "MCC")]
fig, axes = plt.subplots(4, 1, figsize=(10, 16))
for ax, (sname, met) in zip(axes, panels):
    s = SUM[(SUM.subset == sname) & (SUM.metric == met)].set_index("method")
    fr = FR[(FR.subset == sname) & (FR.metric == met) & (FR.blocks == "dataset x classifier")].iloc[0]
    cd_diagram(ax, s.avg_rank, fr.CD_nemenyi,
               f"{sname}: {LABEL.get(met, met)}  (average rank over {fr.n_blocks} dataset x classifier cases; "
               f"Friedman p = {fr.p:.1g})")
fig.text(0.01, 0.005, "Lower rank = better. Methods joined by a thick bar are not significantly different "
         "(Nemenyi, alpha = 0.05). RCA-SMOTE in blue.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.01, 1, 1))
fig.savefig(f"{FIG}/stage3_fig1_cd_diagrams.png", dpi=170)
plt.close(fig)

# fig 2: mean of primary metric per method, coloured by family
fig, axes = plt.subplots(1, 4, figsize=(15, 5.6))
for ax, (sname, met) in zip(axes, panels):
    s = SUM[(SUM.subset == sname) & (SUM.metric == met)].set_index("method")["mean"].sort_values()
    lo = max(0.0, np.floor((s.min() - 0.05) * 10) / 10)
    ax.barh(range(len(s)), s.values - lo, left=lo, height=0.68, color=[FAMCOL[FAMILY[m]] for m in s.index],
            edgecolor=SURF, linewidth=1)
    for i, (m, v) in enumerate(s.items()):
        ax.text(v + 0.004, i, f"{v:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(range(len(s)), s.index)
    for t in ax.get_yticklabels():
        if t.get_text() == OURS:
            t.set_color(BLUE)
            t.set_fontweight("bold")
    ax.set_xlim(lo, min(1.0, s.max() + 0.06))
    ax.grid(axis="x", color=GRID)
    ax.set_title(f"{sname}: {LABEL.get(met, met)}", fontsize=10)
    clean(ax)
from matplotlib.patches import Patch
fig.legend([Patch(color=FAMCOL[f]) for f in ["baseline", "over", "under", "hybrid"]],
           ["No resampling", "Oversampling", "Undersampling", "Hybrid"], loc="lower center", ncol=4, frameon=False)
fig.suptitle("Mean over dataset x classifier cases (x-axis does not start at 0)", fontsize=10.5)
fig.tight_layout(rect=(0, 0.05, 1, 0.96))
fig.savefig(f"{FIG}/stage3_fig2_means_by_family.png", dpi=170)
plt.close(fig)

# fig 3: per-sub-group recall (A-main), a few methods
SHOW = [("None", "#b4b2ab"), ("SMOTE", AQUA), ("AW-SMOTE", ORANGE), (OURS, BLUE)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
ps = PS[PS.subset == "A-main"].set_index("method")
for m, col in SHOW:
    v = ps.loc[m, [f"recall_sub{i}" for i in range(1, 5)]].values
    axes[0].plot(range(1, 5), v, marker="o", ms=6, lw=2, color=col, label=m,
                 markeredgecolor=SURF, markeredgewidth=1.5)
    axes[0].text(4.08, v[-1], m, va="center", fontsize=8, color=INK2)
axes[0].set_xticks(range(1, 5), ["1 (largest)", "2", "3", "4 (smallest)"])
axes[0].set_xlim(0.8, 4.9)
axes[0].set_xlabel("Minority sub-group (by size)")
axes[0].set_ylabel("Recall (mean over dataset x classifier)")
n4 = F[SUBSETS["A-main"](F) & F.recall_sub4.notna()].dataset.nunique()
axes[0].set_title(f"A-main: recall per sub-group (sub-group 4 exists on {n4} of "
                  f"{F[SUBSETS['A-main'](F)].dataset.nunique()} datasets)", fontsize=9.5)
axes[0].grid(color=GRID)
axes[0].legend(fontsize=8, frameon=False, loc="lower left")
clean(axes[0])
v = ps.loc[SAMPLERS + ENSEMBLES, "recall_smallest_sub"].sort_values()
axes[1].barh(range(len(v)), v.values, height=0.68, color=[FAMCOL[FAMILY[m]] for m in v.index], edgecolor=SURF)
for i, x in enumerate(v.values):
    axes[1].text(x + 0.005, i, f"{x:.3f}", va="center", fontsize=7.5, color=INK2)
axes[1].set_yticks(range(len(v)), v.index, fontsize=8)
for t in axes[1].get_yticklabels():
    if t.get_text() == OURS:
        t.set_color(BLUE)
        t.set_fontweight("bold")
axes[1].set_title("A-main: recall of the smallest sub-group (all 20 methods)", fontsize=9.5)
axes[1].set_xlim(0, 1.05)
axes[1].grid(axis="x", color=GRID)
clean(axes[1])
fig.legend([Patch(color=FAMCOL[f]) for f in FAMCOL], ["No resampling", "Oversampling", "Undersampling", "Hybrid",
                                                      "Ensemble"], loc="lower right", ncol=5, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(f"{FIG}/stage3_fig3_per_subgroup.png", dpi=170)
plt.close(fig)

# fig 4: starvation measures for oversamplers (A-main)
stv = STV.loc["A-main"]
order = [m for m in SAMPLERS if m in stv.index]
cols = [("runs_smallest_sub_zero_pct", "% runs: smallest sub-group got 0 synthetic points"),
        ("synthetic_in_majority_area_pct", "% synthetic points in majority areas"),
        ("real_minority_used_pct", "% real minority points used")]
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
for ax, (c, t) in zip(axes, cols):
    v = stv.loc[order, c]
    ax.barh(range(len(order)), v.values, height=0.66, color=[BLUE if m == OURS else "#b4b2ab" for m in order])
    for i, x in enumerate(v.values):
        ax.text(x + 0.8, i, f"{x:.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(order)), order)
    ax.set_title(t, fontsize=9.5)
    ax.set_xlim(0, max(10, v.max() * 1.18))
    ax.grid(axis="x", color=GRID)
    ax.invert_yaxis()
    clean(ax)
fig.suptitle("Starvation measures, oversamplers, A-main (training folds)", fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{FIG}/stage3_fig4_starvation.png", dpi=170)
plt.close(fig)

# fig 5: ensembles vs RF + resampler
E = ENS[ENS.reference == "(mean only)"]
show = [("A-main", "wBA"), ("A-main", "recall_smallest_sub"), ("B-all", "G_mean"), ("B-all", "MCC")]
order = ["RF + None", "RF + SMOTE", "RF + RCA-SMOTE"] + ENSEMBLES
fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharey=True)
for ax, (sname, met) in zip(axes, show):
    v = E[(E.subset == sname) & (E.metric == met)].set_index("ensemble").mean_ensemble.reindex(order)
    colors = [BLUE if m == "RF + RCA-SMOTE" else ("#6f6e69" if m == "RF + SMOTE" else
              ("#b4b2ab" if m == "RF + None" else YELLOW)) for m in order]
    ax.barh(range(len(order)), v.values, height=0.66, color=colors)
    for i, x in enumerate(v.values):
        ax.text(x + 0.008, i, f"{x:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlim(0, 1.08)
    ax.set_title(f"{sname}: {LABEL.get(met, met)}", fontsize=9.5)
    ax.grid(axis="x", color=GRID)
    ax.invert_yaxis()
    clean(ax)
fig.suptitle("Ensemble methods vs random forest + resampling (mean over datasets)", fontsize=10.5)
fig.tight_layout()
fig.savefig(f"{FIG}/stage3_fig5_ensembles.png", dpi=170)
plt.close(fig)

open(f"{OUT}/summary.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
print("\nsaved tables to results/supporting/stage3/ and figures to figures/stage3_*.png")
