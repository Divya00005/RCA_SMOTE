"""Stage 4 checks: re-analysis of the saved stage-3 and stage-4 result files only (no new experiments).

1. Ablation and plug-in tables for: A-main (19 datasets, as reported), all 24 Group A datasets, and the 5
   development datasets alone.
2. Plug-in fix, original vs + rescue + quota: Wilcoxon on wBA, smallest-group recall, MCC, F1, specificity,
   precision (Holm over the 6 metrics, per method).
3. KMeans-SMOTE starvation: smote-variants vs imbalanced-learn, split by whether imbalanced-learn ran with its
   defaults or with the stage-1/2 settings (and whether it fell back to plain SMOTE).
4. Rank-change analysis with precision and specificity added.
5. Per-dataset wBA change for the plug-in fix.
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, kendalltau

OUT = "results/supporting/stage4"
DEV = ["ecoli", "led7", "shuttle", "texture", "wine_quality_white"]
SUBSETS = {"A-main (19, as reported)": lambda d: ~d.dataset.isin(DEV),
           "all 24 (with development)": lambda d: d.dataset.notna(),
           "development only (5)": lambda d: d.dataset.isin(DEV)}
log = []


def p(s=""):
    log.append(str(s))
    print(s)


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


def wtest(a, b):
    d = a - b
    try:
        pv = wilcoxon(a, b).pvalue if (d.abs() > 1e-12).any() else 1.0
    except ValueError:
        pv = np.nan
    return dict(mean_after=a.mean(), mean_before=b.mean(), diff=d.mean(), wins=int((d > 1e-9).sum()),
                ties=int((d.abs() <= 1e-9).sum()), losses=int((d < -1e-9).sum()), p=pv)


MET = ["wBA", "recall_smallest_sub", "MCC", "F1", "Specificity", "Precision", "Recall", "G_mean"]


def case_means(path):
    F = pd.read_csv(path)
    F["method"] = F["method"].fillna("None")
    return F.groupby(["dataset", "classifier", "method"])[MET].mean().reset_index()


# ======================================================================= 1. without development datasets
A = case_means(f"{OUT}/ablation_all_folds.csv.gz")
SA = pd.read_csv(f"{OUT}/ablation_starvation_folds.csv.gz")
DEF = "RCA-SMOTE (default)"
ABL = sorted(m for m in A.method.unique() if m.startswith("ablation"))
rows = []
for sname, mask in SUBSETS.items():
    sub = A[mask(A)]
    for met in ["wBA", "recall_smallest_sub", "MCC"]:
        piv = sub.pivot_table(index=["dataset", "classifier"], columns="method", values=met)
        R = pd.DataFrame([dict(subset=sname, metric=met, setting=m, **wtest(piv[m], piv[DEF])) for m in ABL])
        R["p_holm"] = holm(R.p.values)
        rows.append(R)
    st = SA[mask(SA)].groupby("method").smallest_sub_got_zero.mean() * 100
    rows.append(pd.DataFrame([dict(subset=sname, metric="runs_smallest_zero_pct", setting=m, mean_after=st[m],
                                   mean_before=st[DEF], diff=st[m] - st[DEF]) for m in ABL]))
AB = pd.concat(rows, ignore_index=True)
AB.to_csv(f"{OUT}/checks_1_ablation_by_subset.csv", index=False)
p("=== Check 1a: ablation, difference to full RCA-SMOTE (Holm p over the 6 ablations) ===")
for met in ["wBA", "recall_smallest_sub", "MCC", "runs_smallest_zero_pct"]:
    t = AB[AB.metric == met].copy()
    t["cell"] = t.apply(lambda r: f"{r['diff']:+.4f}" + ("" if pd.isna(r.get("p_holm")) else
                                                        f" (p={r['p_holm']:.3f})"), axis=1)
    p(f"-- {met}")
    p(t.pivot(index="setting", columns="subset", values="cell")[list(SUBSETS)].to_string())

C = case_means(f"{OUT}/cluster_all_folds.csv.gz")
SC = pd.read_csv(f"{OUT}/cluster_starvation_folds.csv.gz")
BASES = ["KMeans-SMOTE (sv)", "DBSMOTE"]
rows = []
for sname, mask in SUBSETS.items():
    sub = C[mask(C)]
    st = SC[mask(SC)].groupby("method").smallest_sub_got_zero.mean() * 100
    for base in BASES:
        for v in [base, f"{base} + rescue", f"{base} + quota", f"{base} + rescue + quota"]:
            m = sub[sub.method == v]
            rows.append(dict(subset=sname, base=base, variant=v.replace(base, "original").replace("original + ", "+ "),
                             runs_smallest_zero_pct=st[v], wBA=m.wBA.mean(), recall_smallest_sub=m.recall_smallest_sub.mean(),
                             MCC=m.MCC.mean()))
PL = pd.DataFrame(rows)
PL.to_csv(f"{OUT}/checks_1_plugin_by_subset.csv", index=False)
p("\n=== Check 1b: plug-in fix by subset ===")
for base in BASES:
    t = PL[PL.base == base]
    for met in ["runs_smallest_zero_pct", "wBA", "recall_smallest_sub", "MCC"]:
        p(f"-- {base}: {met}")
        p(t.pivot(index="variant", columns="subset", values=met)[list(SUBSETS)]
          .reindex(["original", "+ rescue", "+ quota", "+ rescue + quota"]).round(4).to_string())

# ======================================================================= 2. significance of the plug-in fix
rows = []
for sname in ["A-main (19, as reported)", "all 24 (with development)"]:
    sub = C[SUBSETS[sname](C)]
    for base in BASES:
        R = []
        for met in ["wBA", "recall_smallest_sub", "MCC", "F1", "Specificity", "Precision"]:
            piv = sub.pivot_table(index=["dataset", "classifier"], columns="method", values=met)
            R.append(dict(subset=sname, base=base, metric=met, **wtest(piv[f"{base} + rescue + quota"], piv[base])))
        R = pd.DataFrame(R)
        R["p_holm"] = holm(R.p.values)
        rows.append(R)
SIG = pd.concat(rows, ignore_index=True)
SIG.to_csv(f"{OUT}/checks_2_plugin_significance.csv", index=False)
p("\n=== Check 2: original vs + rescue + quota (Wilcoxon over dataset x classifier cases; Holm over 6 metrics) ===")
p(SIG[["subset", "base", "metric", "mean_before", "mean_after", "diff", "wins", "ties", "losses", "p", "p_holm"]]
  .round(4).to_string(index=False))

# ======================================================================= 3. KMeans-SMOTE 60% vs 31.6%
S3 = pd.read_csv("results/supporting/stage3/starvation_folds.csv.gz")
k3 = S3[(S3.method == "KMeans-SMOTE")][["dataset", "fold", "subset", "sampler_status", "smallest_sub_got_zero"]]
ksv = SC[SC.method == "KMeans-SMOTE (sv)"][["dataset", "fold", "smallest_sub_got_zero", "pct_minority_discarded"]]
K = k3.merge(ksv, on=["dataset", "fold"], suffixes=("_imblearn", "_sv"))
rows = []
for sname, mask in [("A-main (19)", lambda d: ~d.dataset.isin(DEV)), ("all 24", lambda d: d.dataset.notna())]:
    k = K[mask(K)]
    groups = [("all folds", k), ("imblearn ran with library defaults", k[k.sampler_status == "ok"]),
              ("imblearn used stage-1/2 settings", k[k.sampler_status == "kmeans: stage-1 settings"]),
              ("imblearn fell back to plain SMOTE", k[k.sampler_status == "kmeans: smote fallback"]),
              ("did not fall back to plain SMOTE", k[k.sampler_status != "kmeans: smote fallback"])]
    for g, d in groups:
        rows.append(dict(subset=sname, folds=g, n_folds=len(d),
                         imblearn_starved_pct=100 * d.smallest_sub_got_zero_imblearn.mean() if len(d) else np.nan,
                         sv_starved_pct_same_folds=100 * d.smallest_sub_got_zero_sv.mean() if len(d) else np.nan))
KT = pd.DataFrame(rows)
KT.to_csv(f"{OUT}/checks_3_kmeans_gap.csv", index=False)
p("\n=== Check 3: KMeans-SMOTE starvation, imbalanced-learn (stage 3) vs smote-variants (stage 4), same folds ===")
p(KT.round(1).to_string(index=False))
# smote-variants: filter vs random allocation (from the plug-in variants)
st = SC[~SC.dataset.isin(DEV)].groupby("method").smallest_sub_got_zero.mean() * 100
p(f"smote-variants decomposition (A-main): original {st['KMeans-SMOTE (sv)']:.1f}%; keep its filter but fixed "
  f"never-zero quota {st['KMeans-SMOTE (sv) + quota']:.1f}%; remove its filter but keep its random allocation "
  f"{st['KMeans-SMOTE (sv) + rescue']:.1f}%; both changed {st['KMeans-SMOTE (sv) + rescue + quota']:.1f}%")
zq = SC[(~SC.dataset.isin(DEV)) & (SC.method == "KMeans-SMOTE (sv) + rescue")]
p(f"With the filter removed, the random sparsity allocation still gives 0 points to "
  f"{zq.zero_quota_clusters.mean():.1f} of {zq.n_clusters.mean():.1f} clusters per run on average.")
p(f"Minority points in clusters dropped by the smote-variants filter (A-main): "
  f"{ksv[~ksv.dataset.isin(DEV)].pct_minority_discarded.mean():.1f}%")

# ======================================================================= 4. rank change with precision, specificity
M3 = pd.read_csv("results/supporting/stage3/means_per_dataset_classifier.csv")
M3["method"] = M3["method"].fillna("None")
SAMPLERS = ["None", "RandomOverSampler", "SMOTE", "ADASYN", "Borderline-SMOTE", "SVM-SMOTE", "KMeans-SMOTE",
            "AW-SMOTE", "RCA-SMOTE", "RandomUnderSampler", "NearMiss-1", "ClusterCentroids", "TomekLinks", "ENN",
            "SMOTEENN", "SMOTETomek"]
AM = M3[(M3.group == "A") & (M3.subset != "A-dev") & M3.method.isin(SAMPLERS)]
RM = ["F1", "AUC", "wBA", "recall_smallest_sub", "Recall", "Precision", "Specificity", "MCC"]
avg = {m: AM.pivot_table(index=["dataset", "classifier"], columns="method", values=m)[SAMPLERS]
       .rank(axis=1, ascending=False).mean() for m in RM}
POS = pd.DataFrame({m: avg[m].rank(method="min").astype(int) for m in RM})
MEAN = AM.groupby("method")[RM].mean().loc[SAMPLERS]
POS.to_csv(f"{OUT}/checks_4_positions.csv")
MEAN.to_csv(f"{OUT}/checks_4_means.csv")
p("\n=== Check 4: positions (1 = best, average rank over 76 A-main cases) ===")
p(POS.sort_values("recall_smallest_sub").to_string())
p("\nMeans:")
p(MEAN.loc[POS.sort_values("recall_smallest_sub").index].round(3).to_string())
taus = []
for a, b in [("recall_smallest_sub", "Precision"), ("recall_smallest_sub", "Specificity"),
             ("Recall", "Precision"), ("Recall", "Specificity"), ("recall_smallest_sub", "Recall")]:
    t = kendalltau(avg[a][SAMPLERS], avg[b][SAMPLERS])
    taus.append(dict(ranking_1=a, ranking_2=b, kendall_tau=t.statistic, p=t.pvalue))
TT = pd.DataFrame(taus)
TT.to_csv(f"{OUT}/checks_4_tau.csv", index=False)
p(TT.round(3).to_string(index=False))
# within-case: does a method beat RCA-SMOTE / SMOTE on smallest recall only when its specificity is lower?
wc = []
piv_r = AM.pivot_table(index=["dataset", "classifier"], columns="method", values="recall_smallest_sub")
piv_s = AM.pivot_table(index=["dataset", "classifier"], columns="method", values="Specificity")
piv_p = AM.pivot_table(index=["dataset", "classifier"], columns="method", values="Precision")
for m in ["NearMiss-1", "ClusterCentroids"]:
    for ref in ["RCA-SMOTE", "SMOTE"]:
        better = piv_r[m] > piv_r[ref] + 1e-9
        wc.append(dict(method=m, vs=ref, cases_higher_smallest_recall=int(better.sum()),
                       of_those_lower_specificity=int((better & (piv_s[m] < piv_s[ref] - 1e-9)).sum()),
                       of_those_lower_precision=int((better & (piv_p[m] < piv_p[ref] - 1e-9)).sum()),
                       mean_specificity_drop=float((piv_s[ref] - piv_s[m]).mean()),
                       mean_precision_drop=float((piv_p[ref] - piv_p[m]).mean())))
WC = pd.DataFrame(wc)
WC.to_csv(f"{OUT}/checks_4_within_case.csv", index=False)
p(WC.round(3).to_string(index=False))

# ======================================================================= 5. per-dataset wBA change
rows = []
for base in BASES:
    d = C.pivot_table(index="dataset", columns="method", values="wBA")
    dc = C.pivot_table(index=["dataset", "classifier"], columns="method", values="wBA")
    diff = d[f"{base} + rescue + quota"] - d[base]
    dcc = (dc[f"{base} + rescue + quota"] - dc[base]).unstack()
    for ds, v in diff.items():
        rows.append(dict(base=base, dataset=ds, development=ds in DEV, wBA_before=d.loc[ds, base],
                         wBA_after=d.loc[ds, f"{base} + rescue + quota"], diff=v,
                         classifiers_worse=", ".join(c for c in dcc.columns if dcc.loc[ds, c] < -1e-9)))
PD = pd.DataFrame(rows).sort_values(["base", "diff"])
PD.to_csv(f"{OUT}/checks_5_plugin_per_dataset.csv", index=False)
p("\n=== Check 5: per-dataset wBA (mean over 4 classifiers x 10 folds), original vs + rescue + quota ===")
for base in BASES:
    t = PD[PD.base == base]
    p(f"-- {base}: worse on {int((t['diff'] < -1e-9).sum())} of {len(t)} datasets")
    p(t[t["diff"] < -1e-9][["dataset", "development", "wBA_before", "wBA_after", "diff", "classifiers_worse"]]
      .round(4).to_string(index=False))
    p(f"   largest gains: " + ", ".join(f"{r.dataset} {r.diff:+.3f}" for r in t.sort_values('diff').tail(3).itertuples()))
    dc = C.pivot_table(index=["dataset", "classifier"], columns="method", values="wBA")
    cd = (dc[f"{base} + rescue + quota"] - dc[base]).rename("diff").reset_index()
    cd["development"] = cd.dataset.isin(DEV)
    cd["base"] = base
    worse = cd[cd["diff"] < -1e-9].sort_values("diff")
    worse.to_csv(f"{OUT}/checks_5_plugin_worse_cases_{'kmeans' if 'KMeans' in base else 'dbsmote'}.csv", index=False)
    p(f"   dataset x classifier cases worse: {len(worse)} of {len(cd)} "
      f"(A-main: {int((~worse.development).sum())} of {int((~cd.development).sum())}); worst cases:")
    p(worse.head(8)[["dataset", "classifier", "development", "diff"]].round(4).to_string(index=False))

open(f"{OUT}/checks_summary.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
