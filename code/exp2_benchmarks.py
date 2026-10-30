"""Experiment 2: 15 real benchmark datasets with known minority sub-concepts (Bellinger et al. protocol).
5x2-fold cross-validation stratified on sub-concept labels; scaling and oversampling are fitted on the
training fold only. Outputs per-fold metrics and per-sub-concept synthetic-sample counts."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from common import bellinger_binary, make_samplers, make_classifiers, evaluate, SafeKMeansSMOTE

DATASETS = ["ecoli", "yeast", "vowel", "segmentation", "satimage", "pendigits", "dermatology",
            "page_blocks", "optdigits", "texture", "letter", "wine_quality_white", "new_thyroid",
            "led7", "shuttle"]


def synth_share(X_tr, y_tr, s_tr, X_res, n_subs):
    syn = X_res[len(X_tr):]
    out = np.zeros(n_subs, dtype=int)
    if len(syn):
        _, i = NearestNeighbors(n_neighbors=1).fit(X_tr[y_tr == 1]).kneighbors(syn)
        lab = s_tr[y_tr == 1][i[:, 0]]
        for s in range(n_subs):
            out[s] = (lab == s).sum()
    return out


def run_dataset(name):
    t0 = time.time()
    X, y, sub, info = bellinger_binary(name)
    n_subs = len(info["minority_sizes"])
    strat = np.where(y == 0, -1, sub)
    cv = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0)
    rows, counts = [], []
    for fold, (tr, te) in enumerate(cv.split(X, strat)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for sname, smp in make_samplers(fold).items():
            fb0 = SafeKMeansSMOTE.fallbacks
            Xr, yr = smp.fit_resample(Xtr, y[tr])
            fell_back = SafeKMeansSMOTE.fallbacks > fb0
            if sname != "None":
                share = synth_share(Xtr, y[tr], sub[tr], Xr, n_subs)
                info_s = getattr(smp, "last_info_", {})
                n_min_tr = (y[tr] == 1).sum()
                counts.append(dict(dataset=name, fold=fold, method=sname,
                                   syn_total=int(len(Xr) - len(Xtr)),
                                   syn_largest_sub=int(share[0]), syn_smallest_sub=int(share[-1]),
                                   smallest_sub_train=int((sub[tr] == n_subs - 1).sum()),
                                   frac_minority_discarded=info_s.get("n_noise_dropped", 0) / n_min_tr,
                                   n_zero_quota_clusters=int(sum(q == 0 for q in info_s.get("quota", []))),
                                   n_clusters=len(info_s.get("quota", [])),
                                   kmeans_fallback_to_smote=bool(fell_back)))
            for cname, clf in make_classifiers(fold).items():
                clf.fit(Xr, yr)
                rows.append(dict(dataset=name, fold=fold, method=sname, classifier=cname,
                                 **evaluate(clf, Xte, y[te], sub[te])))
    pd.DataFrame(rows).to_csv(f"results/per_dataset/{name}_folds.csv", index=False)
    pd.DataFrame(counts).to_csv(f"results/per_dataset/{name}_counts.csv", index=False)
    print(f"{name}: done in {time.time() - t0:.0f}s", flush=True)
    return rows, counts


if __name__ == "__main__":
    import os
    os.makedirs("results/per_dataset", exist_ok=True)
    names = sys.argv[1:] or DATASETS
    out = Parallel(n_jobs=2, verbose=0)(delayed(run_dataset)(n) for n in names)
    rows = [r for o in out for r in o[0]]
    counts = [c for o in out for c in o[1]]
    pd.DataFrame(rows).to_csv("results/exp2_benchmark_folds.csv", index=False)
    pd.DataFrame(counts).to_csv("results/exp2_synthetic_counts.csv", index=False)
    print("saved")
