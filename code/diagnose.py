"""Why does plain SMOTE beat AW-SMOTE? Measured on the 5 DEV datasets only (training folds, no classifier)."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from common import bellinger_binary
from rc_aw_smote import AWSMOTE, RCAWSMOTE

DEV = json.load(open("results/protocol_stage2.json"))["dev_datasets"]
rows = []
for name in DEV:
    X, y, sub, info = bellinger_binary(name)
    n_subs = len(info["minority_sizes"])
    strat = np.where(y == 0, -1, sub)
    for fold, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0).split(X, strat)):
        Xtr = StandardScaler().fit(X[tr]).transform(X[tr])
        ytr, str_ = y[tr], sub[tr]
        Xmin = Xtr[ytr == 1]
        nn_min = NearestNeighbors(n_neighbors=2).fit(Xmin)
        med_nn = np.median(nn_min.kneighbors(Xmin)[0][:, 1])
        nn_all = NearestNeighbors(n_neighbors=5).fit(Xtr)
        for mname, smp in [("SMOTE", SMOTE(random_state=fold)), ("AW-SMOTE", AWSMOTE(random_state=fold)),
                           ("RC-AW-SMOTE", RCAWSMOTE(random_state=fold))]:
            Xr, yr = smp.fit_resample(Xtr, ytr)
            syn = Xr[len(Xtr):]
            d_min, i_min = NearestNeighbors(n_neighbors=1).fit(Xmin).kneighbors(syn)
            _, i_all = nn_all.kneighbors(syn)
            maj_frac = (ytr[i_all] == 0).mean(axis=1)
            sub_of = str_[ytr == 1][i_min[:, 0]]
            info_s = getattr(smp, "last_info_", {})
            sizes = info_s.get("cluster_sizes", [])
            # how many distinct real minority points act as "nearest parent" of synthetic points (diversity)
            distinct_parents = len(np.unique(i_min[:, 0])) / len(Xmin)
            rows.append(dict(
                dataset=name, fold=fold, method=mname,
                n_clusters=len(sizes), mean_cluster_size=np.mean(sizes) if sizes else np.nan,
                pct_clusters_size_le3=100 * np.mean(np.array(sizes) <= 3) if sizes else np.nan,
                n_true_subconcepts=n_subs,
                spread=np.mean(d_min[:, 0]) / med_nn,                 # how far synthetic points move from real ones
                pct_real_minority_used=100 * distinct_parents,      # coverage of the minority class
                pct_syn_in_majority_area=100 * np.mean(maj_frac >= 0.6),  # >=3 of 5 nearest real points are majority
                syn_share_smallest_vs_size=(sub_of == n_subs - 1).mean() / ((str_ == n_subs - 1).sum() / (ytr == 1).sum()),
            ))
D = pd.DataFrame(rows)
D.to_csv("results/diagnosis_dev.csv", index=False)
cols = ["n_clusters", "mean_cluster_size", "pct_clusters_size_le3", "spread", "pct_real_minority_used",
        "pct_syn_in_majority_area", "syn_share_smallest_vs_size"]
print(D.groupby("method")[cols].mean().round(2).to_string())
print()
print(D.groupby(["dataset", "method"])[["n_clusters", "n_true_subconcepts", "spread", "pct_syn_in_majority_area"]].mean().round(2).to_string())
