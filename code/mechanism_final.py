"""Limitation measurements (training folds only) for AW-SMOTE vs the final proposed method, all 15 datasets."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from common import bellinger_binary
from rc_aw_smote import AWSMOTE
from improved import ICOS, VARIANTS
from exp2_benchmarks import DATASETS
rows = []
for name in DATASETS:
    X, y, sub, info = bellinger_binary(name); k = len(info["minority_sizes"])
    strat = np.where(y == 0, -1, sub)
    for f, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0).split(X, strat)):
        Xtr = StandardScaler().fit(X[tr]).transform(X[tr]); ytr, s = y[tr], sub[tr]
        Xmin = Xtr[ytr == 1]
        nn5 = NearestNeighbors(n_neighbors=5).fit(Xtr)
        for m, smp in [("AW-SMOTE", AWSMOTE(random_state=f)),
                       ("Proposed (final)", ICOS(random_state=f, **VARIANTS["V7 V4 with global SMOTE partners"]))]:
            Xr, yr = smp.fit_resample(Xtr, ytr); syn = Xr[len(Xtr):]
            _, i = NearestNeighbors(n_neighbors=1).fit(Xmin).kneighbors(syn)
            sub_of = s[ytr == 1][i[:, 0]]
            q = smp.last_info_.get("quota", [])
            rows.append(dict(dataset=name, fold=f, method=m,
                pct_discarded=100 * smp.last_info_.get("n_noise_dropped", 0) / len(Xmin),
                smallest_zero=(sub_of == k - 1).sum() == 0,
                zero_quota=sum(v == 0 for v in q), n_clusters=len(q),
                pct_used=100 * len(np.unique(i[:, 0])) / len(Xmin),
                pct_in_maj=100 * np.mean((ytr[nn5.kneighbors(syn)[1]] == 0).sum(1) >= 3)))
D = pd.DataFrame(rows); D.to_csv("results/mechanism_final.csv", index=False)
t = D.groupby("method").agg(minority_discarded_pct=("pct_discarded", "mean"),
    runs_smallest_sub_zero_pct=("smallest_zero", lambda x: 100 * x.mean()),
    zero_quota_clusters=("zero_quota", "mean"), clusters=("n_clusters", "mean"),
    real_points_used_pct=("pct_used", "mean"), synthetic_in_majority_area_pct=("pct_in_maj", "mean")).round(1)
print(t.to_string()); t.to_csv("results/table_mechanism_final.csv")
