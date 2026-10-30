"""Shared runner for the stage-4 experiments (ablation/sensitivity, starvation audit, plug-in fix).

Same protocol as stage 3 on the 24 Group A datasets: 5x2-fold CV stratified on sub-groups, scaling and resampling
on the training fold only, seed = fold number, 4 classifiers (SVM, kNN, DT, RF). Every sampler here is an
oversampler, so starvation measures are recorded for all of them.
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
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

from common import make_classifiers
from exp4_all_methods import evaluate_all
from stage3_datasets import all_datasets, load

warnings.filterwarnings("ignore")


def split_synthetic(Xtr, ytr, Xr):
    """Synthetic rows of the resampled set, and how many original minority points were removed.
    Most methods return [original training set; synthetic points]. A-SUWO first removes some points
    (noise), so its output starts with the kept original rows."""
    n = len(Xtr)
    if len(Xr) >= n and np.array_equal(Xr[:n], Xtr):
        return Xr[n:], 0
    keys = {r.tobytes() for r in Xtr}
    i = 0
    while i < len(Xr) and Xr[i].tobytes() in keys:
        i += 1
    kept = {r.tobytes() for r in Xr[:i]}
    removed_min = sum(1 for r in Xtr[ytr == 1] if r.tobytes() not in kept)
    return Xr[i:], removed_min


def starvation_row(Xtr, ytr, str_, syn, removed_min, info, n_subs, nn5):
    Xmin = Xtr[ytr == 1]
    row = dict(syn_total=len(syn), pct_minority_removed_from_data=100 * removed_min / len(Xmin))
    if len(syn):
        _, i = NearestNeighbors(n_neighbors=1).fit(Xmin).kneighbors(syn)
        lab = str_[ytr == 1][i[:, 0]]
        row["pct_real_minority_used"] = 100 * len(np.unique(i[:, 0])) / len(Xmin)
        row["pct_syn_in_majority_area"] = 100 * np.mean((ytr[nn5.kneighbors(syn)[1]] == 0).sum(1) >= 3)
    else:
        lab = np.array([], dtype=int)
        row["pct_real_minority_used"] = 0.0
        row["pct_syn_in_majority_area"] = np.nan
    for s in range(4):
        row[f"syn_sub{s + 1}"] = int((lab == s).sum()) if s < n_subs else np.nan
        row[f"train_sub{s + 1}"] = int((str_ == s).sum()) if s < n_subs else np.nan
    row["smallest_sub_got_zero"] = bool((lab == n_subs - 1).sum() == 0)
    row["smallest_sub_share_pct"] = 100 * (lab == n_subs - 1).mean() if len(lab) else np.nan
    row["smallest_sub_real_share_pct"] = 100 * (str_ == n_subs - 1).sum() / (ytr == 1).sum()
    if info:
        if "n_noise_dropped" in info:
            row["pct_minority_discarded"] = 100 * info["n_noise_dropped"] / len(Xmin)
        if "n_discarded_by_method" in info:
            row["pct_discarded_by_base_method"] = 100 * info["n_discarded_by_method"] / len(Xmin)
        if "quota" in info:
            q = info["quota"]
            row["n_clusters"] = len(q)
            row["zero_quota_clusters"] = int(sum(v == 0 for v in q))
    return row


def run_task(name, subset, fold, make_samplers, out_dir, classifiers=True):
    t0 = time.time()
    X, y, sub, info = load(name, "A", "PMLB")
    n_subs = len(info["minority_sizes"])
    strat = np.where(y == 0, -1, sub)
    tr, te = list(RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0).split(X, strat))[fold]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    ytr, yte, str_, ste = y[tr], y[te], sub[tr], sub[te]
    nn5 = NearestNeighbors(n_neighbors=5).fit(Xtr)
    base = dict(dataset=name, subset=subset, fold=fold)
    rows, starv = [], []
    for sname, smp in make_samplers(fold, int(ytr.sum())).items():
        t1 = time.time()
        try:
            Xr, yr = smp.fit_resample(Xtr, ytr)
            status = "ok"
        except Exception as e:
            Xr, yr, status = Xtr, ytr, f"failed: {type(e).__name__}: {str(e)[:80]}"
        t_s = time.time() - t1
        syn, removed = split_synthetic(Xtr, ytr, Xr) if status == "ok" else (Xr[:0], 0)
        starv.append(dict(base, method=sname, sampler_status=status, sec_resample=t_s,
                          **starvation_row(Xtr, ytr, str_, syn, removed, getattr(smp, "last_info_", {}), n_subs, nn5)))
        if classifiers:
            for cname, clf in make_classifiers(fold).items():
                clf.fit(Xr, yr)
                rows.append(dict(base, method=sname, classifier=cname, sampler_status=status,
                                 **evaluate_all(clf, Xte, yte, ste, n_subs)))
    if rows:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}__{fold}.csv"), index=False)
    pd.DataFrame(starv).to_csv(os.path.join(out_dir, f"{name}__{fold}_starvation.csv"), index=False)
    print(f"{name} fold {fold}: {time.time() - t0:.0f}s", flush=True)


def run_all(make_samplers, out_dir, jobs=18, classifiers=True, datasets=None):
    os.makedirs(out_dir, exist_ok=True)
    D = all_datasets()
    D = D[D.group == "A"]
    if datasets:
        D = D[D.name.isin(datasets)]
    big = {"sleep", "satimage", "pendigits", "optdigits", "texture", "letter", "shuttle", "page_blocks",
           "wine_quality_white", "mfeat_factors", "gesture_phase", "first_order_theorem_proving",
           "wall_robot_navigation", "steel_plates_fault", "led7"}
    tasks = [(r.name, r.subset, f) for r in D.itertuples() for f in range(10)
             if not os.path.exists(os.path.join(out_dir, f"{r.name}__{f}_starvation.csv"))]
    tasks.sort(key=lambda t: t[0] not in big)
    print(f"{len(tasks)} dataset-fold tasks, {jobs} jobs", flush=True)
    t0 = time.time()
    Parallel(n_jobs=jobs)(delayed(run_task)(n, s, f, make_samplers, out_dir, classifiers) for n, s, f in tasks)
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


def load_results(out_dir):
    import glob
    F = [pd.read_csv(f) for f in glob.glob(os.path.join(out_dir, "*.csv")) if not f.endswith("_starvation.csv")]
    S = [pd.read_csv(f) for f in glob.glob(os.path.join(out_dir, "*_starvation.csv"))]
    return (pd.concat(F, ignore_index=True) if F else None), pd.concat(S, ignore_index=True)
