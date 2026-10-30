"""Stage 3: RCA-SMOTE (frozen) against 19 other imbalance methods on Group A + Group B datasets.

Protocol (same as stages 1-2): 5x2-fold CV (stratified on sub-groups for Group A, on the class for Group B),
scaling and resampling fitted on the training fold only, seed of every method = fold number.
- 16 resampling settings (no resampling + 8 over + 5 under + 2 hybrid) x 4 classifiers (SVM, kNN, DT, RF).
- 4 ensemble classifiers (BRF, BalancedBagging, EasyEnsemble, RUSBoost), trained on the scaled training fold.
- Group A only: starvation measures for the 8 oversamplers (training folds only).
Settings: library defaults, every neighbour count = 5 (k = min(5, minority training size - 1) when the
minority class is smaller). If a method raises an error, the classifier is trained on the original data and
the failure is recorded in `sampler_status`.

  python exp4_all_methods.py                 # everything (resumes: finished dataset-folds are skipped)
  python exp4_all_methods.py --timing        # fold 0 of every available dataset, timings only, no metrics
  options: --jobs N   --datasets a,b,c   --out DIR
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

import sys
import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import (RandomOverSampler, SMOTE, ADASYN, BorderlineSMOTE, SVMSMOTE, KMeansSMOTE)
from imblearn.under_sampling import (RandomUnderSampler, NearMiss, ClusterCentroids, TomekLinks,
                                     EditedNearestNeighbours)
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import (BalancedRandomForestClassifier, BalancedBaggingClassifier,
                               EasyEnsembleClassifier, RUSBoostClassifier)

from common import make_classifiers, evaluate, NoSampling, SafeKMeansSMOTE
from rc_aw_smote import AWSMOTE
from rca_smote import RCASMOTE
from stage3_datasets import all_datasets, load, keel_path

warnings.filterwarnings("ignore")

FAMILY = {
    "None": "baseline",
    "RandomOverSampler": "over", "SMOTE": "over", "ADASYN": "over", "Borderline-SMOTE": "over",
    "SVM-SMOTE": "over", "KMeans-SMOTE": "over", "AW-SMOTE": "over", "RCA-SMOTE": "over",
    "RandomUnderSampler": "under", "NearMiss-1": "under", "ClusterCentroids": "under",
    "TomekLinks": "under", "ENN": "under",
    "SMOTEENN": "hybrid", "SMOTETomek": "hybrid",
    "BalancedRandomForest": "ensemble", "BalancedBagging": "ensemble", "EasyEnsemble": "ensemble",
    "RUSBoost": "ensemble",
}
OVERSAMPLERS = [m for m, f in FAMILY.items() if f == "over"]


class KMeansSMOTEDefault:
    """imblearn KMeansSMOTE with library defaults and k_neighbors = k. It raises an error when no k-means
    cluster has enough minority points; then we use the stage-1/2 settings (SafeKMeansSMOTE, which itself
    falls back to SMOTE). The path taken is recorded in `status_`."""

    def __init__(self, k, random_state):
        self.k, self.rs = k, random_state
        self.status_ = ""

    def fit_resample(self, X, y):
        try:
            out = KMeansSMOTE(k_neighbors=self.k, random_state=self.rs).fit_resample(X, y)
            self.status_ = "ok"
            return out
        except Exception:
            fb = SafeKMeansSMOTE.fallbacks
            out = SafeKMeansSMOTE(random_state=self.rs).fit_resample(X, y)
            self.status_ = "kmeans: smote fallback" if SafeKMeansSMOTE.fallbacks > fb else "kmeans: stage-1 settings"
            return out


def make_samplers(seed, n_min_train):
    k = int(max(1, min(5, n_min_train - 1)))
    return {
        "None": NoSampling(),
        "RandomOverSampler": RandomOverSampler(random_state=seed),
        "SMOTE": SMOTE(k_neighbors=k, random_state=seed),
        "ADASYN": ADASYN(n_neighbors=k, random_state=seed),
        "Borderline-SMOTE": BorderlineSMOTE(k_neighbors=k, random_state=seed),
        "SVM-SMOTE": SVMSMOTE(k_neighbors=k, random_state=seed),
        "KMeans-SMOTE": KMeansSMOTEDefault(k, seed),
        "AW-SMOTE": AWSMOTE(random_state=seed),
        "RCA-SMOTE": RCASMOTE(random_state=seed),
        "RandomUnderSampler": RandomUnderSampler(random_state=seed),
        "NearMiss-1": NearMiss(version=1, n_neighbors=min(5, n_min_train)),
        "ClusterCentroids": ClusterCentroids(random_state=seed),
        "TomekLinks": TomekLinks(),
        "ENN": EditedNearestNeighbours(n_neighbors=5),
        "SMOTEENN": SMOTEENN(smote=SMOTE(k_neighbors=k, random_state=seed),
                             enn=EditedNearestNeighbours(n_neighbors=5, sampling_strategy="all"),
                             random_state=seed),
        "SMOTETomek": SMOTETomek(smote=SMOTE(k_neighbors=k, random_state=seed), random_state=seed),
    }


def make_ensembles(seed):
    return {
        "BalancedRandomForest": BalancedRandomForestClassifier(random_state=seed, n_jobs=1),
        "BalancedBagging": BalancedBaggingClassifier(random_state=seed, n_jobs=1),
        "EasyEnsemble": EasyEnsembleClassifier(random_state=seed, n_jobs=1),
        "RUSBoost": RUSBoostClassifier(random_state=seed),
    }


def evaluate_all(clf, Xte, yte, ste, n_subs):
    out = evaluate(clf, Xte, yte, ste)
    if n_subs > 1:
        pred = clf.predict(Xte)
        for s in range(4):
            out[f"recall_sub{s + 1}"] = (pred[ste == s] == 1).mean() if s < n_subs else np.nan
    return out


def starvation(Xtr, ytr, str_, Xr, smp, n_subs, nn5):
    """Training-fold measures for oversamplers (synthetic points are appended after the real ones)."""
    syn = Xr[len(Xtr):]
    Xmin = Xtr[ytr == 1]
    row = dict(syn_total=len(syn))
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
    info = getattr(smp, "last_info_", None)
    if info:
        row["pct_minority_discarded"] = 100 * info.get("n_noise_dropped", 0) / len(Xmin)
        q = info.get("quota", [])
        row["n_clusters"] = len(q)
        row["zero_quota_clusters"] = int(sum(v == 0 for v in q))
    return row


def run_task(name, group, source, subset, fold, out_dir, timing_only=False):
    t_task = time.time()
    X, y, sub, info = load(name, group, source)
    n_subs = len(info["minority_sizes"])
    strat = np.where(y == 0, -1, sub)
    splits = list(RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0).split(X, strat))
    tr, te = splits[fold]
    sc = StandardScaler().fit(X[tr])
    Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
    ytr, yte, str_, ste = y[tr], y[te], sub[tr], sub[te]
    n_min = int(ytr.sum())
    nn5 = NearestNeighbors(n_neighbors=5).fit(Xtr) if group == "A" else None
    base = dict(dataset=name, group=group, subset=subset, fold=fold)
    rows, starv, times = [], [], []

    for sname, smp in make_samplers(fold, n_min).items():
        t0 = time.time()
        try:
            Xr, yr = smp.fit_resample(Xtr, ytr)
            status = getattr(smp, "status_", "") or "ok"
        except Exception as e:
            Xr, yr, status = Xtr, ytr, f"failed: {type(e).__name__}: {str(e)[:80]}"
        t_s = time.time() - t0
        if group == "A" and sname in OVERSAMPLERS and not timing_only:
            starv.append(dict(base, method=sname, sampler_status=status,
                              **starvation(Xtr, ytr, str_, Xr, smp, n_subs, nn5)))
        for cname, clf in make_classifiers(fold).items():
            t1 = time.time()
            clf.fit(Xr, yr)
            t_c = time.time() - t1
            times.append(dict(base, method=sname, classifier=cname, sec_resample=t_s, sec_classifier=t_c))
            if not timing_only:
                rows.append(dict(base, method=sname, family=FAMILY[sname], classifier=cname,
                                 sampler_status=status, n_train_after=len(yr),
                                 n_min_after=int(yr.sum()), sec_resample=t_s, sec_fit=t_c,
                                 **evaluate_all(clf, Xte, yte, ste, n_subs)))

    for ename in make_ensembles(fold):
        # RUSBoost (library default: decision stumps) raises "worse than random" when its first stump,
        # trained on a random undersample, has error >= 0.5 on the training fold. We then retry with a new
        # seed (up to 10 tries); if every try fails the fold is recorded as failed with empty metrics.
        t1 = time.time()
        clf, status = None, ""
        for attempt in range(10):
            seed = fold + 1000 * attempt
            try:
                clf = make_ensembles(seed)[ename].fit(Xtr, ytr)
                status = "ok" if attempt == 0 else f"ok after {attempt} reseed(s)"
                break
            except Exception as e:
                clf, status = None, f"failed: {type(e).__name__}: {str(e)[:60]}"
        t_c = time.time() - t1
        times.append(dict(base, method=ename, classifier="(ensemble)", sec_resample=0.0, sec_classifier=t_c))
        if not timing_only:
            res = evaluate_all(clf, Xte, yte, ste, n_subs) if clf is not None else {}
            rows.append(dict(base, method=ename, family="ensemble", classifier="(ensemble)",
                             sampler_status=status, n_train_after=len(ytr), n_min_after=n_min,
                             sec_resample=0.0, sec_fit=t_c, **res))

    if timing_only:
        T = pd.DataFrame(times)
        T["sec_task"] = time.time() - t_task
        T.to_csv(os.path.join(out_dir, f"{name}__{fold}_timing.csv"), index=False)
    else:
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{name}__{fold}.csv"), index=False)
        if starv:
            pd.DataFrame(starv).to_csv(os.path.join(out_dir, f"{name}__{fold}_starvation.csv"), index=False)
    dt = time.time() - t_task
    print(f"{name} fold {fold}: {dt:.0f}s", flush=True)
    return name, fold, dt


def available(D):
    ok = []
    for r in D.itertuples():
        if r.source == "KEEL" and not os.path.exists(keel_path(r.name)):
            continue
        ok.append(r)
    return ok


def main(argv):
    if '--help' in argv or '-h' in argv:
        print('Historical protocol runner: --jobs N --datasets a,b --out work/benchmark_folds --timing. Does not reproduce the updated aggregate means without their producing run.'); return

    jobs = int(argv[argv.index("--jobs") + 1]) if "--jobs" in argv else 18
    timing = "--timing" in argv
    out_dir = argv[argv.index("--out") + 1] if "--out" in argv else os.path.join("work", "benchmark_folds")
    os.makedirs(out_dir, exist_ok=True)
    D = all_datasets()
    if "--datasets" in argv:
        D = D[D.name.isin(argv[argv.index("--datasets") + 1].split(","))]
    rows = available(D)
    missing = sorted(set(D.name) - {r.name for r in rows})
    if missing:
        print("not available (skipped):", ", ".join(missing))
    folds = [0] if timing else range(10)
    tasks = [(r.name, r.group, r.source, r.subset, f) for r in rows for f in folds
             if timing or not os.path.exists(os.path.join(out_dir, f"{r.name}__{f}.csv"))]
    # big datasets first, for better load balance
    size = {"sleep": 9, "satimage": 8, "pendigits": 8, "optdigits": 8, "texture": 8, "letter": 8,
            "shuttle": 8, "page_blocks": 8, "wine_quality_white": 8, "mfeat_factors": 7, "scene": 7,
            "ozone_level": 7, "yeast_ml8": 7, "us_crime": 6, "gesture_phase": 8,
            "first_order_theorem_proving": 8, "wall_robot_navigation": 8, "steel_plates_fault": 7, "abalone19": 6}
    tasks.sort(key=lambda t: -size.get(t[0], 0))
    print(f"{len(tasks)} dataset-fold tasks, {jobs} parallel jobs", flush=True)
    t0 = time.time()
    Parallel(n_jobs=jobs, verbose=0)(delayed(run_task)(*t, out_dir, timing) for t in tasks)
    print(f"all done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
