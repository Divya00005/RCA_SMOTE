"""Shared helpers: oversamplers, classifiers, metrics, and Bellinger-style datasets."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import warnings
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, matthews_corrcoef, average_precision_score, precision_score
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, KMeansSMOTE
from sklearn.cluster import KMeans

from rc_aw_smote import AWSMOTE, RCAWSMOTE

warnings.filterwarnings("ignore")
DATA_DIR = str(PROJECT_ROOT / "data" / "raw")


# ---------------------------------------------------------------- oversamplers
class NoSampling:
    def fit_resample(self, X, y):
        return X, y


class SafeKMeansSMOTE:
    """imblearn's KMeansSMOTE raises an error when no cluster passes its filter.
    In that case we fall back to SMOTE and count it (reported in the results)."""
    fallbacks = 0

    def __init__(self, random_state=0):
        self.rs = random_state

    def fit_resample(self, X, y):
        n_min = np.bincount(y).min()
        k = max(2, min(10, n_min // 4))
        try:
            return KMeansSMOTE(kmeans_estimator=KMeans(n_clusters=k, n_init=3, random_state=self.rs),
                               cluster_balance_threshold=0.1, k_neighbors=2,
                               random_state=self.rs).fit_resample(X, y)
        except Exception:
            SafeKMeansSMOTE.fallbacks += 1
            return SMOTE(k_neighbors=min(5, n_min - 1), random_state=self.rs).fit_resample(X, y)


def make_samplers(seed):
    return {
        "None": NoSampling(),
        "SMOTE": SMOTE(k_neighbors=5, random_state=seed),
        "Borderline-SMOTE": BorderlineSMOTE(k_neighbors=5, random_state=seed),
        "KMeans-SMOTE": SafeKMeansSMOTE(random_state=seed),
        "AW-SMOTE": AWSMOTE(random_state=seed),
        "RC-AW-SMOTE (ours)": RCAWSMOTE(random_state=seed),
        "Ablation: floor only": RCAWSMOTE(use_rescue=False, random_state=seed),
        "Ablation: rescue only": RCAWSMOTE(use_floor=False, random_state=seed),
    }


def make_classifiers(seed):
    return {
        "SVM": SVC(C=1.0, kernel="rbf", random_state=seed),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "DT": DecisionTreeClassifier(random_state=seed),
        "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1),
    }


# ---------------------------------------------------------------- metrics
def evaluate(clf, X_te, y_te, sub_te):
    """y_te: 0 = majority, 1 = minority. sub_te: minority sub-concept id (0 = largest), -1 for majority."""
    pred = clf.predict(X_te)
    if hasattr(clf, "decision_function"):
        score = clf.decision_function(X_te)
    else:
        score = clf.predict_proba(X_te)[:, 1]
    maj = y_te == 0
    spec = (pred[maj] == 0).mean()
    rec = (pred[~maj] == 1).mean()
    subs = sorted(set(sub_te[~maj]))
    rec_sub = {s: (pred[sub_te == s] == 1).mean() for s in subs}
    out = dict(
        G_mean=np.sqrt(spec * rec),
        F1=f1_score(y_te, pred, zero_division=0),
        AUC=roc_auc_score(y_te, score) if len(np.unique(y_te)) == 2 else np.nan,
        BA=(spec + rec) / 2,
        # weighted BA (Bellinger et al. 2023): every minority sub-concept counts equally
        wBA=(spec + np.mean(list(rec_sub.values()))) / 2,
        Accuracy=(pred == y_te).mean(),
        Precision=precision_score(y_te, pred, zero_division=0),
        Recall=rec,
        Specificity=spec,
        MCC=matthews_corrcoef(y_te, pred),
        PR_AUC=average_precision_score(y_te, score),
        recall_largest_sub=rec_sub[subs[0]],
        recall_smallest_sub=rec_sub[subs[-1]],
        min_sub_recall=min(rec_sub.values()),
    )
    return out


# ---------------------------------------------------------------- datasets
def load_raw(name):
    df = pd.read_csv(f"{DATA_DIR}/{name}.tsv.gz", sep="\t")
    X = df.drop(columns="target").values.astype(float)
    y = df["target"].values
    return X, y


def bellinger_binary(name, max_minority_subs=4, min_sub=10, max_majority=2000, seed=0):
    """Binary dataset with KNOWN minority sub-concepts, following Bellinger et al. (PMLR 241, 2023):
    classes sorted by size, the floor(C/2) largest form the majority class, the remaining classes are
    minority sub-concepts, and each successive minority sub-concept is halved in size.
    Practical limits (ours): classes with < min_sub samples are dropped, at most 4 minority sub-concepts,
    the smallest sub-concept keeps >= min_sub samples so it appears in every test fold, and the
    majority class is subsampled to <= 2000 points to keep run time manageable."""
    rng = np.random.RandomState(seed)
    X, y = load_raw(name)
    cls, cnt = np.unique(y, return_counts=True)
    keep = cls[cnt >= min_sub]
    order = keep[np.argsort([-(y == c).sum() for c in keep])]
    C = len(order)
    maj_cls = order[: C // 2]
    min_cls = order[C // 2:][:max_minority_subs]
    if len(min_cls) < 2:
        return None
    maj_idx = np.where(np.isin(y, maj_cls))[0]
    if len(maj_idx) > max_majority:
        maj_idx = rng.choice(maj_idx, max_majority, replace=False)
    # halving sizes
    sizes = []
    n_prev = None
    for c in min_cls:
        avail = (y == c).sum()
        n = avail if n_prev is None else min(avail, max(n_prev // 2, min_sub))
        # keep overall minority meaningfully smaller than majority
        if n_prev is None:
            n = min(n, max(len(maj_idx) // 10, min_sub * 4))
        sizes.append(n)
        n_prev = n
    idx, ybin, sub = list(maj_idx), [0] * len(maj_idx), [-1] * len(maj_idx)
    for s, (c, n) in enumerate(zip(min_cls, sizes)):
        ci = rng.choice(np.where(y == c)[0], n, replace=False)
        idx += list(ci)
        ybin += [1] * n
        sub += [s] * n
    idx = np.array(idx)
    return X[idx], np.array(ybin), np.array(sub), dict(
        name=name, n_maj=len(maj_idx), minority_sizes=sizes, n_features=X.shape[1],
        IR=round(len(maj_idx) / sum(sizes), 2))
