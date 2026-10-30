"""Experiment 1: controlled 2D demonstration of the AW-SMOTE limitation.

Minority class = three sub-concepts:
  S1  large, overlaps the majority class (boundary sub-concept)       60 points
  S2  rare and compact, sits in a low-density part of majority space    8 points
  S3  rare and sparse, also in a low-density region                     8 points
Test sets are drawn fresh from the same generators (large, so per-sub-concept recall is precise).
Repeated over 20 random seeds.
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from common import make_classifiers, evaluate, NoSampling
from imblearn.over_sampling import SMOTE
from rc_aw_smote import AWSMOTE, RCAWSMOTE

SPEC = {  # name: (n_train, n_test, mean, std)
    "maj": (600, 3000, (0.0, 0.0), 2.2),
    "S1": (60, 1000, (2.0, 1.0), 0.6),
    "S2": (8, 1000, (-3.5, 4.5), 0.3),
    "S3": (8, 1000, (-4.0, -3.5), 0.9),
}


def draw(rng, which):
    Xs, ys, ss = [], [], []
    for k, (ntr, nte, mu, sd) in SPEC.items():
        n = ntr if which == "train" else nte
        Xs.append(rng.normal(mu, sd, size=(n, 2)))
        ys += [0 if k == "maj" else 1] * n
        ss += [{"maj": -1, "S1": 0, "S2": 1, "S3": 2}[k]] * n
    return np.vstack(Xs), np.array(ys), np.array(ss)


def synth_per_sub(X_tr, y_tr, s_tr, X_res, y_res):
    """Assign each synthetic point to the sub-concept of its nearest real minority point."""
    syn = X_res[len(X_tr):]
    out = {0: 0, 1: 0, 2: 0}
    if len(syn) == 0:
        return out
    real = X_tr[y_tr == 1]
    lab = s_tr[y_tr == 1]
    _, i = NearestNeighbors(n_neighbors=1).fit(real).kneighbors(syn)
    for s, c in zip(*np.unique(lab[i[:, 0]], return_counts=True)):
        out[int(s)] = int(c)
    return out


def main(n_seeds=20):
    rows, counts = [], []
    for seed in range(n_seeds):
        rng = np.random.RandomState(seed)
        X_tr, y_tr, s_tr = draw(rng, "train")
        X_te, y_te, s_te = draw(rng, "test")
        sc = StandardScaler().fit(X_tr)
        X_tr_s, X_te_s = sc.transform(X_tr), sc.transform(X_te)
        samplers = {"None": NoSampling(), "SMOTE": SMOTE(random_state=seed),
                    "AW-SMOTE": AWSMOTE(random_state=seed), "RC-AW-SMOTE (ours)": RCAWSMOTE(random_state=seed)}
        for sname, smp in samplers.items():
            Xr, yr = smp.fit_resample(X_tr_s, y_tr)
            if sname != "None":
                c = synth_per_sub(X_tr_s, y_tr, s_tr, Xr, yr)
                info = getattr(smp, "last_info_", {})
                counts.append(dict(seed=seed, method=sname, syn_S1=c[0], syn_S2=c[1], syn_S3=c[2],
                                   noise_dropped=info.get("n_noise_dropped", 0)))
            for cname, clf in make_classifiers(seed).items():
                clf.fit(Xr, yr)
                m = evaluate(clf, X_te_s, y_te, s_te)
                pred = clf.predict(X_te_s)
                m.update({f"recall_{k}": (pred[s_te == v] == 1).mean() for k, v in [("S1", 0), ("S2", 1), ("S3", 2)]})
                rows.append(dict(seed=seed, method=sname, classifier=cname, **m))
    res = pd.DataFrame(rows)
    cnt = pd.DataFrame(counts)
    res.to_csv("results/exp1_synthetic_runs.csv", index=False)
    cnt.to_csv("results/exp1_synthetic_counts.csv", index=False)
    return res, cnt


def figure(seed=0):
    rng = np.random.RandomState(seed)
    X_tr, y_tr, s_tr = draw(rng, "train")
    sc = StandardScaler().fit(X_tr)
    Xs = sc.transform(X_tr)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharex=True, sharey=True)
    for ax, (title, smp) in zip(axes, [("AW-SMOTE (original)", AWSMOTE(random_state=seed)),
                                         ("RC-AW-SMOTE (proposed)", RCAWSMOTE(random_state=seed))]):
        Xr, yr = smp.fit_resample(Xs, y_tr)
        syn = Xr[len(Xs):]
        c = synth_per_sub(Xs, y_tr, s_tr, Xr, yr)
        ax.scatter(*Xs[y_tr == 0].T, s=9, c="#b8bec6", label="Majority", zorder=1)
        ax.scatter(*syn.T, s=7, c="#e8a33d", alpha=0.55, label="Synthetic minority", zorder=2)
        for s, col, nm in [(0, "#2a6fdb", "S1 large (boundary)"), (1, "#c2185b", "S2 rare, compact"),
                           (2, "#2e7d32", "S3 rare, sparse")]:
            ax.scatter(*Xs[s_tr == s].T, s=46, c=col, edgecolor="white", linewidth=0.8, label=nm, zorder=3)
        ax.set_title(f"{title}\nsynthetic samples: S1={c[0]}, S2={c[1]}, S3={c[2]}", fontsize=11)
        ax.set_xlabel("feature 1 (standardised)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("feature 2 (standardised)")
    axes[0].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("Where AW-SMOTE puts its synthetic samples (one training set, seed 0)", fontsize=12)
    fig.tight_layout()
    fig.savefig("figures/fig1_synthetic_demo.png", dpi=170)


if __name__ == "__main__":
    res, cnt = main()
    figure(0)
    print(cnt.groupby("method")[["syn_S1", "syn_S2", "syn_S3", "noise_dropped"]].mean().round(1))
    cols = ["recall_S1", "recall_S2", "recall_S3", "wBA", "G_mean", "F1", "AUC"]
    print(res.groupby(["classifier", "method"])[cols].mean().round(3).to_string())
