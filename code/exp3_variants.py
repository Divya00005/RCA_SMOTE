"""Experiment 3: develop variants on the DEV datasets, then evaluate the chosen one on HELD-OUT datasets.

  python exp3_variants.py dev                 -> all variants + SMOTE + AW-SMOTE on the 5 dev datasets
  python exp3_variants.py heldout "<variant>"  -> chosen variant + baselines on the 10 held-out datasets
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import json
import os
import sys
import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from common import bellinger_binary, make_classifiers, evaluate, SafeKMeansSMOTE, NoSampling
from rc_aw_smote import AWSMOTE, RCAWSMOTE
from improved import ICOS, VARIANTS

P = json.load(open("results/protocol_stage2.json"))


def samplers(seed, mode, chosen=None, only=None):
    base = {"SMOTE": SMOTE(k_neighbors=5, random_state=seed), "AW-SMOTE": AWSMOTE(random_state=seed)}
    if mode == "dev":
        keep = [k for k in VARIANTS if (only is None or k in only)]
        base.update({k: ICOS(random_state=seed, **VARIANTS[k]) for k in keep})
    else:
        base.update({"None": NoSampling(), "Borderline-SMOTE": BorderlineSMOTE(k_neighbors=5, random_state=seed),
                     "KMeans-SMOTE": SafeKMeansSMOTE(random_state=seed),
                     "RC-AW-SMOTE": RCAWSMOTE(random_state=seed),
                     f"FINAL {chosen}": ICOS(random_state=seed, **VARIANTS[chosen])})
    return base


def run(name, mode, chosen=None, only=None, tag=""):
    t0 = time.time()
    X, y, sub, info = bellinger_binary(name)
    strat = np.where(y == 0, -1, sub)
    rows = []
    for fold, (tr, te) in enumerate(RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state=0).split(X, strat)):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for sname, smp in samplers(fold, mode, chosen, only).items():
            Xr, yr = smp.fit_resample(Xtr, y[tr])
            for cname, clf in make_classifiers(fold).items():
                clf.fit(Xr, yr)
                rows.append(dict(dataset=name, fold=fold, method=sname, classifier=cname,
                                 **evaluate(clf, Xte, y[te], sub[te])))
    df = pd.DataFrame(rows)
    os.makedirs(f"results/stage2_{mode}{tag}", exist_ok=True)
    df.to_csv(f"results/stage2_{mode}{tag}/{name}.csv", index=False)
    print(f"{name}: {time.time() - t0:.0f}s", flush=True)
    return df


if __name__ == "__main__":
    mode = sys.argv[1]
    chosen = sys.argv[2] if len(sys.argv) > 2 else None
    names = P["dev_datasets"] if mode == "dev" else P["heldout_datasets"]
    out = Parallel(n_jobs=2)(delayed(run)(n, mode, chosen) for n in names)
    pd.concat(out).to_csv(f"results/stage2_{mode}_all.csv", index=False)
    print("saved")
