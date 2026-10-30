"""Stage-2 variants that target the three diagnosed causes (fragmentation, low coverage, majority intrusion).

All variants extend RC-AW-SMOTE. Options:
  eps_percentile : DBSCAN radius rule (AW-SMOTE uses 65). Larger -> fewer, more meaningful clusters.
  gen            : 'aw'       = authors' stage-3 (includes the mask misalignment)
                   'aw_fixed' = authors' stage-3 logic with the majority mask aligned correctly
                   'smote'    = SMOTE-style generation inside each cluster: seeds spread uniformly over
                                the cluster, partner = one of the k nearest points of the same cluster,
                                lambda ~ U(0, 1)
  alpha_q, beta  : quota mix (alpha_q = weight on AW-SMOTE boundary share; beta: size share ~ |C|^(1-beta))
  reject         : drop/retry a synthetic point if >= 4 of its 5 nearest REAL training points are majority
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
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from rc_aw_smote import RCAWSMOTE


class ICOS(RCAWSMOTE):
    def __init__(self, gen="aw", reject=False, k_gen=5, max_tries=10, **kw):
        super().__init__(**kw)
        self.gen = gen
        self.reject = reject
        self.k_gen = k_gen
        self.max_tries = max_tries

    # ---- correct version of the authors' stage 3 (only the majority mask differs) ----
    def _aw_fixed(self, C, n, X, y, minority_label, knn_all):
        _, ind = knn_all.kneighbors(C)
        w_b = (y[ind] != minority_label).sum(axis=1).astype(float)
        wb = w_b / w_b.sum() if w_b.sum() > 0 else np.ones(len(C)) / len(C)
        D = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
        dbar = D.sum(axis=1) / (len(C) - 1)
        rho = 1 / (dbar + 1e-8)
        wd = rho / rho.sum()
        a = 1 / (1 + np.exp(-self.gamma * (wb - self.theta)))
        w = a * wb + (1 - a) * wd
        p = w / w.sum()
        out = []
        for i in self.rng.choice(len(C), size=n, p=p):
            k = self.rng.randint(len(C) - 1)
            k = k + 1 if k >= i else k
            out.append(C[i] + self.rng.uniform(0.2, 0.8) * (C[k] - C[i]))
        return np.array(out)

    def _smote_gen(self, C, n):
        k = min(self.k_gen, len(C) - 1)
        _, nb = NearestNeighbors(n_neighbors=k + 1).fit(C).kneighbors(C)
        seeds = np.arange(n) % len(C)          # spread seeds evenly over all cluster points
        self.rng.shuffle(seeds)
        out = []
        for i in seeds:
            j = nb[i, 1 + self.rng.randint(k)]
            out.append(C[i] + self.rng.uniform(0, 1) * (C[j] - C[i]))
        return np.array(out)

    def _smote_global(self, C, n, all_min):
        """Seeds come from this cluster (so the quota decides WHERE samples go), but the partner is one of
        the k nearest points among ALL minority points kept after denoising, as in plain SMOTE."""
        k = min(self.k_gen, len(all_min) - 1)
        _, nb = NearestNeighbors(n_neighbors=k + 1).fit(all_min).kneighbors(C)
        seeds = np.arange(n) % len(C)
        self.rng.shuffle(seeds)
        return np.array([C[i] + self.rng.uniform(0, 1) * (all_min[nb[i, 1 + self.rng.randint(k)]] - C[i])
                         for i in seeds])

    def _ball(self, x, n, X, y, minority_label, all_min):
        S_maj = X[y != minority_label]
        d_maj = NearestNeighbors(n_neighbors=1).fit(S_maj).kneighbors([x])[0][0, 0]
        d_min = NearestNeighbors(n_neighbors=2).fit(all_min).kneighbors([x])[0][0, 1] if len(all_min) > 1 else d_maj
        r = 0.5 * min(d_maj, d_min)
        d = X.shape[1]
        dirs = self.rng.normal(size=(n, d))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
        return x + dirs * r * self.rng.uniform(size=(n, 1)) ** (1.0 / d)

    def _generate(self, C, n, X, y, minority_label, knn_all, all_min):
        if n <= 0:
            return np.empty((0, X.shape[1]))
        if len(C) == 1 and self.gen != "smote_global":
            return self._ball(C[0], n, X, y, minority_label, all_min)
        if self.gen == "smote":
            return self._smote_gen(C, n)
        if self.gen == "smote_global":
            return self._smote_global(C, n, all_min)
        return self._aw_fixed(C, n, X, y, minority_label, knn_all)

    def _stage3(self, clusters, n_j, X, y, minority_label):
        if self.gen == "aw" and not self.reject:
            return super()._stage3(clusters, n_j, X, y, minority_label)
        knn_all = NearestNeighbors(n_neighbors=self.k_neighbors).fit(X)
        all_min = np.vstack(clusters)
        parts = []
        for C, n in zip(clusters, n_j):
            if n <= 0:
                continue
            if not self.reject:
                parts.append(self._generate(C, n, X, y, minority_label, knn_all, all_min))
                continue
            kept, need, tries = [], n, 0
            while need > 0 and tries < self.max_tries:
                cand = self._generate(C, need, X, y, minority_label, knn_all, all_min)
                _, ind = knn_all.kneighbors(cand)
                ok = (y[ind] != minority_label).sum(axis=1) < 4
                kept.append(cand[ok])
                need -= ok.sum()
                tries += 1
            if need > 0:   # could not find enough safe positions: keep the remaining candidates anyway
                kept.append(self._generate(C, need, X, y, minority_label, knn_all, all_min))
            parts.append(np.vstack(kept))
        parts = [p for p in parts if len(p)]
        return np.vstack(parts) if parts else np.empty((0, X.shape[1]))


VARIANTS = {
    "V0 RC-AW-SMOTE":                     dict(),
    "V1 +bug fix":                        dict(gen="aw_fixed"),
    "V2 +bug fix +coarser clusters":      dict(gen="aw_fixed", eps_percentile=90),
    "V3 +coarser +SMOTE-style gen":       dict(gen="smote", eps_percentile=90),
    "V4 V3 +size-only quota":             dict(gen="smote", eps_percentile=90, alpha_q=0.0),
    "V5 V3 +equal quota per cluster":     dict(gen="smote", eps_percentile=90, alpha_q=0.0, beta=1.0),
    "V6 V4 +reject majority-area points": dict(gen="smote", eps_percentile=90, alpha_q=0.0, reject=True),
    # second (last) development round, added after seeing round-1 dev results:
    "V7 V4 with global SMOTE partners":   dict(gen="smote_global", eps_percentile=90, alpha_q=0.0),
    "V8 V7 with original AW eps (65)":    dict(gen="smote_global", alpha_q=0.0),
}
