"""
RC-AW-SMOTE: Rare-Cluster-aware AW-SMOTE (proposed method).

Built directly on the authors' AW-SMOTE implementation (aw_smote_original.py).
Everything AW-SMOTE does is kept, except two changes that target the
limitations studied in this project:

Fix 1 - Quota floor (Stage 2).
    AW-SMOTE gives cluster j the weight W_j = max_i w_b,i, where w_b,i is the
    number of majority points among x_i's 5 nearest neighbours. A minority
    sub-cluster that is far from the majority class (a "safe" rare sub-concept)
    has W_j = 0 and therefore receives 0 synthetic samples.
    We mix the boundary share with a size-based share:

        q_j = alpha * (W_j / sum W) + (1 - alpha) * (|C_j|^(1-beta) / sum |C|^(1-beta))

    alpha = 1           -> original AW-SMOTE allocation
    beta  = 0           -> size share proportional to cluster size (SMOTE-like floor)
    beta  = 1           -> every cluster gets the same size share (full intra-class balancing)
    Default alpha = 0.5, beta = 0.5 (chosen a priori, not tuned on test data).

Fix 2 - Rescue of sparse minority points (Stage 1).
    The authors' code sets DBSCAN eps to the 65th percentile of nearest-neighbour
    distances with MinPts = 2, so roughly the sparsest third of minority points
    is labelled noise and never used for oversampling. Sparse rare sub-concepts
    are exactly these points.
    We only treat a DBSCAN-noise point as real noise if ALL of its k nearest
    neighbours (in the whole training set) are majority points. Other noise
    points are rescued: they join the nearest existing cluster when they are
    close to it, otherwise they form their own micro-clusters.
    Single-point clusters cannot be interpolated, so new points are drawn in a
    small ball around them whose radius is less than half the distance to the
    nearest majority point (so synthetic points never cross into majority space).
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import contextlib
import io
import numpy as np
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN, AgglomerativeClustering

from aw_smote_original import ADWSMOTE


def _safe_eps(nn_dist, pct):
    """AW-SMOTE's eps rule (percentile of nearest-neighbour distance). If many minority points are exact
    duplicates the percentile is 0 and DBSCAN raises an error (happens on led7). Guard, applied identically
    to AW-SMOTE and RC-AW-SMOTE: fall back to the smallest positive nearest-neighbour distance."""
    eps = np.percentile(np.sort(nn_dist), pct)
    if eps <= 0:
        pos = nn_dist[nn_dist > 0]
        eps = pos.min() if len(pos) else 1e-6
    return eps


class RCAWSMOTE(ADWSMOTE):
    def __init__(self, alpha_q=0.5, beta=0.5, use_floor=True, use_rescue=True,
                 rescue_percentile=95, random_state=42, **kw):
        super().__init__(random_state=random_state, **kw)
        self.alpha_q = alpha_q
        self.beta = beta
        self.use_floor = use_floor
        self.use_rescue = use_rescue
        self.rescue_percentile = rescue_percentile
        self.seed_ = random_state
        self.last_info_ = {}

    # ---------- Stage 1 with rescue ----------
    def _stage1(self, S_min, X, y, minority_label):
        nn = NearestNeighbors(n_neighbors=self.min_samples).fit(S_min)
        dist, _ = nn.kneighbors(S_min)
        eps = _safe_eps(dist[:, -1], self.eps_percentile)
        labels = DBSCAN(eps=eps, min_samples=self.min_samples).fit(S_min).labels_.copy()

        if np.all(labels == -1):             # same fallback as the original code
            labels[:] = 0

        noise_idx = np.where(labels == -1)[0]
        n_true_noise = len(noise_idx)
        n_rescued = 0

        if self.use_rescue and len(noise_idx) > 0:
            # true noise test: all k nearest neighbours (excluding itself) are majority
            k = self.k_neighbors
            nn_all = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
            _, ind = nn_all.kneighbors(S_min[noise_idx])
            is_maj = (y[ind[:, 1:]] != minority_label)
            true_noise = is_maj.all(axis=1)
            rescue_idx = noise_idx[~true_noise]
            n_true_noise = int(true_noise.sum())
            n_rescued = len(rescue_idx)

            if n_rescued > 0:
                tau = np.percentile(dist[:, -1], self.rescue_percentile)
                clustered = np.where(labels != -1)[0]
                unattached = []
                if len(clustered) > 0:
                    nn_c = NearestNeighbors(n_neighbors=1).fit(S_min[clustered])
                    d_c, i_c = nn_c.kneighbors(S_min[rescue_idx])
                    for r, dd, ii in zip(rescue_idx, d_c[:, 0], i_c[:, 0]):
                        if dd <= tau:
                            labels[r] = labels[clustered[ii]]       # joins nearby cluster
                        else:
                            unattached.append(r)
                else:
                    unattached = list(rescue_idx)
                # far rescued points form their own micro-clusters (single linkage).
                # Sparse sub-concepts have wider spacing, so the linking distance adapts
                # to the rescued points' own nearest-neighbour spacing.
                if len(unattached) == 1:
                    labels[unattached[0]] = labels.max() + 1
                elif len(unattached) > 1:
                    ua = np.array(unattached)
                    d_ua = NearestNeighbors(n_neighbors=2).fit(S_min[ua]).kneighbors(S_min[ua])[0][:, 1]
                    tau_r = max(tau, 2.0 * np.median(d_ua))
                    sub = AgglomerativeClustering(n_clusters=None, linkage="single",
                                                  distance_threshold=tau_r).fit(S_min[ua]).labels_
                    labels[ua] = sub + labels.max() + 1

        clusters = [S_min[labels == c] for c in np.unique(labels) if c != -1]
        self.last_info_.update(eps=eps, n_noise_dropped=n_true_noise,
                               n_rescued=n_rescued, n_clusters=len(clusters),
                               cluster_sizes=[len(c) for c in clusters])
        return clusters, labels

    # ---------- Stage 2 with quota floor ----------
    def _stage2(self, clusters, total_gen, X, y):
        if not self.use_floor:
            with contextlib.redirect_stdout(io.StringIO()):
                return super()._phase2_quota_allocation(clusters, total_gen, X, y)
        knn = NearestNeighbors(n_neighbors=self.k_neighbors).fit(X)
        maj_mask = (y != Counter(y).most_common()[-1][0])
        W = []
        for C in clusters:
            _, ind = knn.kneighbors(C)
            W.append(maj_mask[ind].sum(axis=1).max())
        W = np.array(W, dtype=float)
        b = W / W.sum() if W.sum() > 0 else np.ones_like(W) / len(W)
        sizes = np.array([len(c) for c in clusters], dtype=float)
        s = sizes ** (1.0 - self.beta)
        s = s / s.sum()
        q = self.alpha_q * b + (1 - self.alpha_q) * s
        n_j = np.round(q * total_gen).astype(int)
        diff = total_gen - n_j.sum()
        if diff != 0:
            n_j[np.argmax(n_j)] += diff
        return n_j

    # ---------- Stage 3: original for |C|>=2, ball sampling for singletons ----------
    def _stage3(self, clusters, n_j, X, y, minority_label):
        S_maj = X[y != minority_label]
        multi = [(c, n) for c, n in zip(clusters, n_j) if len(c) >= 2]
        single = [(c, n) for c, n in zip(clusters, n_j) if len(c) == 1 and n > 0]
        parts = []
        if multi:
            cs, ns = zip(*multi)
            parts.append(super()._phase3_intracluster_sampling(list(cs), np.array(ns), S_maj, y, minority_label))
        if single:
            nn_maj = NearestNeighbors(n_neighbors=1).fit(S_maj)
            all_min = np.vstack(clusters)
            nn_min = NearestNeighbors(n_neighbors=2).fit(all_min)
            d = X.shape[1]
            for c, n in single:
                x = c[0]
                d_maj = nn_maj.kneighbors([x])[0][0, 0]
                d_min = nn_min.kneighbors([x])[0][0, 1] if len(all_min) > 1 else d_maj
                r = 0.5 * min(d_maj, d_min)
                dirs = self.rng.normal(size=(n, d))
                dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
                rad = r * self.rng.uniform(size=(n, 1)) ** (1.0 / d)
                parts.append(x + dirs * rad)
        parts = [p for p in parts if len(p)]
        return np.vstack(parts) if parts else np.empty((0, X.shape[1]))

    def fit_resample(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        self.rng = np.random.RandomState(self.seed_)
        self.last_info_ = {}
        minority_label = Counter(y).most_common()[-1][0]
        S_min = X[y == minority_label]
        S_maj = X[y != minority_label]
        N = len(S_maj) - len(S_min)
        if N <= 0:
            return X, y
        clusters, _ = self._stage1(S_min, X, y, minority_label)
        n_j = self._stage2(clusters, N, X, y)
        self.last_info_["quota"] = [int(v) for v in n_j]
        with contextlib.redirect_stdout(io.StringIO()):
            S_syn = self._stage3(clusters, n_j, X, y, minority_label)
        if len(S_syn):
            return np.vstack([X, S_syn]), np.hstack([y, np.full(len(S_syn), minority_label)])
        return X, y


class AWSMOTE(ADWSMOTE):
    """Thin wrapper around the authors' code: silences prints, records cluster info."""
    def __init__(self, random_state=42, **kw):
        super().__init__(random_state=random_state, **kw)
        self.seed_ = random_state
        self.last_info_ = {}

    def _phase1_cluster_denoise(self, S_min):
        # identical to the authors' phase 1, except for the eps = 0 guard (_safe_eps)
        knn = NearestNeighbors(n_neighbors=self.min_samples).fit(S_min)
        dist, _ = knn.kneighbors(S_min)
        eps = _safe_eps(dist[:, -1], self.eps_percentile)
        db = DBSCAN(eps=eps, min_samples=self.min_samples).fit(S_min)
        core_mask = db.labels_ != -1
        S_min_clean = S_min[core_mask]
        cluster_labels = db.labels_[core_mask]
        if np.unique(cluster_labels).size == 0:
            S_min_clean = S_min
            cluster_labels = np.zeros(len(S_min_clean), dtype=int)
        clusters = [S_min_clean[cluster_labels == l] for l in np.unique(cluster_labels) if l != -1]
        return clusters, cluster_labels

    def fit_resample(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        minority_label = Counter(y).most_common()[-1][0]
        S_min = X[y == minority_label]
        with contextlib.redirect_stdout(io.StringIO()):
            clusters, _ = self._phase1_cluster_denoise(S_min)
            N = (y != minority_label).sum() - len(S_min)
            if N > 0 and clusters:
                n_j = self._phase2_quota_allocation(clusters, N, X, y)
                self.last_info_ = dict(cluster_sizes=[len(c) for c in clusters],
                                       quota=[int(v) for v in n_j],
                                       n_noise_dropped=len(S_min) - sum(len(c) for c in clusters))
            # reset RNG so the output is exactly what the authors' fit_resample gives
            self.rng = np.random.RandomState(self.seed_)
            return super().fit_resample(X, y)
