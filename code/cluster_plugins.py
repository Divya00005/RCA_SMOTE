"""Plug-in fix: RCA-SMOTE's rescue rule and size-based quota added to two other cluster-based oversamplers
(smote-variants' kmeans_SMOTE and DBSMOTE). Each method keeps its own clustering and its own way of
generating points; only the two parts below are switched on or off.

  rescue : a minority point that the method would discard is kept unless all 5 of its nearest training
           neighbours are majority points (Napierala & Stefanowski 2016 "outlier"; same test as RCA-SMOTE).
           kmeans_SMOTE discards whole k-means clusters (too many majority points, or 1 minority point):
               their minority points are kept as their own cluster; a cluster left with one point joins
               the cluster of its nearest other minority point.
           DBSMOTE discards DBSCAN noise points: a kept noise point joins the nearest cluster when it is
               within tau (95th percentile of minority nearest-neighbour distances), otherwise rescued
               points form micro-clusters by single linkage (the RCA-SMOTE rule).
  quota  : cluster j receives n_j proportional to |C_j|^0.5 synthetic points, at least 1 (RCA-SMOTE's quota),
           instead of the method's own random allocation (kmeans_SMOTE: by cluster sparsity;
           DBSMOTE: by cluster size).

With rescue=False and quota=False the original library method is called unchanged.
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import logging
import warnings
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import smote_variants as sv

logging.getLogger("smote_variants").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def true_noise_mask(X, y, idx_min, min_label, k=5):
    """True for minority points (indices into X) whose k nearest training neighbours are all majority."""
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(X))).fit(X)
    _, ind = nn.kneighbors(X[idx_min])
    return (y[ind[:, 1:]] != min_label).all(axis=1)


def size_quota(sizes, total):
    s = np.asarray(sizes, float) ** 0.5
    n = np.round(s / s.sum() * total).astype(int)
    n[n < 1] = 1
    diff = total - n.sum()
    while diff != 0:                       # keep the total exact without creating zeros
        j = np.argmax(n) if diff < 0 else np.argmax(s)
        step = -1 if diff < 0 else 1
        if n[j] + step >= 1:
            n[j] += step
            diff -= step
        else:
            break
    return n


class KMeansSMOTEPlugin(sv.kmeans_SMOTE):
    def __init__(self, rescue=False, quota=False, **kw):
        super().__init__(**kw)
        self.rescue, self.quota = rescue, quota
        self.last_info_ = {}

    def sampling_algorithm(self, X, y):
        n_to_sample = self.det_n_to_sample(self.proportion)
        idx_min_all = np.where(y == self.min_label)[0]
        n_min = len(idx_min_all)
        # the library's clustering (k-means on all points) and its filter
        filt = self.do_the_clustering(X, y)
        kept = set(np.concatenate([c[y[c] == self.min_label] for c in filt])) if filt else set()
        self.last_info_ = dict(n_discarded_by_method=n_min - len(kept))
        if not self.rescue and not self.quota:
            out = super().sampling_algorithm(X, y)
            self.last_info_.update(n_noise_dropped=n_min - len(kept))
            return out
        if self.rescue:
            n_clusters = min(self.n_clusters, len(X))
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=self._random_state_init).fit(X)
            noise = true_noise_mask(X, y, idx_min_all, self.min_label)
            keep_min = idx_min_all[~noise]
            groups = [keep_min[km.labels_[keep_min] == c] for c in range(n_clusters)]
            groups = [g for g in groups if len(g)]
            # a group with a single point joins the group of its nearest other kept minority point
            singles = [g[0] for g in groups if len(g) == 1]
            groups = [list(g) for g in groups if len(g) > 1]
            if not groups:                     # every group is a single point: one group with all of them
                groups = [list(singles)] if len(singles) > 1 else []
                singles = []
            for s in singles:
                others = np.array([i for g in groups for i in g])
                j = others[np.argmin(np.linalg.norm(X[others] - X[s], axis=1))]
                for g in groups:
                    if j in g:
                        g.append(s)
                        break
            clusters = [np.array(g) for g in groups]
            self.last_info_.update(n_noise_dropped=int(noise.sum()))
        else:
            clusters = [c[y[c] == self.min_label] for c in filt]
            self.last_info_.update(n_noise_dropped=n_min - len(kept))
        if not clusters:
            self.last_info_.update(quota=[])
            return X.copy(), y.copy()
        weights, cmi, nns = self.calculate_weights(clusters, X, y)
        if self.quota:
            n_j = size_quota([len(c) for c in cmi], n_to_sample)
        else:
            sel = self.random_state.choice(len(weights), n_to_sample, p=weights)
            n_j = np.bincount(sel, minlength=len(weights))
        self.last_info_.update(quota=[int(v) for v in n_j])
        samples = [self.sample_simplex(X=X[cmi[j]], indices=nns[j][1], n_to_sample=int(n_j[j]))
                   for j in range(len(cmi)) if n_j[j] > 0]
        return np.vstack([X, np.vstack(samples)]), np.hstack([y, np.repeat(self.min_label, sum(len(s) for s in samples))])


class DBSMOTEPlugin(sv.DBSMOTE):
    def __init__(self, rescue=False, quota=False, **kw):
        super().__init__(**kw)
        self.rescue, self.quota = rescue, quota
        self.last_info_ = {}

    def sampling_algorithm(self, X, y):
        n_to_sample = self.det_n_to_sample(self.proportion)
        scaler = StandardScaler().fit(X)
        X_ss = scaler.transform(X)
        idx_min_all = np.where(y == self.min_label)[0]
        X_min = X_ss[idx_min_all]
        labels = self.execute_dbscan(X_min).copy()
        n_noise = int((labels == -1).sum())
        self.last_info_ = dict(n_discarded_by_method=n_noise)
        if not self.rescue and not self.quota:
            out = super().sampling_algorithm(X, y)
            self.last_info_.update(n_noise_dropped=n_noise)
            return out
        if self.rescue and n_noise:
            noise_idx = np.where(labels == -1)[0]
            true_noise = true_noise_mask(X_ss, y, idx_min_all[noise_idx], self.min_label)
            rescue_idx = noise_idx[~true_noise]
            d1 = NearestNeighbors(n_neighbors=2).fit(X_min).kneighbors(X_min)[0][:, 1]
            tau = np.percentile(d1, 95)
            clustered = np.where(labels != -1)[0]
            unattached = []
            if len(clustered):
                d_c, i_c = NearestNeighbors(n_neighbors=1).fit(X_min[clustered]).kneighbors(X_min[rescue_idx])
                for r, dd, ii in zip(rescue_idx, d_c[:, 0], i_c[:, 0]):
                    if dd <= tau:
                        labels[r] = labels[clustered[ii]]
                    else:
                        unattached.append(r)
            else:
                unattached = list(rescue_idx)
            if len(unattached) == 1:
                labels[unattached[0]] = labels.max() + 1
            elif len(unattached) > 1:
                ua = np.array(unattached)
                d_ua = NearestNeighbors(n_neighbors=2).fit(X_min[ua]).kneighbors(X_min[ua])[0][:, 1]
                tau_r = max(tau, 2.0 * np.median(d_ua))
                sub = AgglomerativeClustering(n_clusters=None, linkage="single",
                                              distance_threshold=tau_r).fit(X_min[ua]).labels_
                labels[ua] = sub + labels.max() + 1
            self.last_info_.update(n_noise_dropped=int(true_noise.sum()))
        else:
            self.last_info_.update(n_noise_dropped=n_noise)
        num = labels.max() + 1
        if num == 0:
            self.last_info_.update(quota=[])
            return X.copy(), y.copy()
        clusters = [np.where(labels == i)[0] for i in range(num)]
        clusters = [c for c in clusters if len(c)]
        nn_params = {**self.nn_params}
        nn_params["metric_tensor"] = self.metric_tensor_from_nn_params(nn_params, X, y)
        paths = self.extract_shortest_paths(X_min, clusters, nn_params)
        if self.quota:
            n_j = size_quota([len(c) for c in clusters], n_to_sample)
            samples = []
            for j, c in enumerate(clusters):
                samples += self.generate_samples(int(n_j[j]), [c], [paths[j]], X_min)
        else:
            samples = self.generate_samples(n_to_sample, clusters, paths, X_min)
            n_j = []
        self.last_info_.update(quota=[int(v) for v in n_j])
        return (np.vstack([X, scaler.inverse_transform(np.vstack(samples))]),
                np.hstack([y, np.repeat(self.min_label, len(samples))]))


class SVWrapper:
    """imblearn-style fit_resample for a smote-variants object."""

    def __init__(self, obj):
        self.obj = obj

    @property
    def last_info_(self):
        return getattr(self.obj, "last_info_", {})

    def fit_resample(self, X, y):
        return self.obj.sample(np.asarray(X, float), np.asarray(y))
