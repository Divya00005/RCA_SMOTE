"""RCA-SMOTE: Rare-Cluster-Aware SMOTE (the final proposed method).

Usage
-----
    from rca_smote import RCASMOTE
    X_res, y_res = RCASMOTE(random_state=0).fit_resample(X, y)

Binary problems only (the smaller class is treated as the minority class). Features should be scaled.

What it does (see the report for details):
  1. DBSCAN on the minority class with MinPts = 2 and eps = 90th percentile of nearest-neighbour distances.
  2. Rescue: a DBSCAN noise point is removed only if all of its 5 nearest training neighbours are majority
     points; other noise points join a nearby cluster or form their own micro-cluster.
  3. Quota by cluster size: n_j proportional to |C_j|^0.5, with rounding and a residual correction; no universal minimum is guaranteed.
  4. SMOTE-style generation: seeds spread over each cluster, partner among the 5 nearest kept minority
     points, lambda ~ U(0, 1).

Internally this is variant "V7" of improved.ICOS, the variant selected on the development datasets and
evaluated once on the held-out datasets.
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

from improved import ICOS, VARIANTS

FINAL_VARIANT = "V7 V4 with global SMOTE partners"


class RCASMOTE(ICOS):
    def __init__(self, random_state=42):
        super().__init__(random_state=random_state, **VARIANTS[FINAL_VARIANT])
