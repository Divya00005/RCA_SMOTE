"""Stage 4b + 4c: starvation audit of 8 cluster-based oversamplers (smote-variants 1.0.1, library defaults) and
the plug-in fix (RCA-SMOTE's rescue rule and size-based quota added to kmeans_SMOTE and DBSMOTE; cluster_plugins.py).

  python exp6_cluster_audit.py [--jobs N]
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import sys
import smote_variants as sv
from cluster_plugins import KMeansSMOTEPlugin, DBSMOTEPlugin, SVWrapper
from stage4_common import run_all

AUDIT = ["kmeans_SMOTE", "DBSMOTE", "A_SUWO", "CBSO", "CURE_SMOTE", "SOMO", "MWMOTE", "cluster_SMOTE"]
NAMES = {"kmeans_SMOTE": "KMeans-SMOTE (sv)", "DBSMOTE": "DBSMOTE", "A_SUWO": "A-SUWO", "CBSO": "CBSO",
         "CURE_SMOTE": "CURE-SMOTE", "SOMO": "SOMO", "MWMOTE": "MWMOTE", "cluster_SMOTE": "Cluster-SMOTE"}


def make_samplers(seed, n_min):
    out = {}
    for m in AUDIT:
        if m == "kmeans_SMOTE":
            out[NAMES[m]] = SVWrapper(KMeansSMOTEPlugin(random_state=seed))      # = library method
        elif m == "DBSMOTE":
            out[NAMES[m]] = SVWrapper(DBSMOTEPlugin(random_state=seed))          # = library method
        else:
            out[NAMES[m]] = SVWrapper(getattr(sv, m)(random_state=seed))
    for base, cls in [("KMeans-SMOTE (sv)", KMeansSMOTEPlugin), ("DBSMOTE", DBSMOTEPlugin)]:
        out[f"{base} + rescue"] = SVWrapper(cls(rescue=True, random_state=seed))
        out[f"{base} + quota"] = SVWrapper(cls(quota=True, random_state=seed))
        out[f"{base} + rescue + quota"] = SVWrapper(cls(rescue=True, quota=True, random_state=seed))
    return out


if __name__ == "__main__":
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 18
    run_all(make_samplers, "work/cluster_folds", jobs=jobs)
