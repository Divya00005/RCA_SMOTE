"""Stage 4a: ablation and sensitivity of RCA-SMOTE (the method itself is NOT changed: the existing class is run
with one setting changed at a time; the default row reproduces the frozen method).

Ablation (leave one part out = replace it by AW-SMOTE's version):
  coarse clusters  eps percentile 90 -> 65 (AW-SMOTE)
  rescue           off (all DBSCAN noise discarded, as in AW-SMOTE)
  size quota       -> AW-SMOTE's boundary quota (alpha_q = 1: clusters far from the majority can get 0)
  SMOTE generation -> AW-SMOTE's stage 3 (as published, and with its mask bug fixed)
  all four         -> close to AW-SMOTE itself (sanity check)
Sensitivity (one setting at a time, others at default):
  eps percentile {50, 65, 75, 80, 85, 90*, 95, 99}; quota exponent: n_j ~ |C_j|^(1 - beta), beta {0, .25, .5*, .75, 1};
  k partners {3, 5*, 7, 10}; k for the rescue test {3, 5*, 7, 10}.   (* = RCA-SMOTE default)

  python exp5_ablation.py [--jobs N]
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
from improved import ICOS, VARIANTS
from rca_smote import FINAL_VARIANT
from stage4_common import run_all

FULL = VARIANTS[FINAL_VARIANT]          # dict(gen="smote_global", eps_percentile=90, alpha_q=0.0)
CONFIGS = {
    "RCA-SMOTE (default)": {},
    "ablation: no coarse clusters (eps 65)": dict(eps_percentile=65),
    "ablation: no rescue": dict(use_rescue=False),
    "ablation: AW boundary quota": dict(alpha_q=1.0),
    "ablation: AW generation (published)": dict(gen="aw"),
    "ablation: AW generation (mask fixed)": dict(gen="aw_fixed"),
    "ablation: all four parts removed": dict(eps_percentile=65, use_rescue=False, alpha_q=1.0, gen="aw"),
}
for v in [50, 65, 75, 80, 85, 95, 99]:
    if v != 65:
        CONFIGS[f"eps percentile {v}"] = dict(eps_percentile=v)
for b in [0.0, 0.25, 0.75, 1.0]:
    CONFIGS[f"beta {b:g}"] = dict(beta=b)
for k in [3, 7, 10]:
    CONFIGS[f"k partners {k}"] = dict(k_gen=k)
for k in [3, 7, 10]:
    CONFIGS[f"k rescue test {k}"] = dict(k_neighbors=k)


def make_samplers(seed, n_min):
    return {name: ICOS(random_state=seed, **{**FULL, **over}) for name, over in CONFIGS.items()}


if __name__ == "__main__":
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 18
    run_all(make_samplers, "work/ablation_folds", jobs=jobs)
