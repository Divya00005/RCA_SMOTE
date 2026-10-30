"""Download the datasets used in the experiments.

- PMLB (MIT licence): the 15 stage-1/2 datasets and the 5 new stage-3 Group A datasets,
  saved to data/<name>.tsv.gz. Source: https://github.com/EpistasisLab/pmlb
- imbalanced-learn benchmark collection (stage-3 Group B), saved to data/imblearn/.
- OpenML-CC18 (4 stage-3 Group A datasets), saved like the PMLB files.
- KEEL (stage-3 Group B, AW-SMOTE's 13 datasets + 8 more) cannot be downloaded by this script: download the
  zip files by hand from the links printed below and save them as data/keel/<name>.zip.
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import pathlib
import sys
import urllib.request

ROOT = PROJECT_ROOT
sys.path.insert(0, str(CODE))

DATASETS = ["ecoli", "yeast", "vowel", "segmentation", "satimage", "pendigits", "dermatology",
            "page_blocks", "optdigits", "texture", "letter", "wine_quality_white", "new_thyroid",
            "led7", "shuttle"]
STAGE3_NEW = ["analcatdata_authorship", "collins", "mfeat_factors", "movement_libras", "sleep"]
BASE = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"
DATA = ROOT / "data" / "raw"
DATA.mkdir(exist_ok=True, parents=True)

for name in DATASETS + STAGE3_NEW:
    out = DATA / f"{name}.tsv.gz"
    if out.exists() and out.stat().st_size > 1000:
        print(f"{name}: already present")
        continue
    out.write_bytes(urllib.request.urlopen(BASE.format(name), timeout=120).read())
    print(f"{name}: {out.stat().st_size // 1024} KB")

if "--stage12-only" not in sys.argv:
    from imblearn.datasets import fetch_datasets
    from stage3_datasets import IMBLEARN_B, IMBLEARN_DIR, KEEL_ALL, KEEL_BASE, keel_path, OPENML_A_NEW
    fetch_datasets(data_home=str(ROOT / IMBLEARN_DIR), filter_data=tuple(IMBLEARN_B))
    print("imblearn datasets:", ", ".join(IMBLEARN_B))
    import pandas as pd
    from sklearn.datasets import fetch_openml
    for n, i in OPENML_A_NEW.items():
        out = DATA / f"{n}.tsv.gz"
        if out.exists():
            print(f"{n}: already present")
            continue
        b = fetch_openml(data_id=i, as_frame=True, parser="auto", data_home=str(DATA / "openml"))
        df = b.data.apply(pd.to_numeric, errors="raise")
        df["target"] = b.target.astype(str).values
        df.to_csv(out, sep="\t", index=False)
        print(f"{n}: OpenML id {i}")
    missing = [n for n in KEEL_ALL if not (ROOT / keel_path(n)).exists()]
    if missing:
        print(f"\nKEEL: {len(missing)} file(s) missing. Download and save as data/keel/<name>.zip:")
        for n in missing:
            print(f"  {KEEL_BASE}{KEEL_ALL[n]}/{n}.zip")
    else:
        print(f"KEEL: all {len(KEEL_ALL)} files present")
