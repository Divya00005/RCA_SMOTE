"""Stage 3 datasets: lists, selection rule and loaders.

Group A: multi-class datasets turned into binary problems with KNOWN minority sub-groups
         (Bellinger protocol, common.bellinger_binary, majority capped at 2000).
Group B: standard binary imbalanced benchmarks (sub-groups unknown, standard metrics only).

The full selection rule, every included dataset and every excluded candidate (with the reason) are written
to results/supporting/stage3/DATASETS.md and results/supporting/stage3/protocol_stage3.json by `python stage3_datasets.py`.
This was done BEFORE any stage-3 result was looked at.
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import io
import json
import os
import zipfile
import numpy as np
import pandas as pd

from common import bellinger_binary

DATA_DIR = str(PROJECT_ROOT / "data" / "raw")
KEEL_DIR = os.path.join(DATA_DIR, "keel")
IMBLEARN_DIR = os.path.join(DATA_DIR, "imblearn")
OUT_DIR = os.path.join("results", "supporting", "stage3")

# ------------------------------------------------------------------ Group A
# 15 datasets from stages 1-2 (5 were used to choose the RCA-SMOTE design -> reported separately)
GROUP_A_DEV = ["ecoli", "led7", "shuttle", "texture", "wine_quality_white"]
GROUP_A_STAGE2_HELDOUT = ["yeast", "vowel", "segmentation", "satimage", "pendigits", "dermatology",
                          "page_blocks", "optdigits", "letter", "new_thyroid"]
# new PMLB datasets selected by the rule below (never seen by any stage-1/2 experiment)
GROUP_A_NEW = ["analcatdata_authorship", "collins", "mfeat_factors", "movement_libras", "sleep"]
# added because PMLB has no more datasets that pass the rule: OpenML-CC18 suite (Bischl et al. 2021), same rule.
# Saved in the same format as PMLB files (data/<name>.tsv.gz, column "target") by scripts/fetch_data.py.
OPENML_A_NEW = {"gesture_phase": 4538,                 # GesturePhaseSegmentationProcessed
                "first_order_theorem_proving": 1475,
                "steel_plates_fault": 40982,
                "wall_robot_navigation": 1497}

# ------------------------------------------------------------------ Group B
KEEL_BASE = "https://sci2s.ugr.es/keel/keel-dataset/datasets/imbalanced/"
# AW-SMOTE's 13 datasets (Yan, Guo & Su 2026), file name -> KEEL folder
KEEL_AWSMOTE = {
    "wisconsin": "imb_IRlowerThan9",
    "pima": "imb_IRlowerThan9",
    "haberman": "imb_IRlowerThan9",
    "vehicle0": "imb_IRlowerThan9",
    "ecoli1": "imb_IRlowerThan9",
    "new-thyroid2": "imb_IRlowerThan9",
    "ecoli3": "imb_IRlowerThan9",
    "yeast-0-2-5-6_vs_3-7-8-9": "imb_IRhigherThan9p2",
    "yeast-0-2-5-7-9_vs_3-6-8": "imb_IRhigherThan9p2",
    "yeast-2_vs_8": "imb_IRhigherThan9p1",
    "flare-F": "imb_IRhigherThan9p3",
    "winequality-red-3_vs_5": "imb_IRhigherThan9p3",
    "winequality-red-8_vs_6-7": "imb_IRhigherThan9p3",
}
# 8 more KEEL datasets from families not used anywhere else (added to reach about 50 datasets; rule below)
KEEL_EXTRA = {
    "glass1": "imb_IRlowerThan9",
    "glass-0-1-6_vs_2": "imb_IRhigherThan9p1",
    "glass5": "imb_IRhigherThan9p1",
    "abalone9-18": "imb_IRhigherThan9p1",
    "abalone-21_vs_8": "imb_IRhigherThan9p3",
    "abalone19": "imb_IRhigherThan9p1",
    "cleveland-0_vs_4": "imb_IRhigherThan9p2",
    "iris0": "imb_IRlowerThan9",
}
KEEL_ALL = {**KEEL_AWSMOTE, **KEEL_EXTRA}
# KEEL datasets made from the same raw data as a Group A dataset (kept because they are AW-SMOTE's own
# benchmarks; Group B is also reported without them as a check)
KEEL_SHARES_RAW_WITH_A = {"ecoli1": "ecoli (dev)", "ecoli3": "ecoli (dev)", "new-thyroid2": "new_thyroid",
                          "yeast-0-2-5-6_vs_3-7-8-9": "yeast", "yeast-0-2-5-7-9_vs_3-6-8": "yeast",
                          "yeast-2_vs_8": "yeast"}
IMBLEARN_B = ["spectrometer", "us_crime", "yeast_ml8", "scene", "oil", "ozone_level"]

# ------------------------------------------------------------------ selection rule and exclusions
RULE_A = [
    "Source: PMLB (same source as the 15 stage-1/2 datasets), classification datasets, not marked deprecated.",
    "At least 4 classes (so the Bellinger construction gives a majority of >= 2 classes and >= 2 minority sub-groups).",
    "Numeric features: at least 80% of features continuous according to PMLB's metadata (all_summary_stats.tsv).",
    "Enough samples per class: after the Bellinger construction (classes < 10 samples dropped, sub-groups halved, "
    "each >= 10, majority <= 2000) there are >= 2 minority sub-groups and the imbalance ratio is >= 2.5 "
    "(the smallest IR among the 15 existing datasets is 2.88).",
    "No near-copies: not made from the same raw data as an existing Group A or Group B dataset, and at most one "
    "dataset from a family of views of the same raw data (first in alphabetical order).",
    "Applying this rule to all PMLB datasets gives exactly 5 new datasets; PMLB has no more.",
    "To reach about 30, the same rule was applied to the OpenML-CC18 benchmark suite (72 curated datasets; "
    "additionally: no missing values, <= 300 features). This adds 4: GesturePhaseSegmentationProcessed (id 4538), "
    "first-order-theorem-proving (1475), steel-plates-fault (40982), wall-robot-navigation (1497).",
]
RULE_B = [
    "AW-SMOTE's 13 KEEL datasets, exactly as named in their paper (downloaded manually from the KEEL "
    "imbalanced repository, full-dataset zip, not the pre-split fold files). Nominal features (flare-F only) "
    "are one-hot encoded.",
    "Plus 8 more KEEL binary imbalanced datasets from families not used anywhere else in this study. Families "
    "with numeric or convertible features (at most one nominal attribute, one-hot encoded) and < 10,000 samples: "
    "glass, abalone, cleveland, iris (zoo, lymphography, car, kr-vs-k, poker, kddcup are mostly nominal; "
    "winequality-white, shuttle, page-blocks, segment, vowel, led7digit, dermatology, ecoli, yeast, vehicle, "
    "new-thyroid, wisconsin, pima, haberman, flare, winequality-red are already used). cleveland and iris have "
    "one binary version each (cleveland-0_vs_4, iris0). glass (13 versions) and abalone (7 versions) are many "
    "re-labellings of the same raw data, so from each we take the versions with the lowest, median and highest "
    "imbalance ratio in KEEL's table: glass1 (1.82), glass-0-1-6_vs_2 (10.29), glass5 (22.78); abalone9-18 (16.4), "
    "abalone-21_vs_8 (40.5), abalone19 (129.44).",
    "Plus every dataset of imblearn.datasets.fetch_datasets (27 datasets) EXCEPT: (a) made from the same raw "
    "data as a Group A dataset; (b) categorical features (nominal attributes in the source description, or "
    "> 10% of features binary); (c) very large: > 10,000 samples or > 300 features.",
]
EXCLUDED = [
    # (source, name, reason)
    ("PMLB", "cloud", "imbalance ratio after construction 1.67 < 2.5 (108 samples, 4 similar-size classes)"),
    ("PMLB", "confidence", "imbalance ratio after construction 1.12 < 2.5 (72 samples)"),
    ("PMLB", "mfeat_fourier / mfeat_karhunen / mfeat_zernike", "other views of the same raw data as mfeat_factors"),
    ("PMLB", "mfeat_morphological / mfeat_pixel", "same raw data as mfeat_factors; also < 80% continuous"),
    ("PMLB", "mnist", "near-copy of optdigits/pendigits (NIST handwritten digits)"),
    ("PMLB", "led24", "near-copy of led7"),
    ("PMLB", "vehicle", "same raw data as KEEL vehicle0 (Group B)"),
    ("PMLB", "wine_quality_red", "same raw data as KEEL winequality-red-* (Group B)"),
    ("PMLB", "kddcup, fars, poker, nursery, krkopt, car_evaluation, soybean, lymphography, flags, "
             "calendarDOW, allhyper, allrep, analcatdata_dmft, analcatdata_germangss, auto_insurance_symboling, "
             "horse_colic_lesion_type", "< 80% continuous features (nominal/coded attributes)"),
    ("OpenML-CC18", "mfeat-*, optdigits, pendigits, satimage, segment, texture, letter, vowel, vehicle, "
                    "analcatdata_authorship", "already used (Group A or B) or same raw data"),
    ("OpenML-CC18", "semeion, mnist_784", "near-copies of the digit datasets (semeion is also binary pixels)"),
    ("OpenML-CC18", "har, isolet, cnae-9, Fashion-MNIST, CIFAR_10, Devnagari-Script", "> 300 features"),
    ("OpenML-CC18", "MiceProtein, eucalyptus", "missing values / nominal features"),
    ("OpenML-CC18", "car, analcatdata_dmft, and the CC18 datasets with < 4 classes", "nominal features or < 4 classes"),
    ("KEEL", "zoo-3, lymphography-normal-fibrosis, car-*, kr-vs-k-*, poker-*, kddcup-*", "mostly nominal features"),
    ("KEEL", "other glass and abalone versions", "same raw data as the 3 versions taken from each family"),
    ("KEEL", "winequality-white-*, shuttle-*, page-blocks*, segment0, vowel0, led7digit-*, dermatology-6, "
             "ecoli-*, yeast*, vehicle1-3, new-thyroid1", "family already used in Group A or B"),
    ("PMLB", "_deprecated_* (glass, cleveland, auto, car, solar_flare, ...)", "marked deprecated by PMLB"),
    ("imblearn", "ecoli, optical_digits, satimage, pen_digits, letter_img, yeast_me2, wine_quality",
     "same raw data as a Group A dataset"),
    ("imblearn", "libras_move", "same raw data as movement_libras (Group A)"),
    ("imblearn", "abalone, abalone_19, sick_euthyroid, thyroid_sick, car_eval_34, car_eval_4, "
                 "solar_flare_m0, coil_2000", "categorical features"),
    ("imblearn", "arrhythmia", "categorical: 83 of 278 features are binary"),
    ("imblearn", "isolet (617 features), webpage (34,780 samples), mammography (11,183), protein_homo (145,751)",
     "very large"),
]
# notes added when the KEEL files arrived (no change to the list)
NOTES = [
    "Some KEEL zips also contain macOS helper files (__MACOSX/._*.dat); they are ignored.",
    "cleveland-0_vs_4: the KEEL file has 173 rows (KEEL's web table says 177, probably counting rows with missing "
    "values that are not in the file). The file is used as it is. KEEL stores this dataset's coded attributes "
    "(sex, chest-pain type, ECG result, ...) as numbers; they are used as given, like every other KEEL dataset.",
    "abalone*: the one nominal attribute (Sex) is one-hot encoded, as for flare-F.",
]
# decided before running (written to the protocol file)
PRIMARY = {"A": "wBA (each minority sub-group counts equally); secondary: recall of the smallest sub-group",
           "B": "G-mean (as in AW-SMOTE's paper); F1, MCC, ROC-AUC, PR-AUC also reported"}


def all_datasets():
    rows = []
    for n in GROUP_A_DEV:
        rows.append(dict(name=n, group="A", subset="A-dev", source="PMLB"))
    for n in GROUP_A_STAGE2_HELDOUT:
        rows.append(dict(name=n, group="A", subset="A-stage2-heldout", source="PMLB"))
    for n in GROUP_A_NEW:
        rows.append(dict(name=n, group="A", subset="A-new", source="PMLB"))
    for n in OPENML_A_NEW:
        rows.append(dict(name=n, group="A", subset="A-new", source="OpenML-CC18"))
    for n in KEEL_AWSMOTE:
        rows.append(dict(name=n, group="B", subset="B-keel-awsmote", source="KEEL",
                         shares_raw_data_with=KEEL_SHARES_RAW_WITH_A.get(n, "")))
    for n in KEEL_EXTRA:
        rows.append(dict(name=n, group="B", subset="B-keel-extra", source="KEEL"))
    for n in IMBLEARN_B:
        rows.append(dict(name=n, group="B", subset="B-imblearn", source="imblearn"))
    return pd.DataFrame(rows).fillna("")


# ------------------------------------------------------------------ loaders
def _read_keel_dat(text):
    attrs, data, in_data = [], [], False
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%"):
            continue
        low = s.lower()
        if in_data:
            data.append([v.strip() for v in s.split(",")])
        elif low.startswith("@attribute"):
            rest = s[len("@attribute"):].strip()
            if "{" in rest:
                name = rest[:rest.index("{")].strip()
                attrs.append((name, "nominal"))
            else:
                name = rest.split()[0]
                attrs.append((name, "numeric"))
        elif low.startswith("@data"):
            in_data = True
    df = pd.DataFrame(data, columns=[a[0] for a in attrs])
    target = df.columns[-1]
    y = (df[target].str.strip().str.lower() == "positive").astype(int).values
    feats = df.drop(columns=target)
    nominal = [a for a, t in attrs[:-1] if t == "nominal"]
    for c in feats.columns:
        if c not in nominal:
            feats[c] = pd.to_numeric(feats[c])
    if nominal:
        feats = pd.get_dummies(feats, columns=nominal, dtype=float)
    return feats.values.astype(float), y, len(nominal)


def keel_path(name):
    return os.path.join(KEEL_DIR, f"{name}.zip")


def load_keel(name):
    with zipfile.ZipFile(keel_path(name)) as z:
        dat = [f for f in z.namelist() if f.endswith(".dat") and not f.startswith("__MACOSX")
               and not f.split("/")[-1].startswith("._")
               and "tra" not in f.split("/")[-1].lower() and "tst" not in f.split("/")[-1].lower()]
        if not dat:
            raise FileNotFoundError(f"no full .dat file inside {keel_path(name)}")
        text = z.read(dat[0]).decode("utf-8", errors="replace")
    X, y, n_nom = _read_keel_dat(text)
    return X, y


def load_imblearn(name):
    from imblearn.datasets import fetch_datasets
    b = fetch_datasets(data_home=IMBLEARN_DIR, filter_data=(name,), download_if_missing=False)[name]
    return b.data.astype(float), (b.target == 1).astype(int)


def load(name, group, source):
    """Returns X, y (1 = minority), sub (minority sub-group id, -1 for majority), info dict."""
    if group == "A":
        X, y, sub, info = bellinger_binary(name)
        return X, y, sub, info
    X, y = load_keel(name) if source == "KEEL" else load_imblearn(name)
    if y.mean() > 0.5:          # make sure 1 = minority
        y = 1 - y
    sub = np.where(y == 1, 0, -1)
    info = dict(name=name, n_maj=int((y == 0).sum()), minority_sizes=[int(y.sum())],
                n_features=X.shape[1], IR=round((y == 0).sum() / y.sum(), 2))
    return X, y, sub, info


# ------------------------------------------------------------------ write the dataset list
def write_protocol():
    os.makedirs(OUT_DIR, exist_ok=True)
    D = all_datasets()
    stats = []
    for r in D.itertuples():
        try:
            X, y, sub, info = load(r.name, r.group, r.source)
            stats.append(dict(n_samples=len(y), n_features=X.shape[1], n_minority=int(y.sum()),
                              IR=info["IR"], minority_subgroup_sizes=" / ".join(str(int(s)) for s in info["minority_sizes"])))
        except Exception as e:      # KEEL files not downloaded yet
            stats.append(dict(n_samples=np.nan, n_features=np.nan, n_minority=np.nan, IR=np.nan,
                              minority_subgroup_sizes=f"not loaded ({type(e).__name__})"))
    D = pd.concat([D, pd.DataFrame(stats)], axis=1)
    D.to_csv(os.path.join(OUT_DIR, "datasets.csv"), index=False)

    proto = dict(
        fixed_before_any_stage3_result_was_looked_at=True, date_fixed="2026-10-23",
        rca_smote="frozen: rca_smote.RCASMOTE (variant V7), no change of design or parameters",
        group_A_rule=RULE_A, group_B_rule=RULE_B, primary_metrics=PRIMARY,
        group_A_dev=GROUP_A_DEV, group_A_stage2_heldout=GROUP_A_STAGE2_HELDOUT, group_A_new=GROUP_A_NEW + list(OPENML_A_NEW),
        openml_ids=OPENML_A_NEW,
        group_B_keel_awsmote=list(KEEL_AWSMOTE), group_B_keel_extra=list(KEEL_EXTRA), group_B_imblearn=IMBLEARN_B,
        keel_sharing_raw_data_with_group_A=KEEL_SHARES_RAW_WITH_A,
        reporting=["Group A main = stage-2 held-out + new (development datasets reported separately)",
                   "Group A development datasets (5) reported separately",
                   "Group B all, and Group B without the 6 KEEL datasets that share raw data with Group A"],
        notes=NOTES,
        excluded=[dict(source=s, name=n, reason=r) for s, n, r in EXCLUDED])
    with open(os.path.join(OUT_DIR, "protocol_stage3.json"), "w") as f:
        json.dump(proto, f, indent=1)

    md = ["# Stage 3 datasets and selection rule", "",
          "Written by `python stage3_datasets.py` **before any stage-3 result was looked at**. The final list (51 datasets) "
          "was fixed on 2026-10-23 while the run on the first 26 datasets was still going; no result had been opened.", "",
          "## Group A: known minority sub-groups (Bellinger protocol, majority capped at 2000)", ""]
    md += [f"- {x}" for x in RULE_A]
    md += ["", "## Group B: standard imbalanced benchmarks (sub-groups unknown, standard metrics only)", ""]
    md += [f"- {x}" for x in RULE_B]
    md += ["", "## Primary metrics (fixed before running)", ""]
    md += [f"- Group {k}: {v}" for k, v in PRIMARY.items()]
    md += ["", "## Reporting subsets", "",
           "- **A-dev** (5): used in stage 2 to choose the RCA-SMOTE design. Reported separately.",
           "- **A-stage2-heldout** (10): used once in stage 2 to test the frozen method.",
           "- **A-new** (9: 5 PMLB + 4 OpenML-CC18): never used before.",
           "- **Group A main result** = A-stage2-heldout + A-new.",
           "- **B-keel-awsmote** (13), **B-keel-extra** (8) and **B-imblearn** (6). Six KEEL datasets share raw data with a Group A dataset "
           "(column `shares_raw_data_with`); Group B is also reported without them.", "",
           "## Included datasets", "", "| name | group | subset | source | samples | features | minority | IR | minority sub-group sizes | shares raw data with |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in D.itertuples():
        f = lambda v: "" if pd.isna(v) else (f"{v:.0f}" if isinstance(v, float) and v == int(v) else str(v))
        md.append(f"| {r.name} | {r.group} | {r.subset} | {r.source} | {f(r.n_samples)} | {f(r.n_features)} | "
                  f"{f(r.n_minority)} | {f(r.IR)} | {r.minority_subgroup_sizes} | {r.shares_raw_data_with} |")
    md += ["", "## Excluded candidates", "", "| source | dataset(s) | reason |", "|---|---|---|"]
    md += [f"| {s} | {n} | {r} |" for s, n, r in EXCLUDED]
    md += ["", "## Notes", ""] + [f"- {x}" for x in NOTES]
    md += ["", "## KEEL download links (full dataset zip; save as `data/keel/<name>.zip`)", ""]
    md += [f"- {n}: {KEEL_BASE}{d}/{n}.zip" for n, d in KEEL_ALL.items()]
    with open(os.path.join(OUT_DIR, "DATASETS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(D.to_string())


if __name__ == "__main__":
    write_protocol()
