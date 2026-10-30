"""Shared style and data helpers for the three new figures (read-only use of the rca-smote results).

All values are read from the saved, verified experiment files of the rca-smote repository. Nothing is written
inside the repository. Aggregation rule used everywhere (as recommended in verification/STATISTICAL_AUDIT.md):
  1. mean over the 10 cross-validation folds for each dataset x classifier x method;
  2. mean over the 4 classifiers (SVM, kNN, DT, RF), equal weight, for each dataset x method;
  3. the 19 Group A-main datasets are the units (blocks) for every summary, rank and test.
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = str(PROJECT_ROOT / "work" / "generated_figures")                                   # claude_new_figures/
REPO = str(PROJECT_ROOT)
os.makedirs(OUT, exist_ok=True)  # the research repository (read only)

DEV = ["ecoli", "led7", "shuttle", "texture", "wine_quality_white"]   # development datasets (excluded)
CLASSIFIERS = ["SVM", "kNN", "DT", "RF"]
A_NEW = ["analcatdata_authorship", "collins", "mfeat_factors", "movement_libras", "sleep", "gesture_phase",
         "first_order_theorem_proving", "steel_plates_fault", "wall_robot_navigation"]
DISPLAY = {  # readable dataset names (documented in DATA_VERIFICATION.md)
    "analcatdata_authorship": "authorship", "collins": "collins", "dermatology": "dermatology",
    "first_order_theorem_proving": "theorem proving", "gesture_phase": "gesture phase", "letter": "letter",
    "mfeat_factors": "mfeat factors", "movement_libras": "movement libras", "new_thyroid": "new thyroid",
    "optdigits": "optdigits", "page_blocks": "page blocks", "pendigits": "pendigits", "satimage": "satimage",
    "segmentation": "segmentation", "sleep": "sleep", "steel_plates_fault": "steel plates",
    "vowel": "vowel", "wall_robot_navigation": "wall robot", "yeast": "yeast"}

# colours: one per method, same as the paper figures
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF, GRAY = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff", "#b4b2ab"
COL = {"RCA-SMOTE": BLUE, "AW-SMOTE": ORANGE, "SMOTE": AQUA}

SINGLE = 84 / 25.4      # Springer single-column width, inches (84 mm)
plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": INK2, "axes.linewidth": 0.5, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.labelcolor": INK, "text.color": INK, "figure.facecolor": SURF, "axes.facecolor": SURF,
    "savefig.facecolor": SURF, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def clean(ax, grid="x"):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=600)
    plt.close(fig)
    print("saved", name)


def dataset_means(fold_df, metrics):
    """Steps 1-2 of the aggregation rule, A-main only; checks that every cell is complete."""
    F = fold_df.copy()
    F["method"] = F["method"].fillna("None")
    F = F[~F.dataset.isin(DEV) & F.classifier.isin(CLASSIFIERS)]
    n_folds = F.groupby(["dataset", "classifier", "method"]).fold.nunique()
    assert n_folds.min() == 10 and n_folds.max() == 10, "every dataset x classifier x method needs 10 folds"
    C = F.groupby(["dataset", "classifier", "method"])[metrics].mean()
    n_clf = C.groupby(["dataset", "method"]).size()
    assert (n_clf == 4).all(), "every dataset x method needs 4 classifiers"
    D = C.groupby(["dataset", "method"]).mean()
    assert D.index.get_level_values("dataset").nunique() == 19
    return D


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[order]))
    out = np.empty(m)
    out[order] = adj
    return out
