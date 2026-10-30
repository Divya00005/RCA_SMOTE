"""Figure B: ablation of RCA-SMOTE (difference from full RCA-SMOTE, 19 Group A-main datasets).

Source: rca-smote/results/stage4/ablation_all_folds.csv.gz (stage-4 ablation runs; the default configuration
reproduces stage-3 RCA-SMOTE exactly and "all four removed" reproduces AW-SMOTE exactly, see
verification/STAGE4_AUDIT.md). Aggregation: figstyle.dataset_means.
Bar = mean of the 19 dataset-level differences (equal to the reported A-main mean); dots = the 19 datasets.
No p-values are drawn; dataset-level tests are listed in DATA_VERIFICATION.md.
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
import matplotlib.pyplot as plt
from figstyle import (REPO, OUT, BLUE, ORANGE, INK, INK2, MUTED, GRAY, SURF, SINGLE, clean, save, dataset_means)

F = pd.read_csv(os.path.join(REPO, "results", "supporting", "stage4", "ablation_all_folds.csv.gz"))
DEF = "RCA-SMOTE (default)"
ROWS = [  # (saved configuration name, label in the figure)
    ("ablation: no rescue", "No sparse-point rescue"),
    ("ablation: AW boundary quota", "AW-SMOTE quota\n(boundary weights)"),
    ("ablation: no coarse clusters (eps 65)", "AW-SMOTE clustering\n(eps = 65th percentile)"),
    ("ablation: AW generation (published)", "AW-SMOTE generation\n(published code)"),
    ("ablation: all four parts removed", "All four reverted\n(= AW-SMOTE)"),
]
F = F[F.method.isin([DEF] + [r[0] for r in ROWS])]
D = dataset_means(F, ["wBA", "recall_smallest_sub"])
out = []
for cfg, _ in ROWS:
    for m in ["wBA", "recall_smallest_sub"]:
        d = (D[m].xs(cfg, level="method") - D[m].xs(DEF, level="method")) * 100
        for ds, v in d.items():
            out.append(dict(configuration=cfg, metric=m, dataset=ds, diff_pp=v))
L = pd.DataFrame(out)
L.to_csv(os.path.join(OUT, "data_fig_ablation_analysis.csv"), index=False, float_format="%.4f")

rng = np.random.RandomState(0)                     # jitter of the dots only (vertical position inside a row)
y = np.arange(len(ROWS))[::-1]
fig, axes = plt.subplots(1, 2, figsize=(SINGLE, 3.0), sharey=True, gridspec_kw=dict(wspace=0.1))
specs = [("wBA", "(a) wBA", (-6.6, 1.6)),
         ("recall_smallest_sub", "(b) Smallest-sub-group recall", (-23.0, 13.5))]
# each row: text band on top (y + 0.30), bar and dots below it (centred at y - 0.08)
for ax, (m, title, lim) in zip(axes, specs):
    for yi, (cfg, _) in zip(y, ROWS):
        v = L[(L.configuration == cfg) & (L.metric == m)].diff_pp.values
        mean = v.mean()
        col = ORANGE if cfg.endswith("all four parts removed") else MUTED
        yc = yi - 0.08
        ax.barh(yc, mean, height=0.42, color=col, alpha=0.9 if col == ORANGE else 0.55, zorder=2,
                edgecolor=SURF, linewidth=0.3)
        ax.scatter(v, yc + rng.uniform(-0.17, 0.17, len(v)), s=4, color=INK, alpha=0.55, lw=0, zorder=3)
        worse = int((v < -1e-9).sum())
        ax.text(lim[0] + 0.015 * (lim[1] - lim[0]), yi + 0.31, f"{mean:+.2f}", ha="left", va="center",
                fontsize=6.2, fontweight="bold", color=INK)
        ax.text(lim[0] + 0.20 * (lim[1] - lim[0]), yi + 0.31, f"({worse}/19 lower)", ha="left", va="center",
                fontsize=5.7, color=INK2)
    ax.axvline(0, color=BLUE, lw=1.0, zorder=4)
    ax.set_xlim(*lim)
    ax.set_title(title, loc="left", fontsize=7)
    clean(ax, "x")
fig.supxlabel("Difference from full RCA-SMOTE (percentage points)", fontsize=7, y=-0.005)
axes[0].set_yticks(y, [r[1] for r in ROWS])
axes[0].set_ylim(-0.45, len(ROWS) - 0.4)
axes[1].tick_params(axis="y", length=0)
fig.text(0.0, -0.05, "Bars: mean over the 19 datasets (bold value; in brackets: datasets where the reverted version is "
         "lower).\nDots: individual datasets (each the mean of 4 classifiers Ã— 10 folds). Blue line at 0: full RCA-SMOTE.",
         fontsize=5.8, color=INK2, va="top", ha="left", linespacing=1.4)
save(fig, "fig_ablation_analysis")
print(L.groupby(["configuration", "metric"]).diff_pp.agg(["mean", "min", "max", lambda x: (x < -1e-9).sum()]).round(3))
