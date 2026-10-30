# RCA-SMOTE

Code, main-paper tables and figure sources for **RCA-SMOTE: Mitigating Rare Minority Subgroup Starvation in Cluster-Based Oversampling**.

Authors listed in the supplied final manuscript: Divya Yadav, Mayankkumar Rathva, Pratipalsinh Jadeja and Sankhadeep Chatterjee.

## Contents

- `paper/`: Elsevier main-paper source, bibliography, class/style and compiled PDF. Supplementary documents are excluded.
- `figures/`: the eight exact main-paper figure PDFs, with PNG previews.
- `code/`: the 40 selected existing sources (including plotting sources under `plotting/`), with portable paths and required dependencies. Additional release helpers are documented separately.
- `results/main_tables/`: CSV and TeX exports of Tables 1-7. Tables 2 and 3 contain only the updated reported means, not old summary versions.
- `results/supporting/`: earlier-run evidence still used by the main paper's comparator rows, starvation, ablation, sensitivity, transfer and statistical analyses.
- `reproducibility/`: protocols, source manifest, environment records and the main-text AW baseline reconstruction.

## Final reported means

| Cohort | wBA | Smallest-subgroup recall | G-mean | F1 | MCC |
|---|---:|---:|---:|---:|---:|
| Group A-main, RCA-SMOTE | 0.853 | 0.674 | 0.861 | 0.759 | 0.719 |
| Group B, RCA-SMOTE | — | — | 0.693 | 0.502 | 0.457 |

These are author-supplied updated aggregates exported exactly from the final manuscript. The producing code revision and underlying updated per-fold predictions/results were not provided. Existing experiment code and statistical evidence belong to the documented supporting run; they do not independently reproduce the updated aggregates. Higher means alone do not establish significant or universal superiority over SMOTE. Table 2's mean-rank column is explicitly historical.

## Setup (Python 3.10)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install --no-deps -r requirements-stage4.txt
python code/scripts/fetch_baseline.py
```

The pinned AW-SMOTE code is fetched into `.cache/aw_baseline/`, verified by SHA-256 and ignored by Git. Dataset files are downloaded separately into `data/raw/`.

## Use the sampler

```python
import sys
sys.path.insert(0, 'code')
from rca_smote import RCASMOTE
X_res, y_res = RCASMOTE(random_state=0).fit_resample(X_scaled, y)
```

Binary numerical data are required; scale using training data only. The frozen implementation has known quota and duplicate-neighbour edge cases; see `docs/limitations.md`.

## Verify this release

```powershell
python scripts/check_release.py
python scripts/run.py tests/test_sampler_and_metrics.py
python analysis/recompute_statistics.py
```

The first command uses the standard library; sampler tests require the installed dependencies and fetched baseline. No new classification benchmark is run by these checks.

## Reproduction commands

See `docs/reproduction_instructions.md`. Generated output belongs in `work/`, which is ignored by Git. Exported final tables and canonical figures should not be overwritten by historical experiment runners.

## Paper

Read `paper/RCA_SMOTE_Final.pdf`. Compile `paper/main.tex` from the `paper/` directory; figures are referenced from `../figures/`. Upload the complete project to Overleaf, use pdfLaTeX and select `paper/main.tex`. A local Tectonic build is included; a separate pdfLaTeX build was not performed here.

## Scope and licence

This repository excludes supplementary documents and expanded supplementary presentation tables. Shared code remains where needed by main-paper claims. Project-authored code uses the existing MIT licence; third-party baseline/data retain their own terms and are not automatically redistributed. Cite the manuscript using `CITATION.cff`.
