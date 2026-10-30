# Reproduction instructions

Run commands from the project root. `scripts/run.py` supplies the module paths and working directory. All original selected source files are included; portability changes are recorded in the manifest. Algorithm parameters and sampler arithmetic were not tuned or repaired.

## Inputs

1. Install requirements as described in the README. The default pins are the known independent reconstruction environment, not a claim about the unavailable environment of the new aggregate measurements.
2. Run `python code/scripts/fetch_baseline.py`. For an existing downloaded file, `--from-file PATH` verifies the same upstream SHA-256.
3. Run `python scripts/run.py code/scripts/fetch_data.py` for PMLB/OpenML/imblearn inputs; follow its KEEL links and save ZIPs under `data/raw/keel/`.

## Supporting benchmark protocol

`python scripts/run.py code/exp4_all_methods.py --jobs 2 --out work/benchmark_folds`

This is the earlier supporting experiment, not a generator for the updated Table 2/3 measurements. Full experiments may be lengthy. No full benchmark was run during release assembly.

## Ablation, sensitivity and transfer

`python scripts/run.py code/exp5_ablation.py --jobs 2`

`python scripts/run.py code/exp6_cluster_audit.py --jobs 2`

These write new fold records under work/. They support the historical main-paper analyses. Do not replace the updated Table 2/3 summaries with their default full-method values.

## Table exports and statistical checks

`python analysis/export_main_tables.py` exports displayed main.tex values into work/exported_tables/. It does not estimate results.

`python analysis/recompute_statistics.py` recomputes dataset-block Wilcoxon/Holm comparisons into work/recomputed_statistics/ from saved supporting fold records. The script does not test the updated aggregate means.

## Figures

`python scripts/run.py plotting/make_fig1_corrected.py`

`python scripts/run.py plotting/make_fig2.py`

`python scripts/run.py plotting/fig_ablation_analysis.py`

`python scripts/run.py plotting/make_updated_figures.py`

These generate all eight figure types into work/generated_figures/. Canonical figures/ files remain byte-identical to the final source ZIP. Fig1 includes observed earlier-run mechanism counts. Fig2 executes sampler generation only. Fig5 uses saved ablation folds. Updated aggregate plots read the final Table 2 CSV.

## AW baseline reconstruction (main Section 8.3)

`python scripts/run.py code/verification/aw_paper_replication/prepare_sources.py`

`python scripts/run.py code/verification/aw_paper_replication/run_replication.py --primary --jobs 2`

The primary runner includes diagnostic samplers through its existing dependency chain. `--diagnostics` additionally runs preprocessing scenarios and deliberately invalid pre-CV controls. Invalid controls must never be presented as valid evaluation results. Downloaded datasets remain ignored.

The broad original analyse_replication.py, analyse_verification.py and audit helpers are retained as source evidence. They depend on additional audit outputs and original-analysis context; use the supported clean release commands above for bounded reaggregation. A complete archival audit rerun requires its input records, not just these selected main-paper assets.

## Paper build

`python scripts/build_paper.py` uses latexmk if installed. Alternatively pass `--tectonic PATH_TO_TECTONIC`, or compile paper/main.tex on Overleaf using pdfLaTeX. Build products go under work/latex/.
