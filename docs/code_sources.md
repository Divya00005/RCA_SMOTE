# Code sources and supported entry points

All 40 selected existing Python sources are included. Additional release helpers provide portable paths, clean table exports, a safe runner, release checks and bounded tests.

| Source group | Files | Role |
|---|---:|---|
| code root | 19 | Frozen sampler/dependencies, experiments, development/mechanism and original analysis sources |
| code/scripts | 2 | Pinned baseline and dataset acquisition |
| code/verification | 8 | Main-paper reproduction, metrics, independence and edge-case checks |
| code/verification/aw_paper_replication | 7 | Main Section 8.3 reconstruction and validation |
| plotting original sources | 4 | Corrected methodology, updated aggregate figures, ablation and shared style |

The generated `plotting/make_fig2.py` extracts the Figure 2 block from `code/make_paper_figures.py`; it is an additional helper, not one of the original 40.

## Ready-to-use release commands

- `python scripts/check_release.py`: standard-library release asset/table/syntax/hash validation.
- `python scripts/run.py tests/test_sampler_and_metrics.py`: bounded sampler/metric fixtures after baseline fetching and dependency installation.
- `python analysis/export_main_tables.py`: exact displayed paper-table exports into work/.
- `python analysis/recompute_statistics.py`: saved-supporting-run dataset-level tests with the original folds-then-classifiers aggregation order and 15-comparison Holm families.
- Four plotting commands listed in reproduction_instructions.md: eight generated figures under work/.
- `python scripts/build_paper.py`: local main-paper compilation when a TeX compiler is available.

## Historical source retained for completeness

The original analysis and audit files can read additional historical verification inputs or emit superseded presentation tables. They are included because parts support the main paper and because imported dependencies must remain available. Their presence does not mean every original archival workflow is a standalone release command. The clean entry points above avoid requiring supplementary documents or misleadingly replacing new aggregate means with earlier outputs.

`supplementary_checks.py` includes the main-paper duplicate/source-family sensitivity and Figure 2 checks. Its filename is retained to preserve provenance. No supplementary document is included.

`mechanism_final.py` imports the dataset constant in `exp2_benchmarks.py`; keep both even if the older Experiment-2 result tables are excluded.

The sampler and metric arithmetic were preserved. Portability changes affect resource/output locations; the baseline downloader verifies the same upstream commit/hash. The AW preparation helper now uses the supplied dataset specification and hashes instead of relying on an old Git checkout or extracted publisher-PDF pages.
