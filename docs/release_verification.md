# Release verification

The assembled folder contains all 40 selected original Python source files plus 8 release helpers. All 48 parse successfully. Two bounded sampler/metric tests passed. Five fixed-seed sampler outputs match the original implementation byte-for-byte. The reaggregation script reproduced all 120 checked supporting-run Wilcoxon/Holm comparisons, with maximum adjusted-p difference 0. No new classification benchmark was run.

All eight figure generators were executed into temporary output; canonical figure PDFs remain byte-identical to the final source ZIP. Seven main-paper table CSVs and TeX excerpts were exported from the final main.tex. Tables 2/3 contain the updated means only; historical ranks are labelled historical. Source preservation checks found zero changes among 93 fingerprinted original source files.

The main paper was rebuilt with Tectonic/XeTeX and has 29 pages in this build. All supplementary mentions were removed, including the standalone supplementary paragraph. The dangling undefined t1 title-note marker was removed. Scientific table values, author identities and declarations were retained. No separate pdfLaTeX compile was performed; the same LaTeX source is supplied for Overleaf. Font/float differences can change pagination compared with the 28-page source PDF.

The compiler reports a small 2.22pt horizontal overfull box in frontmatter and one float-only page. All rendered pages were reviewed; no visible clipping/overlap was observed. The encoding warning is in a compiler package comment. There are no unresolved citations or cross-references in the final log.

No Git repository, commits or pushes were created. Public files exclude temporary build output, downloaded upstream code/data, environment installations and supplementary documents. Selected historical supporting data remain because the MAIN paper uses them.

The remaining scientific provenance limit is documented throughout: exact producing code/environment/fold records for the updated aggregate means are unavailable. This folder does not claim an end-to-end reproduction of those means.
