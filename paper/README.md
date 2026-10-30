# Main paper only

Compile main.tex from this directory, with the supplied elsarticle.cls, elsarticle-num.bst and references.bib. The canonical figure assets are ../figures/*.pdf. Main.tex and the rebuilt PDF omit all five supplementary mentions and the supplementary heading/paragraph. No supplementary TeX, tables or PDF are included.

Tables and scientific claims remain those of the source manuscript, including its explicit earlier-run disclosures. The undefined t1 title-note marker was removed because the supplied source defines no such note. Author names, emails and declarations were retained from the supplied final source.

Overleaf: upload the complete repository and choose paper/main.tex, compiler pdfLaTeX. For local compilation use scripts/build_paper.py. The included local build uses Tectonic/XeTeX; it is not evidence of a separate pdfLaTeX build.
