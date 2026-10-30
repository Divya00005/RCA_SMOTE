"""Compile main paper with latexmk/pdfLaTeX or an explicitly selected Tectonic executable."""
from pathlib import Path
import argparse,shutil,subprocess
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--tectonic',help='Optional path to Tectonic; otherwise use latexmk or pdflatex/BibTeX.');a=p.parse_args();out=ROOT/'work/latex';out.mkdir(parents=True,exist_ok=True)
 if a.tectonic:subprocess.run([a.tectonic,'-X','compile','--keep-logs','--outdir',str(out),'main.tex'],cwd=ROOT/'paper',check=True)
 elif shutil.which('latexmk'):subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','-outdir='+str(out),'main.tex'],cwd=ROOT/'paper',check=True)
 else:raise SystemExit('Install latexmk with a TeX distribution, compile paper/main.tex on Overleaf, or pass --tectonic PATH.')
 shutil.copyfile(out/'main.pdf',ROOT/'paper/RCA_SMOTE_Final.pdf');print('Built paper/RCA_SMOTE_Final.pdf')
if __name__=='__main__':main()
