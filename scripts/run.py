"""Run a selected Python entry point from the portable project root."""
from pathlib import Path
import os,subprocess,sys
root=Path(__file__).resolve().parents[1]
if len(sys.argv)<2:raise SystemExit('Usage: python scripts/run.py code/exp4_all_methods.py --help')
target=(root/sys.argv[1]).resolve()
if not target.is_relative_to(root) or target.suffix!='.py' or not target.is_file():raise SystemExit('Expected an existing Python file inside the project.')
env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(root/'code'),str(root/'plotting'),str(root/'code/verification/aw_paper_replication'),env.get('PYTHONPATH','')]);env['PYTHONDONTWRITEBYTECODE']='1'
env['MPLCONFIGDIR']=str(root/'work/matplotlib')
raise SystemExit(subprocess.call([sys.executable,str(target),*sys.argv[2:]],cwd=root,env=env))
