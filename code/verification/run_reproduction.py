"""Audit-only launcher; original Stage 3 task implementation is unchanged."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import sys, time, json, argparse
from pathlib import Path
ROOT=PROJECT_ROOT
sys.path.insert(0, str(CODE))
os.chdir(ROOT)
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits
from exp4_all_methods import run_task
from stage3_datasets import all_datasets

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--full',action='store_true'); p.add_argument('--jobs',type=int,default=4)
    args=p.parse_args()
    out=ROOT/'work'/'verification'/'stage3_folds'; out.mkdir(exist_ok=True)
    start=time.time()
    if args.smoke:
        with threadpool_limits(limits=1):
            run_task('new_thyroid','A','PMLB','A-stage2-heldout',0,str(out))
    elif args.full:
        D=all_datasets(); D=D[(D.group=='A')&(D.subset!='A-dev')]
        tasks=[(r.name,r.group,r.source,r.subset,f) for r in D.itertuples() for f in range(10)
               if not (out/f'{r.name}__{f}.csv').exists()]
        print(f'Actual local reproduction: {len(tasks)} pending tasks, {args.jobs} workers',flush=True)
        with parallel_config(backend='loky',inner_max_num_threads=1):
            Parallel(n_jobs=args.jobs)(delayed(run_task)(*t,str(out)) for t in tasks)
    else:
        p.error('choose --smoke or --full')
    record=dict(mode='smoke' if args.smoke else 'full',elapsed_seconds=time.time()-start,workers=1 if args.smoke else args.jobs,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    (ROOT/'work'/'verification'/f'execution_{record["mode"]}.json').write_text(json.dumps(record,indent=2))
    print(json.dumps(record),flush=True)
