"""Two actual Group B fold reruns, explicitly not a complete Group B reproduction."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,time
from pathlib import Path
ROOT=PROJECT_ROOT;sys.path.insert(0,str(CODE));os.chdir(ROOT)
import pandas as pd
from threadpoolctl import threadpool_limits
from exp4_all_methods import run_task
V=ROOT/'results'/'supporting'/'verification';out=V/'group_b_probe_folds';out.mkdir(exist_ok=True);start=time.time()
with threadpool_limits(limits=1):
    for name in ['pima','yeast-2_vs_8']:run_task(name,'B','KEEL','B-keel-awsmote',0,str(out))
new=pd.concat([pd.read_csv(p) for p in out.glob('*.csv')],ignore_index=True);old=pd.read_csv('results/supporting/stage3/all_folds.csv.gz');new.method=new.method.fillna('None');old.method=old.method.fillna('None')
j=new.merge(old,on=['dataset','fold','method','classifier'],suffixes=('_new','_old'),validate='one_to_one');metrics=['wBA','G_mean','MCC','F1','BA','AUC','PR_AUC','Accuracy','Precision','Recall','Specificity']
results=dict(status='PARTIAL_TWO_FOLDS_ONLY',datasets=['pima','yeast-2_vs_8'],fold=0,rows=len(j),max_metric_delta=max(float(abs(j[m+'_new']-j[m+'_old']).max()) for m in metrics),status_mismatches=int((j.sampler_status_new!=j.sampler_status_old).sum()),elapsed_seconds=time.time()-start)
(V/'GROUP_B_ACTUAL_PROBES.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
