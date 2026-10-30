
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
import sys,json
from pathlib import Path
ROOT=PROJECT_ROOT;sys.path.insert(0,str(CODE));os.chdir(ROOT)
import numpy as np
from rca_smote import RCASMOTE
from cluster_plugins import DBSMOTEPlugin
X=np.vstack([np.full((26,2),-1000.),np.repeat(np.c_[np.arange(10)*10.,np.zeros(10)],2,axis=0)])
y=np.r_[np.zeros(26,int),np.ones(20,int)]
s=RCASMOTE(0);xr,yr=s.fit_resample(X,y)
out={'near_balanced_10_duplicate_pairs':dict(before=np.bincount(y).tolist(),after=np.bincount(yr).tolist(),quota=s.last_info_['quota'],requested=6,generated=len(yr)-len(y))}
X=np.r_[np.random.RandomState(7).normal(0,.01,(20,2)),[[10.,10.],[10.01,10.01],[0.,0.]]];y=np.r_[np.zeros(20,int),np.ones(3,int)]
s=DBSMOTEPlugin(rescue=True,random_state=0)
# Fixed known DBSCAN partition isolates rescue handling from DBSCAN's retry policy.
s.execute_dbscan=lambda _:np.array([0,0,-1])
try:
    s.sample(X,y);out['empty_DBSMOTE_rescue']='did not raise'
except Exception as e:out['empty_DBSMOTE_rescue']=f'{type(e).__name__}: {e}'
(ROOT/'verification'/'PUBLIC_EDGE_CASES.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
