
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
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from common import bellinger_binary
from rca_smote import RCASMOTE
X,y,s,info=bellinger_binary('vowel');tr,te=list(RepeatedStratifiedKFold(n_splits=2,n_repeats=5,random_state=0).split(X,np.where(y==0,-1,s)))[2]
xx=StandardScaler().fit_transform(X[tr]);yy=y[tr];ss=s[tr];m=RCASMOTE(2);clusters,lab=m._stage1(xx[yy==1],xx,yy,1);xr,yr=m.fit_resample(xx,yy)
syn=xr[len(xx):];attrib=ss[yy==1][NearestNeighbors(n_neighbors=1).fit(xx[yy==1]).kneighbors(syn)[1][:,0]]
out=dict(dataset='vowel',fold=2,subgroups=[dict(subgroup=i,training=int(((ss==i)&(yy==1)).sum()),dropped=int(((ss[yy==1]==i)&(lab==-1)).sum()),synthetic_nearest_attribution=int((attrib==i).sum()),clusters=sorted(np.unique(lab[ss[yy==1]==i]).tolist())) for i in range(4)],info=m.last_info_)
(ROOT/'work'/'verification'/'VOWEL_FOLD2_DIAGNOSIS.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)));print(json.dumps(out,indent=2,default=str))
