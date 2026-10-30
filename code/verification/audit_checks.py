"""Independent local integrity/edge-case/metric checks; no original files written."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
import sys,json,hashlib,subprocess,platform,importlib.metadata as md,shutil,ctypes
from pathlib import Path
ROOT=PROJECT_ROOT; os.chdir(ROOT); sys.path.insert(0,str(CODE))
import numpy as np
import pandas as pd
from common import evaluate
from stage3_datasets import all_datasets,load
from rca_smote import RCASMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
V=ROOT/'results'/'supporting'/'verification'
def sha(p):
    try: return hashlib.sha256(p.read_bytes()).hexdigest()
    except PermissionError: return 'UNREADABLE_IN_SANDBOX (redundant raw OpenML cache; model input TSV hashed separately)'
tracked=subprocess.check_output(['git','ls-files'],text=True).splitlines()
inv=[]
for f in tracked:
    p=ROOT/f
    role=('source' if p.suffix=='.py' else 'figure' if p.suffix in ('.png','.pdf') else 'saved evidence' if f.startswith('results/') else 'documentation/environment')
    inv.append(dict(path=f,bytes=p.stat().st_size,sha256=sha(p),role=role))
pd.DataFrame(inv).to_csv(V/'TRACKED_INVENTORY.csv',index=False)
pd.DataFrame([dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in (ROOT/'data').rglob('*') if p.is_file()]).to_csv(V/'DATA_HASHES.csv',index=False)
manifest=dict(platform=platform.platform(),python=sys.version,executable=sys.executable,cpu_logical=os.cpu_count(),packages={d.metadata['Name']:d.version for d in md.distributions()},disk_free=shutil.disk_usage(ROOT).free,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),threads={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},cpu_physical=16,ram_total_KiB=33150328,ram_free_KiB_at_preflight=12917548)
body=(ROOT/'aw_smote_original.py').read_bytes().split(b'\n',2)[2]
manifest['upstream_body_sha256']=hashlib.sha256(body).hexdigest(); manifest['upstream_hash_matches']=manifest['upstream_body_sha256']=='72c3d438093538d0c5c3d19bc5dc456be9cdc743a7b907bfec5abd2591d15124'
(V/'ENVIRONMENT.json').write_text(json.dumps(manifest,indent=2))
rows=[]; duplicates=[]
for r in all_datasets().itertuples():
    X,y,s,info=load(r.name,r.group,r.source)
    keys=[z.tobytes() for z in X]
    splits=list(RepeatedStratifiedKFold(n_splits=2,n_repeats=5,random_state=0).split(X,np.where(y==0,-1,s)))
    cross=[len(set(keys[i] for i in tr)&set(keys[i] for i in te)) for tr,te in splits]
    rows.append(dict(dataset=r.name,group=r.group,subset=r.subset,source=r.source,n=len(y),d=X.shape[1],n_min=int(y.sum()),IR=info['IR'],sub_sizes=json.dumps([int(v) for v in info['minority_sizes']]),finite=bool(np.isfinite(X).all()),duplicate_feature_rows=len(X)-len(set(keys)),crossfold_duplicate_vectors_max=max(cross),binary_labels=sorted(np.unique(y).tolist())==[0,1],subgroup_consistent=bool(np.all((s==-1)==(y==0)))))
pd.DataFrame(rows).to_csv(V/'DATASET_INTEGRITY.csv',index=False)
# Constructive quota counterexamples through the actual final method.
r=RCASMOTE(random_state=0); X=np.arange(80).reshape(40,2).astype(float); y=np.r_[np.zeros(30,int),np.ones(10,int)]
quota={}
for sizes,total in [([100,1],2),([2,2,2],2),([2]*10,6)]:
    clusters=[np.repeat(X[-1:]+i,n,axis=0) for i,n in enumerate(sizes)]
    q=r._stage2(clusters,total,X,y)
    quota[f'{sizes}, budget={total}']=q.tolist()
# Deterministic ordinary case, singleton minority edge and duplicate-only minority.
tests={}
rng=np.random.RandomState(123); xx=np.r_[rng.normal(0,1,(40,2)),rng.normal(3,1,(12,2))]; yy=np.r_[np.zeros(40,int),np.ones(12,int)]
a,b=RCASMOTE(7).fit_resample(xx,yy); c,d=RCASMOTE(7).fit_resample(xx,yy)
tests['deterministic']=bool(np.array_equal(a,c) and np.array_equal(b,d)); tests['ordinary_counts']=np.bincount(b).tolist(); tests['ordinary_finite']=bool(np.isfinite(a).all())
for label,xx,yy in [('singleton_minority',np.arange(22).reshape(11,2),np.r_[np.zeros(10,int),1]),('duplicate_minority',np.r_[np.zeros((20,2)),np.ones((5,2))*10],np.r_[np.zeros(20,int),np.ones(5,int)])]:
    try:
        a,b=RCASMOTE(0).fit_resample(xx,yy); tests[label]=dict(counts=np.bincount(b).tolist(),finite=bool(np.isfinite(a).all()))
    except Exception as e: tests[label]=f'{type(e).__name__}: {e}'
# Independent metric check on known labels, predictions and scores, including two unequal groups.
yt=np.array([0,0,0,0,1,1,1,1,1,1]); pred=np.array([0,0,1,0,1,0,1,1,0,0]); sc=np.array([.1,.2,.8,.3,.9,.4,.7,.6,.35,.25]); sub=np.array([-1]*4+[0]*4+[1]*2)
class Fixed:
    def predict(self,X): return pred
    def decision_function(self,X): return sc
m=evaluate(Fixed(),np.zeros((10,1)),yt,sub)
tn,fp,fn,tp=confusion_matrix(yt,pred).ravel(); spec=tn/(tn+fp); rec=tp/(tp+fn); pre=tp/(tp+fp)
manual=dict(Accuracy=(tp+tn)/len(yt),Precision=pre,Recall=rec,Specificity=spec,F1=2*tp/(2*tp+fp+fn),G_mean=float(np.sqrt(spec*rec)),MCC=float((tp*tn-fp*fn)/np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))),BA=(spec+rec)/2,wBA=(spec+(3/4+0/2)/2)/2,recall_smallest_sub=0.,AUC=roc_auc_score(yt,sc),PR_AUC=average_precision_score(yt,sc))
tests['metric_max_difference']=max(abs(m[k]-v) for k,v in manual.items()); tests['quota_counterexamples']=quota
(V/'SMOKE_CHECKS.json').write_text(json.dumps(tests,indent=2))
print(json.dumps(dict(datasets=len(rows),all_finite=all(x['finite'] for x in rows),upstream_hash_matches=manifest['upstream_hash_matches'],tests=tests),indent=2))
