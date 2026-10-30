"""Audit-local data and evaluation helpers. Never writes original research evidence."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import sys,json,re,hashlib
from pathlib import Path
ROOT=PROJECT_ROOT;OUT=ROOT/'reproducibility'/'aw_baseline_reconstruction'
sys.path.insert(0,str(CODE));sys.path.insert(0,str(OUT/'sources'))
import numpy as np,pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score,accuracy_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from aw_smote_original import ADWSMOTE
from rca_smote import RCASMOTE
from imblearn.over_sampling import SMOTE
SPEC=json.loads((OUT/'DATASET_SPEC.json').read_text())
SEED=42
def load_dataset(name,encoding='ordinal'):
    text=(OUT/'sources'/'keel'/(name+'.dat')).read_text(encoding='utf-8');attrs=[];rows=[];in_data=False
    for line in text.splitlines():
        s=line.strip()
        if not s or s.startswith('%'):continue
        if in_data:rows.append([x.strip().strip("'\"") for x in s.split(',')])
        elif s.lower().startswith('@attribute'):
            rest=s[len('@attribute'):].strip();name_=rest.split()[0].strip("'\"")
            domain=[x.strip().strip("'\"") for x in rest[rest.index('{')+1:rest.index('}')].split(',')] if '{' in rest else None
            attrs.append((name_,domain))
        elif s.lower().startswith('@data'):in_data=True
    a=np.asarray(rows,dtype=object);labels=np.char.lower(a[:,-1].astype(str));assert set(labels)=={'positive','negative'},set(labels)
    y=(labels=='positive').astype(np.int64);assert 0<y.sum()<len(y)/2
    columns=[];names=[];nominal=[]
    for j,(name_,dom) in enumerate(attrs[:-1]):
        col=a[:,j]
        if any(v=='?' for v in col):raise ValueError(f'missing values in {name_}')
        if dom is None:columns.append(col.astype(float));names.append(name_)
        else:
            nominal.append(name_);assert set(col)<=set(dom),(name_,set(col)-set(dom))
            if encoding=='ordinal':columns.append(np.array([dom.index(v) for v in col],float));names.append(name_)
            elif encoding=='onehot':
                for val in dom:columns.append((col==val).astype(float));names.append(name_+'='+val)
            else:raise ValueError(encoding)
    X=np.array(columns).T;assert np.isfinite(X).all()
    info=dict(n=len(y),n_min=int(y.sum()),n_maj=int((y==0).sum()),raw_features=len(attrs)-1,model_features=X.shape[1],nominal=nominal,IR=(y==0).sum()/y.sum(),encoding=encoding,feature_names=names)
    return X,y,info
def classifiers(seed=SEED):
    return {'DT':DecisionTreeClassifier(criterion='gini',max_depth=None,max_features=None,random_state=seed),'kNN':KNeighborsClassifier(n_neighbors=5,metric='euclidean',n_jobs=1),'SVM':SVC(C=1,kernel='rbf',gamma='scale',probability=True,random_state=seed),'GBoost':GradientBoostingClassifier(n_estimators=100,random_state=seed)}
def scaler_for(kind):return StandardScaler() if kind=='standard' else MinMaxScaler() if kind=='minmax' else None
def metrics(y,pred,prob):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel();rec=tp/(tp+fn);spec=tn/(tn+fp)
    return dict(F1=f1_score(y,pred,pos_label=1,zero_division=0),G_mean=float(np.sqrt(rec*spec)),AUC=roc_auc_score(y,prob),Recall=rec,Specificity=spec,Precision=tp/(tp+fp) if tp+fp else 0.,Accuracy=accuracy_score(y,pred),F1_majority=f1_score(y,pred,pos_label=0,zero_division=0),F1_macro=f1_score(y,pred,average='macro',zero_division=0),F1_weighted=f1_score(y,pred,average='weighted',zero_division=0),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),test_n=len(y),test_min=int(y.sum()))
def index_hash(idx):return hashlib.sha256(np.asarray(idx,dtype='<i8').tobytes()).hexdigest()
