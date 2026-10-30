"""Mechanism probes, independent confusion-matrix identities and recorded fresh predictions."""

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
import sys,json,contextlib,io
from pathlib import Path
ROOT=PROJECT_ROOT;os.chdir(ROOT);sys.path.insert(0,str(CODE))
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import confusion_matrix,roc_auc_score,average_precision_score
from stage3_datasets import all_datasets,load
from rca_smote import RCASMOTE
from rc_aw_smote import AWSMOTE
from common import make_classifiers,evaluate
from imblearn.over_sampling import SMOTE
from threadpoolctl import threadpool_limits
V=ROOT/'results'/'supporting'/'verification'; F=pd.read_csv('results/supporting/stage3/all_folds.csv.gz');F.method=F.method.fillna('None')
rows=[];qrows=[];overlaps=[];allB={}; predrows=[];libchecks=[]
with threadpool_limits(limits=1):
 for r in all_datasets().itertuples():
    X,y,s,info=load(r.name,r.group,r.source)
    if r.group=='B':allB[r.name]=(X,y)
    cv=list(RepeatedStratifiedKFold(n_splits=2,n_repeats=5,random_state=0).split(X,np.where(y==0,-1,s)))
    for fold,(tr,te) in enumerate(cv):
        npos=int(y[te].sum());nneg=len(te)-npos
        part=F[(F.dataset==r.name)&(F.fold==fold)].dropna(subset=['Recall','Specificity'])
        tp=part.Recall*npos;tn=part.Specificity*nneg;fp=nneg-tn;fn=npos-tp
        precision=np.divide(tp,tp+fp,out=np.zeros(len(tp)),where=(tp+fp)!=0)
        f1=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(tp)),where=(2*tp+fp+fn)!=0)
        den=np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn));mcc=np.divide(tp*tn-fp*fn,den,out=np.zeros(len(tp)),where=den!=0)
        for key,vals in [('Accuracy',(tp+tn)/len(te)),('Precision',precision),('F1',f1),('MCC',mcc),('BA',(part.Recall+part.Specificity)/2),('G_mean',np.sqrt(part.Recall*part.Specificity))]:
            rows.append(dict(dataset=r.name,fold=fold,metric=key,max_delta=float(np.max(abs(np.asarray(vals)-part[key])))))
        if r.group=='A':
            subrec=part[[f'recall_sub{i}' for i in range(1,len(info['minority_sizes'])+1)]]
            rows.append(dict(dataset=r.name,fold=fold,metric='wBA',max_delta=float(abs((subrec.mean(axis=1)+part.Specificity)/2-part.wBA).max())))
        featurekeys=[row.tobytes() for row in X]
        labeled=[featurekeys[i]+bytes([int(y[i])]) for i in range(len(y))]
        overlaps.append(dict(dataset=r.name,group=r.group,subset=r.subset,fold=fold,identical_labeled_vectors_across_split=len({labeled[i] for i in tr}&{labeled[i] for i in te})))
        if r.group!='A' or r.subset=='A-dev':continue
        scaler=StandardScaler().fit(X[tr]);xt=scaler.transform(X[tr]);xe=scaler.transform(X[te]);sampler=RCASMOTE(fold);xr,yr=sampler.fit_resample(xt,y[tr]);inf=sampler.last_info_;q=np.array(inf['quota']);sizes=np.array(inf['cluster_sizes'])
        qrows.append(dict(dataset=r.name,fold=fold,budget=int((y[tr]==0).sum()-(y[tr]==1).sum()),generated=len(yr)-len(tr),clusters=len(q),min_quota=int(q.min()),zero_quotas=int((q==0).sum()),negative_quotas=int((q<0).sum()),quota_less_than_size_clusters=int((q<sizes).sum()),dropped=inf['n_noise_dropped'],rescued=inf.get('n_rescued',0)))
        if r.name=='new_thyroid' and fold==0:
            for name,smp in [('SMOTE',SMOTE(random_state=fold)),('AW-SMOTE',AWSMOTE(random_state=fold)),('RCA-SMOTE',RCASMOTE(fold))]:
                xx,yy=smp.fit_resample(xt,y[tr])
                for cn,clf in make_classifiers(fold).items():
                    clf.fit(xx,yy);pred=clf.predict(xe);score=clf.decision_function(xe) if hasattr(clf,'decision_function') else clf.predict_proba(xe)[:,1]
                    for i in range(len(te)):predrows.append(dict(dataset=r.name,fold=fold,method=name,classifier=cn,source_constructed_row=int(te[i]),truth=int(y[te[i]]),subgroup=int(s[te[i]]),prediction=int(pred[i]),score=float(score[i])))
            # Verify instrumentation-off library equality, not merely matching aggregate scores.
            import smote_variants as sv
            from cluster_plugins import KMeansSMOTEPlugin,DBSMOTEPlugin,size_quota
            for name,base,wrap in [('KMeans-SMOTE',sv.kmeans_SMOTE,KMeansSMOTEPlugin),('DBSMOTE',sv.DBSMOTE,DBSMOTEPlugin)]:
                with contextlib.redirect_stderr(io.StringIO()):
                    aa,bb=base(random_state=0).sample(xt,y[tr]);cc,dd=wrap(random_state=0).sample(xt,y[tr])
                libchecks.append(dict(method=name,array_equal=bool(np.array_equal(aa,cc) and np.array_equal(bb,dd))))
pd.DataFrame(rows).to_csv(V/'CONFUSION_IDENTITIES.csv',index=False)
pd.DataFrame(qrows).to_csv(V/'TARGET_QUOTA_PROBES.csv',index=False)
pd.DataFrame(overlaps).to_csv(V/'DUPLICATE_SPLIT_DIAGNOSTICS.csv',index=False)
pd.DataFrame(predrows).to_csv(V/'FRESH_REPRESENTATIVE_PREDICTIONS.csv',index=False)
pairs=[]
for i,(name,(X,y)) in enumerate(allB.items()):
 for name2,(XX,yy) in list(allB.items())[i+1:]:
    if X.shape==XX.shape and np.array_equal(X,XX):pairs.append(dict(first=name,second=name2,features_identical=True,labels_identical=bool(np.array_equal(y,yy))))
results=dict(metric_identity_max_delta=float(pd.DataFrame(rows).max_delta.max()),target_folds=len(qrows),target_any_zero_quota=any(r['zero_quotas'] for r in qrows),target_any_negative_quota=any(r['negative_quotas'] for r in qrows),target_budget_mismatches=sum(r['budget']!=r['generated'] for r in qrows),target_clusters_quota_below_size=sum(r['quota_less_than_size_clusters'] for r in qrows),instrumented_baseline_checks=libchecks,identical_B_datasets=pairs,plugin_infeasible_quota=size_quota([2,2,2],2).tolist())
(V/'FOCUSED_PROBES.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
