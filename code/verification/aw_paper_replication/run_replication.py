
# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,time,argparse,contextlib,io,warnings,logging
from pathlib import Path
from replication_common import *
from equation_aw_smote import EquationAWSMOTE,MaskAlignedAWSMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from joblib import Parallel,delayed,parallel_config
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore');logging.getLogger('smote_variants').setLevel(logging.CRITICAL)
CORE=['None','AW-SMOTE','SMOTE','RCA-SMOTE','Equation-AW','AW-mask-only']
EXTRAS=['SYMPROD','SMOTEWB','AMSCO']
def sampler(name):
    if name=='None':return None
    if name=='AW-SMOTE':return ADWSMOTE(random_state=42)
    if name=='SMOTE':return SMOTE(k_neighbors=5,random_state=42)
    if name=='RCA-SMOTE':return RCASMOTE(random_state=42)
    if name=='Equation-AW':return EquationAWSMOTE(random_state=42)
    if name=='AW-mask-only':return MaskAlignedAWSMOTE(random_state=42)
    import smote_variants as sv
    return getattr(sv,name)(random_state=42)
def resample(smp,X,y):
    if smp is None:return X.copy(),y.copy()
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        return smp.fit_resample(X,y) if hasattr(smp,'fit_resample') else smp.sample(X,y)
def describe(smp,X,y):
    if smp is None:return {}
    if hasattr(smp,'last_info_'):return smp.last_info_
    if isinstance(smp,ADWSMOTE):
        with contextlib.redirect_stdout(io.StringIO()):clusters,_=smp._phase1_cluster_denoise(X[y==1]);q=smp._phase2_quota_allocation(clusters,int((y==0).sum()-y.sum()),X,y)
        D=np.vstack([np.vstack(clusters),X[y==0]]);aligned=np.r_[np.zeros(sum(map(len,clusters)),bool),np.ones((y==0).sum(),bool)];oldmask=y!=1;ind=NearestNeighbors(n_neighbors=5).fit(D).kneighbors(np.vstack(clusters),return_distance=False);correct=aligned[ind].sum(axis=1);wrong=oldmask[ind].sum(axis=1)
        return dict(cluster_sizes=list(map(len,clusters)),quota=q.tolist(),noise=int(y.sum()-sum(map(len,clusters))),stage3_boundary_points_affected=int((correct!=wrong).sum()),stage3_boundary_points_total=len(correct),stage3_boundary_count_mean_abs_error=float(abs(correct-wrong).mean()),is_mask_only=isinstance(smp,MaskAlignedAWSMOTE))
    return {}
def fit_one(Xtr,ytr,Xte,yte,test_ids,base,synthetic_test=None):
    rows=[];predictions=[]
    for cn,clf in classifiers().items():
        t=time.time()
        try:
            clf.fit(Xtr,ytr);pred=clf.predict(Xte);ci=int(np.flatnonzero(clf.classes_==1)[0]);prob=clf.predict_proba(Xte)[:,ci];mm=metrics(yte,pred,prob);assert len(pred)==len(yte) and np.isfinite(prob).all()
            rows.append(dict(base,classifier=cn,fit_status='ok',seconds_fit=time.time()-t,**mm))
            pp=pd.DataFrame(dict(test_row=test_ids,y_true=yte,y_pred=pred,p_minority=prob))
            for k,v in base.items():
                if k in ['scenario','dataset','cohort','fold','method']:pp[k]=v
            pp['classifier']=cn
            if synthetic_test is not None:pp['test_is_synthetic']=synthetic_test
            predictions.append(pp)
        except Exception as e:rows.append(dict(base,classifier=cn,fit_status=f'{type(e).__name__}: {e}',seconds_fit=time.time()-t))
    return rows,predictions
def save_task(folder,key,rows,preds,meta):
    folder.mkdir(exist_ok=True,parents=True)
    if preds:pd.concat(preds,ignore_index=True).to_csv(folder/(key+'_predictions.csv.gz'),index=False)
    (folder/(key+'_metadata.json')).write_text(json.dumps(meta,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
    tmp=folder/(key+'.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(folder/(key+'.csv'))
def run_fold(name,fold,scenario='primary',methods=CORE):
    start=time.time();encoding='onehot' if scenario=='onehot' else 'ordinal';kind={'primary':'standard','none':'none','minmax':'minmax','onehot':'standard','extras':'standard'}[scenario];X,y,info=load_dataset(name,encoding);spec=next(x for x in SPEC if x['name']==name)
    tr,te=list(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X,y))[fold];sc=scaler_for(kind);Xtr=sc.fit_transform(X[tr]) if sc else X[tr].copy();Xte=sc.transform(X[te]) if sc else X[te].copy();ytr=y[tr];rows=[];preds=[];meta=[]
    for method in methods:
        t=time.time();smp=sampler(method);status='ok';trace={}
        try:
            xr,yr=resample(smp,Xtr,ytr);assert np.isfinite(xr).all();trace=describe(smp,Xtr,ytr)
        except Exception as e:xr=yr=None;status=f'{type(e).__name__}: {e}'
        base=dict(scenario=scenario,dataset=name,cohort=spec['cohort'],fold=fold,method=method,sampler_status=status,train_n=len(tr),train_min=int(ytr.sum()),test_n=len(te),test_min=int(y[te].sum()),n_after=len(yr) if yr is not None else None,min_after=int((yr==1).sum()) if yr is not None else None,majority_after=int((yr==0).sum()) if yr is not None else None,seconds_resample=time.time()-t,train_index_sha256=index_hash(tr),test_index_sha256=index_hash(te))
        meta.append(dict(method=method,trace=trace,status=status))
        if status=='ok':r,p=fit_one(xr,yr,Xte,y[te],te,base);rows+=r;preds+=p
        else:
            for cn in classifiers():rows.append(dict(base,classifier=cn,fit_status='not_fitted_sampler_failure'))
    save_task(OUT/'folds'/scenario,f'{name}__{fold}',rows,preds,meta);print(f'{scenario} {name} fold {fold}: {time.time()-start:.1f}s',flush=True)
def run_invalid(name,method):
    start=time.time();X,y,info=load_dataset(name);sc=StandardScaler().fit(X);Xs=sc.transform(X);smp=sampler(method);rows=[];preds=[];spec=next(x for x in SPEC if x['name']==name);scenario='INVALID_preCV_resampling'
    try:xr,yr=resample(smp,Xs,y);status='ok'
    except Exception as e:
        save_task(OUT/'invalid_controls',name+'__'+method,[dict(scenario=scenario,dataset=name,method=method,sampler_status=f'{type(e).__name__}: {e}')],[],{});return
    for fold,(tr,te) in enumerate(StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(xr,yr)):
        base=dict(scenario=scenario,dataset=name,cohort=spec['cohort'],fold=fold,method=method,sampler_status=status,train_n=len(tr),train_min=int(yr[tr].sum()),test_n=len(te),test_min=int(yr[te].sum()),n_after=len(yr),min_after=int(yr.sum()),majority_after=int((yr==0).sum()),seconds_resample=0,train_index_sha256=index_hash(tr),test_index_sha256=index_hash(te),original_n=len(y),synthetic_test_count=int((te>=len(y)).sum()))
        r,p=fit_one(xr[tr],yr[tr],xr[te],yr[te],te,base,te>=len(y));rows+=r;preds+=p
    save_task(OUT/'invalid_controls',name+'__'+method,rows,preds,dict(known_invalid=True,full_data_scaling=True,resampling_before_split=True));print(f'INVALID {name} {method}: {time.time()-start:.1f}s',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');p.add_argument('--primary',action='store_true');p.add_argument('--extras',action='store_true');p.add_argument('--diagnostics',action='store_true');p.add_argument('--jobs',type=int,default=4);a=p.parse_args();start=time.time()
    with parallel_config(backend='loky',inner_max_num_threads=1):
        if a.smoke:
            with threadpool_limits(limits=1):run_fold('pima',0)
            mode='smoke'
        elif a.primary or a.extras:
            mode='extras' if a.extras else 'primary';methods=EXTRAS if a.extras else CORE;tasks=[(r['name'],f,mode,methods) for r in SPEC if not a.extras or r['cohort']=='main13' for f in range(5) if not (OUT/'folds'/mode/f'{r["name"]}__{f}.csv').exists()]
            print(f'{mode}: {len(tasks)} tasks, {a.jobs} workers',flush=True);Parallel(n_jobs=a.jobs)(delayed(run_fold)(*t) for t in tasks)
        elif a.diagnostics:
            mode='diagnostics';sensitivity=['pima','flare-F','winequality-red-3_vs_5','abalone19','kr-vs-k-three_vs_eleven','segment0'];tasks=[(name,f,scenario,['None','AW-SMOTE','SMOTE','RCA-SMOTE']) for scenario in ['none','minmax','onehot'] for name in sensitivity for f in range(5) if not (OUT/'folds'/scenario/f'{name}__{f}.csv').exists()]
            Parallel(n_jobs=a.jobs)(delayed(run_fold)(*t) for t in tasks)
            invalid=[(r['name'],m) for r in SPEC for m in ['None','AW-SMOTE','SMOTE','RCA-SMOTE'] if not (OUT/'invalid_controls'/f'{r["name"]}__{m}.csv').exists()]
            Parallel(n_jobs=a.jobs)(delayed(run_invalid)(*t) for t in invalid)
        else:p.error('select a mode')
    (OUT/f'EXECUTION_{mode}.json').write_text(json.dumps(dict(mode=mode,seconds=time.time()-start,workers=1 if a.smoke else a.jobs,completed=True),indent=2));print(f'{mode} completed in {time.time()-start:.1f}s',flush=True)
