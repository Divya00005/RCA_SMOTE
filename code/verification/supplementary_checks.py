"""Additional saved-evidence sensitivities and fresh score/figure checks."""

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
from pathlib import Path
import sys,json
ROOT=PROJECT_ROOT;os.chdir(ROOT);sys.path.insert(0,str(CODE));V=ROOT/'results'/'supporting'/'verification'
import numpy as np,pandas as pd
from scipy.stats import wilcoxon,studentized_range
from PIL import Image
from sklearn.preprocessing import StandardScaler
from exp1_synthetic import draw,synth_per_sub
from rc_aw_smote import AWSMOTE
from rca_smote import RCASMOTE
def read(p):
    t=pd.read_csv(p)
    if 'method' in t:t['method']=t.method.fillna('None')
    return t
def holm(p):
    p=np.asarray(p);idx=np.argsort(p);out=np.empty(len(p));out[idx]=np.minimum(1,np.maximum.accumulate(p[idx]*(len(p)-np.arange(len(p)))));return out
def wt(a,b):return float(wilcoxon(a,b).pvalue) if (abs(a-b)>1e-12).any() else 1.
checks={};pred=read(V/'FRESH_REPRESENTATIVE_PREDICTIONS.csv');orig=read('results/supporting/stage3/all_folds.csv.gz');sr=[]
for (ds,fold,m,c),g in pred.groupby(['dataset','fold','method','classifier']):
    y=g.truth.values;score=g.score.values;pv=score[y==1];nv=score[y==0]
    auc=((pv[:,None]>nv).sum()+.5*(pv[:,None]==nv).sum())/(len(pv)*len(nv))
    # Tie-aware threshold integration of precision against recall increments.
    ap=0;previous=0
    for threshold in np.sort(np.unique(score))[::-1]:
        chosen=score>=threshold;tp=int(y[chosen].sum());rec=tp/y.sum();ap+=(rec-previous)*tp/chosen.sum();previous=rec
    ref=orig[(orig.dataset==ds)&(orig.fold==fold)&(orig.method==m)&(orig.classifier==c)].iloc[0]
    sr.append(dict(dataset=ds,fold=fold,method=m,classifier=c,AUC_manual=auc,AP_manual=ap,AUC_delta=auc-ref.AUC,AP_delta=ap-ref.PR_AUC))
pd.DataFrame(sr).to_csv(V/'FRESH_SCORE_METRIC_CHECKS.csv',index=False);checks['fresh_AUC_AP_max_delta']=float(pd.DataFrame(sr)[['AUC_delta','AP_delta']].abs().max().max())
# Fixed demonstration sampling only; no original figure scripts executed.
rng=np.random.RandomState(0);X,y,s=draw(rng,'train');X=StandardScaler().fit_transform(X);fig2={}
for name,smp in [('AW-SMOTE',AWSMOTE(0)),('RCA-SMOTE',RCASMOTE(0))]:
    xr,yr=smp.fit_resample(X,y);fig2[name]={str(k):int(v) for k,v in synth_per_sub(X,y,s,xr,yr).items()}
(V/'FIGURE2_CHECK.json').write_text(json.dumps(fig2,indent=2))
images=[]
for p in Path('paper_figures').glob('*.png'):
    with Image.open(p) as im:images.append(dict(path=str(p),width=im.width,height=im.height,dpi=str(im.info.get('dpi'))))
pd.DataFrame(images).to_csv(V/'FIGURE_METADATA.csv',index=False)
# Check original Stage 4 p-values under original pooled unit, not just conservative reanalysis.
AB=read('results/supporting/stage4/ablation_all_folds.csv.gz');CL=read('results/supporting/stage4/cluster_all_folds.csv.gz')
for name,df,path in [('ablation',AB,'results/supporting/stage4/ablation_sensitivity_tests.csv'),('plugin',CL,'results/supporting/stage4/plugin_tests.csv')]:
    saved=read(path);rows=[]
    for key,g in saved.groupby((['subset','metric','kind'] if name=='ablation' else ['base','metric'])):
        if name=='ablation':
            subset,metric,kind=key;sel=df[df.subset.eq('A-dev') if subset=='A-dev' else df.subset.ne('A-dev')];ref='RCA-SMOTE (default)'
        else:
            ref,metric=key;sel=df[df.subset!='A-dev']
        p=sel.groupby(['dataset','classifier','method'])[metric].mean().unstack();tests=[wt(p[r.method],p[ref]) for r in g.itertuples()];adj=holm(tests)
        for r,a,b in zip(g.itertuples(),tests,adj):rows.append(dict(method=r.method,metric=metric,p_error=abs(a-r.p),holm_error=abs(b-r.p_holm)))
    t=pd.DataFrame(rows);t.to_csv(V/f'STAGE4_{name.upper()}_ORIGINAL_TEST_CHECK.csv',index=False);checks[name+'_saved_p_max_delta']=float(t.p_error.max());checks[name+'_saved_holm_max_delta']=float(t.holm_error.max())
# Family and no-identical-feature-dataset sensitivity, only from saved classifier/fold means.
M=orig[orig.method.isin(['None','RandomOverSampler','SMOTE','ADASYN','Borderline-SMOTE','SVM-SMOTE','KMeans-SMOTE','AW-SMOTE','RCA-SMOTE','RandomUnderSampler','NearMiss-1','ClusterCentroids','TomekLinks','ENN','SMOTEENN','SMOTETomek'])].groupby(['dataset','group','subset','method'])[['wBA','recall_smallest_sub','G_mean','MCC']].mean().reset_index()
di=read(V/'DATASET_INTEGRITY.csv');bad=set(di[di.duplicate_feature_rows>0].dataset)
def family(n):
    for f in ['ecoli','yeast-','glass','abalone','winequality-red','new-thyroid']:
        if n.startswith(f):return f
    return n
rows=[]
for label,d,unit in [('B-source-family',M[M.group=='B'].copy(),'family'),('A-main-no-feature-duplicates',M[(M.group=='A')&(M.subset!='A-dev')&~M.dataset.isin(bad)].copy(),'dataset')]:
    d['unit']=d.dataset.map(family) if unit=='family' else d.dataset
    for metric in (['G_mean','MCC'] if label.startswith('B') else ['wBA','recall_smallest_sub']):
        p=d.groupby(['unit','method'])[metric].mean().unstack();rr=[]
        for other in p.columns:
            if other=='RCA-SMOTE':continue
            delta=p['RCA-SMOTE']-p[other];rr.append(dict(setting=label,metric=metric,vs=other,n=len(p),mean_RCA=p['RCA-SMOTE'].mean(),mean_other=p[other].mean(),diff=delta.mean(),wins=int((delta>1e-9).sum()),losses=int((delta< -1e-9).sum()),p=wt(p['RCA-SMOTE'],p[other])))
        for r,pv in zip(rr,holm([x['p'] for x in rr])):r['p_holm']=pv;rows.append(r)
pd.DataFrame(rows).to_csv(V/'FAMILY_AND_DUPLICATE_SENSITIVITY.csv',index=False)
checks['source_family_mapping_B']={n:family(n) for n in M[M.group=='B'].dataset.unique()};checks['feature_duplicate_exclusion_datasets']=sorted(bad)
(V/'SUPPLEMENTARY_CHECKS.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
# Replace only the descriptive cache marker left by the initial hash audit.
h=read(V/'DATA_HASHES.csv');h['sha256']=h.sha256.replace('UNREADABLE_IN_SANDBOX (cache only; escalated preflight completed hashing)','UNREADABLE_IN_SANDBOX (redundant raw OpenML cache; model input TSV hashed separately)');h.to_csv(V/'DATA_HASHES.csv',index=False)
