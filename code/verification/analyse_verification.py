"""Independent saved-evidence reanalysis and strictly completeness-gated rerun comparison."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import sys,json,hashlib,subprocess
from pathlib import Path
ROOT=PROJECT_ROOT;os.chdir(ROOT);sys.path.insert(0,str(CODE))
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon,friedmanchisquare,studentized_range,spearmanr,kendalltau
V=ROOT/'results'/'supporting'/'verification'
MET=['wBA','recall_smallest_sub','MCC','F1','G_mean','BA','Accuracy','Precision','Recall','Specificity','AUC','PR_AUC','min_sub_recall']
SAM=['None','RandomOverSampler','SMOTE','ADASYN','Borderline-SMOTE','SVM-SMOTE','KMeans-SMOTE','AW-SMOTE','RCA-SMOTE','RandomUnderSampler','NearMiss-1','ClusterCentroids','TomekLinks','ENN','SMOTEENN','SMOTETomek']
def read(p):
    d=pd.read_csv(p)
    if 'method' in d:d['method']=d.method.fillna('None')
    return d
def holm(vals):
    p=np.asarray(vals); idx=np.argsort(p);out=np.empty(len(p));out[idx]=np.minimum(1,np.maximum.accumulate(p[idx]*(len(p)-np.arange(len(p)))));return out
def wtest(a,b):
    a,b=np.asarray(a),np.asarray(b);d=a-b
    return dict(mean_ours=float(a.mean()),mean_other=float(b.mean()),diff=float(d.mean()),wins=int((d>1e-9).sum()),ties=int((abs(d)<=1e-9).sum()),losses=int((d< -1e-9).sum()),p=float(wilcoxon(a,b).pvalue) if (abs(d)>1e-12).any() else 1.)
F=read('results/supporting/stage3/all_folds.csv.gz');S=read('results/supporting/stage3/starvation_folds.csv.gz')
M=F.groupby(['dataset','group','subset','classifier','method'])[MET].mean().reset_index()
subsets={'A-main':lambda x:(x.group=='A')&(x.subset!='A-dev'),'A-new':lambda x:x.subset=='A-new','A-dev':lambda x:x.subset=='A-dev','B-all':lambda x:x.group=='B','B-no-overlap':lambda x:(x.group=='B')&~x.dataset.isin(json.loads(Path('results/supporting/stage3/protocol_stage3.json').read_text())['keel_sharing_raw_data_with_group_A'])}
summary=[];stat=[];fried=[]
for subset,mask in subsets.items():
    a=M[mask(M)&M.method.isin(SAM)]
    for metric in MET:
        pv=a.pivot(index=['dataset','classifier'],columns='method',values=metric)[SAM].dropna();ranks=pv.rank(axis=1,ascending=False).mean()
        for method in SAM: summary.append(dict(subset=subset,metric=metric,method=method,mean=pv[method].mean(),avg_rank=ranks[method],n_cases=len(pv)))
        for unit,pp in [('dataset_classifier',pv),('dataset',a.groupby(['dataset','method'])[metric].mean().unstack()[SAM].dropna())]:
            tests=[]
            for other in SAM:
                if other!='RCA-SMOTE':tests.append(dict(subset=subset,metric=metric,unit=unit,vs=other,n=len(pp),**wtest(pp['RCA-SMOTE'],pp[other])))
            for row,p in zip(tests,holm([z['p'] for z in tests])):row['p_holm']=p;stat.append(row)
            fr=friedmanchisquare(*[pp[m] for m in SAM]); cd=studentized_range.ppf(.95,len(SAM),np.inf)/np.sqrt(2)*np.sqrt(len(SAM)*(len(SAM)+1)/(6*len(pp)))
            fried.append(dict(subset=subset,metric=metric,unit=unit,n=len(pp),statistic=fr.statistic,p=fr.pvalue,CD=cd))
T=pd.DataFrame(summary);T.to_csv(V/'SAVED_TABLE_RECOMPUTED.csv',index=False)
W=pd.DataFrame(stat);W.to_csv(V/'STATISTICS_RECOMPUTED.csv',index=False)
pd.DataFrame(fried).to_csv(V/'FRIEDMAN_RECOMPUTED.csv',index=False)
orig=read('results/supporting/stage3/table_means_and_ranks.csv');j=T.merge(orig,on=['subset','metric','method'],suffixes=('_audit','_saved'))
checks={'saved_table_mean_max_delta':float(abs(j.mean_audit-j.mean_saved).max()),'saved_table_rank_max_delta':float(abs(j.avg_rank_audit-j.avg_rank_saved).max()),'original_rows':len(F),'original_key_duplicates':int(F.duplicated(['dataset','fold','method','classifier']).sum()),'original_unique_datasets':F.dataset.nunique()}
origw=read('results/supporting/stage3/wilcoxon_holm_rca_vs_each.csv');ow=origw[origw.classifier=='all 4 pooled'];jw=W[W.unit=='dataset_classifier'].merge(ow,on=['subset','metric','vs'],suffixes=('_audit','_saved'));checks['saved_wilcoxon_p_holm_max_delta']=float(abs(jw.p_holm_audit-jw.p_holm_saved).max())
fr=read('results/supporting/stage3/friedman.csv');jf=pd.DataFrame(fried).query("unit=='dataset_classifier'").merge(fr.query("blocks=='dataset x classifier'"),on=['subset','metric']); checks['friedman_stat_max_delta']=float(abs(jf.statistic-jf.chi2).max());checks['CD_max_delta']=float(abs(jf.CD-jf.CD_nemenyi).max())
# Per-dataset/classifier comparisons with explicit units and subgroup metadata.
A=M[subsets['A-main'](M)&M.method.isin(['RCA-SMOTE','SMOTE'])]
ds=[];clf=[]
for metric in MET:
    p=A.pivot(index=['dataset','classifier'],columns='method',values=metric)
    for name,g in p.groupby(level='dataset'):
        t=wtest(g['RCA-SMOTE'],g['SMOTE']);ds.append(dict(dataset=name,metric=metric,**{k:v for k,v in t.items() if k!='p'}))
    for name,g in p.groupby(level='classifier'):clf.append(dict(classifier=name,metric=metric,**wtest(g['RCA-SMOTE'],g['SMOTE'])))
pd.DataFrame(ds).to_csv(V/'RCA_SMOTE_PER_DATASET.csv',index=False);pd.DataFrame(clf).to_csv(V/'RCA_SMOTE_PER_CLASSIFIER.csv',index=False)
di=read(V/'DATASET_INTEGRITY.csv');cor=[]
for met in ['wBA','recall_smallest_sub']:
    dd=pd.DataFrame(ds).query('metric==@met').merge(di,on='dataset')
    dd['smallest_size']=dd.sub_sizes.map(lambda x:json.loads(x)[-1])
    for feat in ['IR','d','smallest_size','duplicate_feature_rows']:
        r=spearmanr(dd[feat],dd['diff']);cor.append(dict(metric=met,feature=feat,rho=r.statistic,p_exploratory=r.pvalue,n=len(dd)))
pd.DataFrame(cor).to_csv(V/'EXPLORATORY_METADATA_ASSOCIATIONS.csv',index=False)
# Starvation denominators (runs, not classifier-expanded rows).
star=S[S.subset!='A-dev'].groupby('method').agg(n=('fold','size'),starved=('smallest_sub_got_zero','sum'),starvation_pct=('smallest_sub_got_zero',lambda x:100*x.mean()),utilization_proxy_pct=('pct_real_minority_used','mean'),majority_area_pct=('pct_syn_in_majority_area','mean'),discarded_pct=('pct_minority_discarded','mean'),zero_quota_clusters=('zero_quota_clusters','mean'))
star.to_csv(V/'STARVATION_RECOMPUTED.csv')
# Stage 4 saved evidence, conservative dataset means and exact default/all-removed checks.
AB=read('results/supporting/stage4/ablation_all_folds.csv.gz');CL=read('results/supporting/stage4/cluster_all_folds.csv.gz');AS=read('results/supporting/stage4/ablation_starvation_folds.csv.gz');CS=read('results/supporting/stage4/cluster_starvation_folds.csv.gz')
st4=[];p4=[]
for label,df,ss in [('ablation',AB,AS),('cluster',CL,CS)]:
    t=df[df.subset!='A-dev'].groupby(['dataset','classifier','method'])[MET].mean().reset_index()
    mn=t.groupby('method')[MET].mean();sv=ss[ss.subset!='A-dev'].groupby('method').smallest_sub_got_zero.agg(n='size',starved='sum',starvation_pct=lambda x:100*x.mean())
    mn.join(sv).to_csv(V/f'STAGE4_{label.upper()}_RECOMPUTED.csv')
    if label=='ablation': comparisons=[('RCA-SMOTE (default)',[m for m in t.method.unique() if m.startswith('ablation')])]
    else:comparisons=[(base,[base+' + rescue',base+' + quota',base+' + rescue + quota']) for base in ['KMeans-SMOTE (sv)','DBSMOTE']]
    for base,others in comparisons:
        for metric in MET:
            piv=t.groupby(['dataset','method'])[metric].mean().unstack()
            rows=[dict(experiment=label,reference=base,method=o,metric=metric,n=len(piv),**wtest(piv[o],piv[base])) for o in others]
            for row,p in zip(rows,holm([z['p'] for z in rows])): row['p_holm_within_metric']=p;p4.append(row)
    if label=='cluster':
        # Alternative correction over six metrics, per base, same family as checks_2.
        for base in ['KMeans-SMOTE (sv)','DBSMOTE']:
            rows=[]
            for metric in ['wBA','recall_smallest_sub','MCC','F1','Specificity','Precision']:
                piv=t.groupby(['dataset','method'])[metric].mean().unstack();rows.append(dict(base=base,metric=metric,n=len(piv),**wtest(piv[base+' + rescue + quota'],piv[base])))
            for row,p in zip(rows,holm([z['p'] for z in rows])):row['p_holm_six_metrics']=p;st4.append(row)
pd.DataFrame(p4).to_csv(V/'STAGE4_DATASET_STATISTICS.csv',index=False);pd.DataFrame(st4).to_csv(V/'PLUGIN_DATASET_SIX_METRIC_TESTS.csv',index=False)
aa=AB.groupby(['dataset','classifier','method'])[MET].mean()
for variant,reference in [('RCA-SMOTE (default)','RCA-SMOTE'),('ablation: all four parts removed','AW-SMOTE')]:
    l=aa.xs(variant,level='method');r=M[M.method==reference].set_index(['dataset','classifier'])[MET].reindex(l.index)
    checks[variant+'_max_delta']=float(abs(l-r).max().max())
audit=read('results/supporting/stage4/audit_starvation.csv');r=spearmanr(audit.runs_smallest_zero_pct,audit.recall_smallest_sub);checks['starvation_spearman']=dict(rho=r.statistic,p=r.pvalue,n=len(audit))
# Rank-change tau recomputation (mean ranks, not ranking of global score means).
rankmet=['F1','AUC','wBA','recall_smallest_sub','Specificity','Precision'];ar={m:M[subsets['A-main'](M)&M.method.isin(SAM)].pivot(index=['dataset','classifier'],columns='method',values=m)[SAM].rank(axis=1,ascending=False).mean() for m in rankmet}
checks['ranking_taus']={f'{a}/{b}':float(kendalltau(ar[a],ar[b]).statistic) for a,b in [('F1','wBA'),('F1','recall_smallest_sub'),('AUC','recall_smallest_sub'),('recall_smallest_sub','Specificity'),('recall_smallest_sub','Precision')]}
# Result artifact metadata for every tracked CSV/GZ/JSON/TXT.
meta=[]
for f in subprocess.check_output(['git','ls-files','results'],text=True).splitlines():
    p=Path(f);row=dict(path=f,bytes=p.stat().st_size)
    if f.endswith(('.csv','.csv.gz')):
        try:
            t=read(p);row.update(rows=len(t),columns='|'.join(t.columns))
        except Exception as e: row['error']=str(e)
    meta.append(row)
pd.DataFrame(meta).to_csv(V/'RESULT_ARTIFACT_METADATA.csv',index=False)
# Actual rerun gate: original 19 datasets, every fold, 16x4 rows plus four ensembles.
files=[p for p in (V/'stage3_folds').glob('*.csv') if not p.name.endswith('_starvation.csv')]
R=pd.concat([read(p) for p in files],ignore_index=True) if files else pd.DataFrame()
expected=set(di.query("group=='A' and subset!='A-dev'").dataset)
complete=len(files)==190 and len(R)==190*68 and not R.duplicated(['dataset','fold','method','classifier']).any() and set(R.dataset)==expected
if len(R):
    RR=R[R.method.isin(SAM)];counts=RR.groupby(['dataset','method','classifier']).fold.nunique();complete=complete and len(counts)==19*16*4 and bool((counts==10).all())
    key=['dataset','fold','method','classifier'];match=R.merge(F,on=key,suffixes=('_new','_old'),validate='one_to_one')
    rows=[]
    difference_metrics=MET+['recall_largest_sub','recall_sub1','recall_sub2','recall_sub3','recall_sub4','n_train_after','n_min_after']
    for metric in difference_metrics:
        d=match[metric+'_new']-match[metric+'_old'];rows.append(dict(metric=metric,rows_compared=int(d.notna().sum()),max_abs_difference=float(d.abs().max()),rows_different_over_1e_12=int((d.abs()>1e-12).sum())))
    pd.DataFrame(rows).to_csv(V/'RERUN_FOLD_METRIC_DIFFERENCES.csv',index=False)
    differ=np.zeros(len(match),bool)
    for metric in difference_metrics:differ|=(abs(match[metric+'_new']-match[metric+'_old'])>1e-12).values
    match[differ].to_csv(V/'RERUN_DISCREPANT_ROWS.csv',index=False)
    new=RR.groupby(['dataset','classifier','method'])[MET].mean().groupby('method').mean()
    old=M[subsets['A-main'](M)&M.method.isin(SAM)].groupby('method')[MET].mean()
    comparison=[]
    for method in new.index:
        for metric in MET:comparison.append(dict(method=method,metric=metric,original=old.loc[method,metric],rerun=new.loc[method,metric],difference=new.loc[method,metric]-old.loc[method,metric],status='COMPLETE_ACTUAL_RERUN' if complete else 'PARTIAL_NOT_COMPARABLE_FULL_COHORT'))
    pd.DataFrame(comparison).to_csv(V/'MAIN_TABLE_COMPARISON.csv',index=False)
    if complete:
        new.reindex(SAM).to_csv(V/'MAIN_TABLE_REPRODUCED.csv')
        R.to_csv(V/'RERUN_ALL_FOLDS.csv.gz',index=False)
    checks['rerun_status_mismatches']=int((match.sampler_status_new!=match.sampler_status_old).sum())
    sf=list((V/'stage3_folds').glob('*_starvation.csv'))
    if sf:
        ns=pd.concat([read(p) for p in sf],ignore_index=True);ms=ns.merge(S,on=['dataset','fold','method'],suffixes=('_new','_old'),validate='one_to_one');deltas=[]
        for metric in ['syn_total','pct_real_minority_used','pct_syn_in_majority_area','syn_sub1','syn_sub2','syn_sub3','syn_sub4','train_sub1','train_sub2','train_sub3','train_sub4','pct_minority_discarded','n_clusters','zero_quota_clusters']:
            dd=ms[metric+'_new']-ms[metric+'_old'];deltas.append(dict(metric=metric,max_abs_difference=float(dd.abs().max()),different_rows=int((dd.abs()>1e-12).sum())))
        pd.DataFrame(deltas).to_csv(V/'RERUN_STARVATION_DIFFERENCES.csv',index=False)
        checks['rerun_starvation_rows']=len(ns);checks['rerun_starvation_flag_mismatches']=int((ms.smallest_sub_got_zero_new!=ms.smallest_sub_got_zero_old).sum())
checks['rerun_complete']=bool(complete);checks['rerun_fold_files']=len(files);checks['rerun_rows']=len(R)
# Check original tracked scientific evidence stayed byte-identical.
inv=read(V/'TRACKED_INVENTORY.csv');checks['modified_original_tracked_files']=[row.path for row in inv.itertuples() if hashlib.sha256(Path(row.path).read_bytes()).hexdigest()!=row.sha256]
(V/'ANALYSIS_CHECKS.json').write_text(json.dumps(checks,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
print(json.dumps(checks,indent=2,default=str))
