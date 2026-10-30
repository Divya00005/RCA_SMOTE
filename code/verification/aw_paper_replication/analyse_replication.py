"""Independent aggregation, prediction checks, source-log comparison and preservation audit."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

from replication_common import *
import re,hashlib,math
from scipy.stats import rankdata,wilcoxon,friedmanchisquare,chi2,f as fdist,norm
MET=['F1','G_mean','AUC','F1_majority','F1_macro','F1_weighted','Accuracy','Recall','Specificity','Precision']
def read(p):
    d=pd.read_csv(p)
    if 'method' in d:d['method']=d.method.fillna('None')
    return d
def holm(p):
    p=np.asarray(p,float);ix=np.argsort(p);out=np.empty(len(p));out[ix]=np.minimum(1,np.maximum.accumulate(p[ix]*(len(p)-np.arange(len(p)))));return out
files=list((OUT/'folds').glob('*/*.csv'));F=pd.concat([read(p) for p in files],ignore_index=True);F.to_csv(OUT/'ALL_LEGITIMATE_FOLDS.csv.gz',index=False)
primary=F[F.scenario=='primary'];keys=['scenario','dataset','fold','method','classifier'];assert not F.duplicated(keys).any()
checks=dict(primary_records=len(primary),primary_fold_files=len(list((OUT/'folds'/'primary').glob('*.csv'))),primary_complete=len(primary)==16*5*6*4,primary_failures=int((primary.fit_status!='ok').sum()),legitimate_records=len(F),duplicate_keys=int(F.duplicated(keys).sum()))
extra=F[F.scenario=='extras'];sensitivity=F[F.scenario.isin(['none','minmax','onehot'])]
checks.update(extras_records=len(extra),extras_complete=bool(len(extra)==780 and extra.fit_status.eq('ok').all()),sensitivity_records=len(sensitivity),sensitivity_complete=bool(len(sensitivity)==1440 and sensitivity.fit_status.eq('ok').all()),all_legitimate_failures=int(F.fit_status.ne('ok').sum()),identical_original_folds_across_methods_and_sensitivities=bool((F.groupby(['dataset','fold'])[['train_index_sha256','test_index_sha256']].nunique()==1).all().all()))
status=F.groupby(['scenario','method','sampler_status','fit_status']).size().reset_index(name='records');status.to_csv(OUT/'RUN_STATUS.csv',index=False)
outputs=F.drop_duplicates(['scenario','dataset','fold','method']).copy()
outputs['minority_added']=outputs.min_after-outputs.train_min;outputs['total_added']=outputs.n_after-outputs.train_n
outputs['returned_equal_class_counts']=outputs.min_after==outputs.majority_after
outputs['no_row_growth']=outputs.n_after==outputs.train_n
outputs[['scenario','dataset','fold','method','sampler_status','train_n','train_min','n_after','min_after','majority_after','minority_added','total_added','returned_equal_class_counts','no_row_growth']].to_csv(OUT/'SAMPLER_OUTPUT_COUNTS.csv',index=False)
M=F.groupby(['scenario','dataset','cohort','classifier','method'],dropna=False)[MET].mean().reset_index();M.to_csv(OUT/'DATASET_CLASSIFIER_MEANS.csv',index=False)
primaryplus=M[M.scenario.isin(['primary','extras'])];published=json.loads((OUT/'PUBLISHED_TABLES.json').read_text());t3=[]
for (cohort,clf,method),g in primaryplus.groupby(['cohort','classifier','method']):
    n_expected=13 if cohort=='main13' else 3;raw=F[F.scenario.isin(['primary','extras'])&(F.cohort==cohort)&(F.classifier==clf)&(F.method==method)];complete=len(raw)==n_expected*5 and raw.fit_status.eq('ok').all();rr=dict(cohort=cohort,classifier=clf,method=method,n_datasets=g.dataset.nunique(),n_folds=len(raw),complete_cohort=bool(complete),protocol='reconstruction_under_explicit_assumptions')
    for metric in MET:rr[metric]=g[metric].mean() if complete else np.nan;rr[metric+'_available_only']=g[metric].mean()
    t3.append(rr)
T3=pd.DataFrame(t3);T3.to_csv(OUT/'TABLE3_RECONSTRUCTION.csv',index=False)
overall=[]
for (cohort,method),g in T3.groupby(['cohort','method']):
    complete=len(g)==4 and g.complete_cohort.all()
    row=dict(cohort=cohort,method=method,n_classifiers=len(g),complete_cohort=bool(complete))
    for metric in MET:row[metric]=g[metric].mean() if complete else np.nan
    overall.append(row)
T4=pd.DataFrame(overall);T4.to_csv(OUT/'TABLE4_TABLE6_RECONSTRUCTION.csv',index=False)
comparisons=[]
for row in T3[T3.cohort=='main13'].itertuples():
    pp=[r for r in published['table3'] if r['classifier']==row.classifier and r['method']==row.method]
    for r in pp:
        for metric in ['F1','G_mean','AUC']:comparisons.append(dict(table=3,cohort='main13',method=row.method,classifier=row.classifier,metric=metric,published=r[metric],reconstructed=getattr(row,metric),difference=getattr(row,metric)-r[metric],complete_cohort=row.complete_cohort))
for row in T4.itertuples():
    table=4 if row.cohort=='main13' else 6
    for r in published['table4' if table==4 else 'table6']:
        if r['method']!=row.method:continue
        for metric in ['F1','G_mean','AUC']:comparisons.append(dict(table=table,cohort=row.cohort,method=row.method,classifier='mean_four',metric=metric,published=r[metric],reconstructed=getattr(row,metric),difference=getattr(row,metric)-r[metric]))
pd.DataFrame(comparisons).to_csv(OUT/'ORIGINAL_VS_REPRODUCED.csv',index=False)
# Partial Table 4 ranks must never be called reproduction of the six published-method ranks.
cohort=[m for m in ['AW-SMOTE','SYMPROD','SMOTEWB','AMSCO'] if m in set(T4[T4.cohort=='main13'].method)]
partial=T4[(T4.cohort=='main13')&T4.method.isin(cohort)].copy()
for metric in ['F1','G_mean','AUC']:partial[metric+'_rank_within_executed_subset']=partial[metric].rank(ascending=False)
partial['rank_scope']='PARTIAL: '+','.join(cohort);partial.to_csv(OUT/'TABLE4_PARTIAL_RANKS.csv',index=False)
sens_source=M[M.scenario.isin(['none','minmax','onehot'])];sens_keys=['scenario','cohort','method']
sens=sens_source.groupby(sens_keys)[MET].mean().reset_index()
sens=sens.merge(sens_source.groupby(sens_keys).dataset.nunique().rename('n_datasets').reset_index(),on=sens_keys)
sens['scope']='PRESPECIFIED SUBSET; not a reconstruction of all Table 1 datasets'
sens.loc[sens.cohort=='larger3','scope']='All three Table 5 datasets'
sens.to_csv(OUT/'PREPROCESSING_SENSITIVITY.csv',index=False)
# Summaries retain dataset-level units; six predeclared contrasts across three metrics form one family.
stats=[]
for c in ['main13','larger3']:
    rr=[]
    for metric in ['F1','G_mean','AUC']:
        p=primaryplus[(primaryplus.cohort==c)&primaryplus.method.isin(['AW-SMOTE','RCA-SMOTE','SMOTE'])].groupby(['dataset','method'])[metric].mean().unstack()
        for other in ['AW-SMOTE','SMOTE']:
            diff=p['RCA-SMOTE']-p[other];pv=wilcoxon(p['RCA-SMOTE'],p[other]).pvalue if (abs(diff)>1e-12).any() else 1.
            rr.append(dict(cohort=c,metric=metric,vs=other,n=len(p),mean_RCA=p['RCA-SMOTE'].mean(),mean_other=p[other].mean(),difference=diff.mean(),wins=int((diff>1e-9).sum()),ties=int((abs(diff)<=1e-9).sum()),losses=int((diff< -1e-9).sum()),p=pv))
    for r,pv in zip(rr,holm([x['p'] for x in rr])):r['p_holm_six_contrasts']=pv;stats.append(r)
pd.DataFrame(stats).to_csv(OUT/'PAIRED_DATASET_STATISTICS.csv',index=False)
# Table3 -> Table4 arithmetic on rounded published values.
arithmetic=[]
for r in published['table4']:
    for metric in ['F1','G_mean','AUC']:
        mean=np.mean([x[metric] for x in published['table3'] if x['method']==r['method']]);arithmetic.append(dict(method=r['method'],metric=metric,table3_mean=mean,table4=r[metric],difference=mean-r[metric]))
pd.DataFrame(arithmetic).to_csv(OUT/'PUBLISHED_TABLE_ARITHMETIC.csv',index=False)
# Author results.docx provides rounded per-dataset console scores, not original fold predictions.
log=read(OUT/'AUTHORS_LOG_DATASET_SCORES.csv');log.dataset=log.dataset.replace({'newthyroid2':'new-thyroid2'});log.to_csv(OUT/'AUTHORS_LOG_DATASET_SCORES_NORMALIZED.csv',index=False)
checks['author_log_records']=len(log)
logmeans=log.groupby(['classifier','method'])[['F1','G_mean','AUC']].mean().reset_index();logmeans.to_csv(OUT/'AUTHORS_LOG_AGGREGATES.csv',index=False)
lj=logmeans.merge(pd.DataFrame(published['table3']),on=['classifier','method'],suffixes=('_log','_paper'))
checks['author_log_table3_max_rounding_delta']=float(max(abs(lj[m+'_log']-lj[m+'_paper']).max() for m in ['F1','G_mean','AUC']))
dj=primaryplus[primaryplus.scenario=='primary'].merge(log,on=['dataset','classifier','method'],suffixes=('_legit','_authorlog'))
for metric in ['F1','G_mean','AUC']:dj[metric+'_difference']=dj[metric+'_legit']-dj[metric+'_authorlog']
dj.to_csv(OUT/'AUTHORS_PER_DATASET_VS_LEGITIMATE.csv',index=False)
invalidfiles=list((OUT/'invalid_controls').glob('*.csv')) if (OUT/'invalid_controls').exists() else []
if invalidfiles:
    I=pd.concat([read(p) for p in invalidfiles],ignore_index=True);I.to_csv(OUT/'INVALID_CONTROL_FOLDS.csv.gz',index=False)
    IM=I.groupby(['dataset','cohort','classifier','method'])[MET].mean().reset_index();IM.to_csv(OUT/'INVALID_CONTROL_DATASET_MEANS.csv',index=False)
    IM.groupby(['cohort','method'])[MET].mean().reset_index().to_csv(OUT/'INVALID_CONTROL_AGGREGATES.csv',index=False)
    ilog=IM.merge(log,on=['dataset','classifier','method'],suffixes=('_invalid','_authorlog'))
    for metric in ['F1','G_mean','AUC']:ilog[metric+'_difference']=ilog[metric+'_invalid']-ilog[metric+'_authorlog']
    ilog.to_csv(OUT/'AUTHORS_PER_DATASET_VS_INVALID_CONTROL.csv',index=False)
    checks['invalid_control_records']=len(I);checks['invalid_complete']=bool(len(I)==16*4*5*4 and I.fit_status.eq('ok').all());checks['invalid_control_failures']=int(I.fit_status.ne('ok').sum())
    if 'synthetic_test_count' in I:checks['invalid_tests_include_synthetic_rows']=int(I.synthetic_test_count.sum())
    distances=[]
    for name,dd,suf in [('legitimate',dj,'legit'),('invalid_control',ilog,'invalid')]:
        for method in ['None','AW-SMOTE','SMOTE','RCA-SMOTE']:
            for metric in ['F1','G_mean','AUC']:
                g=dd[dd.method==method]
                if len(g):distances.append(dict(protocol=name,method=method,metric=metric,n_cases=len(g),mean_abs_difference=float(abs(g[metric+'_'+suf]-g[metric+'_authorlog']).mean()),max_abs_difference=float(abs(g[metric+'_'+suf]-g[metric+'_authorlog']).max())))
    pd.DataFrame(distances).to_csv(OUT/'AUTHOR_LOG_PROTOCOL_COMPATIBILITY.csv',index=False)
# Dataset budgets, actual trace diagnostics, and a bound conditional on equal dataset means.
ds=read(OUT/'TABLE1_TABLE5_DATASET_AUDIT.csv');mm=ds[ds.cohort=='main13'];checks['mean_full_dataset_budget']=float((mm.n_majority-mm.n_minority).mean());checks['mean_four_fifths_budget']=checks['mean_full_dataset_budget']*.8;checks['published_table9_budget']=472.7692
trace=[]
for p in (OUT/'folds'/'primary').glob('*_metadata.json'):
    dataset,fold=p.stem.removesuffix('_metadata').rsplit('__',1)
    for r in json.loads(p.read_text()):
        t=r.get('trace',{})
        if r['method']=='AW-SMOTE':trace.append(dict(dataset=dataset,fold=int(fold),**{k:v for k,v in t.items() if not isinstance(v,list)},quota_min=min(t.get('quota',[0])),zero_quotas=sum(q==0 for q in t.get('quota',[]))))
tr=pd.DataFrame(trace);tr.to_csv(OUT/'PINNED_AW_FOLD_DIAGNOSTICS.csv',index=False)
checks['mask_affected_folds']=int((tr.stage3_boundary_points_affected>0).sum());checks['mask_points_affected']=int(tr.stage3_boundary_points_affected.sum());checks['mask_points_total']=int(tr.stage3_boundary_points_total.sum());checks['negative_raw_quota_folds']=int((tr.quota_min<0).sum())
checks['table6_implied_minimum_abalone_F1']=3*.9886-2;checks['table6_implied_minimum_abalone_Gmean']=3*.9881-2
# Independent prediction -> confusion and rank-sum ROC-AUC; preserve OOF coverage checks.
validation=[];coverage=[]
for p in (OUT/'folds'/'primary').glob('*_predictions.csv.gz'):
    a=read(p)
    for (dataset,fold,method,clf),g in a.groupby(['dataset','fold','method','classifier']):
        y=g.y_true.to_numpy();pred=g.y_pred.to_numpy();prob=g.p_minority.to_numpy();tp=int(((y==1)&(pred==1)).sum());tn=int(((y==0)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum());fn=int(((y==1)&(pred==0)).sum());n1=int(y.sum());n0=len(y)-n1;auc=(rankdata(prob)[y==1].sum()-n1*(n1+1)/2)/(n1*n0);f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.;gm=math.sqrt(tp/(tp+fn)*tn/(tn+fp));ref=primary[(primary.dataset==dataset)&(primary.fold==fold)&(primary.method==method)&(primary.classifier==clf)].iloc[0]
        validation.append(dict(dataset=dataset,fold=fold,method=method,classifier=clf,F1_delta=f1-ref.F1,G_mean_delta=gm-ref.G_mean,AUC_delta=auc-ref.AUC,probability_out_of_range=int(((prob<0)|(prob>1)).sum()),test_label_count_delta=n1-ref.test_min))
        coverage.extend([dict(dataset=dataset,method=method,classifier=clf,test_row=int(i)) for i in g.test_row])
pd.DataFrame(validation).to_csv(OUT/'PREDICTION_METRIC_VALIDATION.csv',index=False);CV=pd.DataFrame(coverage);checks['duplicate_OOF_test_rows']=int(CV.duplicated(['dataset','method','classifier','test_row']).sum());vc=CV.groupby(['dataset','method','classifier']).size();checks['OOF_coverage_matches_original_dataset_sizes']=bool(all(n==int(ds[ds.dataset==name].actual_n.iloc[0]) for (name,m,c),n in vc.items()));checks['prediction_metric_max_delta']=float(pd.DataFrame(validation)[['F1_delta','G_mean_delta','AUC_delta']].abs().max().max())
# Statistics printed in paper: check internal arithmetic, without fabricating unavailable raw ranks.
pranks={'F1':[2.6154,2.7692,3.2308,5.,2.2308,5.1538],'G_mean':[2.7692,2.8462,3.3846,5.3077,2.1538,4.5385],'AUC':[3.0769,2.3077,3.7692,5.2308,2.6154,4.]};pubchi={'F1':29.6593,'G_mean':26.4945,'AUC':21.1758};st=[]
for met,ranks in pranks.items():
    ch=12*13/(6*7)*(sum(x*x for x in ranks)-6*49/4);pc=pubchi[met];ff=12*pc/(65-pc);st.append(dict(metric=met,chi_from_rounded_Table18_ranks=ch,published_chi=pc,Iman_Davenport_from_published_chi=ff,Iman_Davenport_p=fdist.sf(ff,5,60)))
pd.DataFrame(st).to_csv(OUT/'PUBLISHED_STATISTICAL_ARITHMETIC.csv',index=False)
before=json.loads((OUT/'PRESERVATION_BASELINE.json').read_text());changed=[p for p,h in before.items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h];checks['protected_files_checked']=len(before);checks['changed_protected_files']=changed
(OUT/'ANALYSIS_CHECKS.json').write_text(json.dumps(checks,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)));assert not changed,changed
print(json.dumps(checks,indent=2,default=str))
