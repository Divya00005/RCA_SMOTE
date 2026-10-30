"""Recompute dataset-block paired tests from the main-paper supporting run, not updated means."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
ROOT=Path(__file__).resolve().parents[1]
def holm(p):
 p=np.asarray(p);i=np.argsort(p);out=np.empty(len(p));out[i]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[i]));return out
def main(out):
 f=pd.read_csv(ROOT/'results/supporting/stage3/all_folds.csv.gz');f.method=f.method.fillna('None');rows=[]
 methods=['None','RandomOverSampler','SMOTE','ADASYN','Borderline-SMOTE','SVM-SMOTE','KMeans-SMOTE','AW-SMOTE','RCA-SMOTE','RandomUnderSampler','NearMiss-1','ClusterCentroids','TomekLinks','ENN','SMOTEENN','SMOTETomek']
 f=f[f.method.isin(methods)]
 for subset,mask in [('A-main',(f.group=='A')&(f.subset!='A-dev')),('B-all',f.group=='B')]:
  # Match the declared aggregation order exactly: folds, then classifiers.
  # Floating-point differences near ties can otherwise change signed-rank p-values.
  x=f[mask].groupby(['dataset','classifier','method'])[['wBA','recall_smallest_sub','G_mean','MCC']].mean().reset_index().groupby(['dataset','method'])[['wBA','recall_smallest_sub','G_mean','MCC']].mean()
  for metric in ['wBA','recall_smallest_sub','G_mean','MCC']:
   t=x[metric].unstack('method');family=[]
   for other in t.columns:
    if other=='RCA-SMOTE':continue
    a=t['RCA-SMOTE'];b=t[other];d=a-b
    family.append(dict(subset=subset,metric=metric,vs=other,n=len(d),difference=float(d.mean()),p=float(wilcoxon(a,b).pvalue) if np.any(abs(d)>1e-12) else 1.))
   for r,p in zip(family,holm([r['p'] for r in family])):r['p_holm']=float(p)
   rows+=family
 out.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(out/'paired_dataset_tests.csv',index=False)
 print('Recomputed',len(rows),'comparisons from earlier supporting fold records; updated means are not tested.')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'work/recomputed_statistics');a=p.parse_args();main(a.output)
