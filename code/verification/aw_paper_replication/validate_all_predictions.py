"""Independent confusion/rank-sum metric check for every saved evaluation."""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

from replication_common import *
from scipy.stats import rankdata

def read(p):
    # Preserve adjacent IEEE-754 probabilities; the default CSV parser can merge near-ties.
    d=pd.read_csv(p,float_precision='round_trip')
    if 'method' in d:d['method']=d.method.fillna('None')
    return d

keys=['scenario','dataset','fold','method','classifier'];rows=[]
for scope,folder in [('legitimate',OUT/'folds'),('INVALID',OUT/'invalid_controls')]:
    files=list(folder.glob('*/*.csv')) if scope=='legitimate' else list(folder.glob('*.csv'))
    reference=pd.concat([read(p) for p in files],ignore_index=True).set_index(keys)
    predictions=folder.glob('*/*_predictions.csv.gz') if scope=='legitimate' else folder.glob('*_predictions.csv.gz')
    for path in predictions:
        data=read(path)
        for key,g in data.groupby(keys):
            y=g.y_true.to_numpy();p=g.y_pred.to_numpy();s=g.p_minority.to_numpy()
            tp=int(((y==1)&(p==1)).sum());tn=int(((y==0)&(p==0)).sum());fp=int(((y==0)&(p==1)).sum());fn=int(((y==1)&(p==0)).sum())
            n1=int(y.sum());n0=len(y)-n1;assert n1>0 and n0>0 and np.isfinite(s).all() and ((s>=0)&(s<=1)).all()
            score={'F1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,'G_mean':np.sqrt(tp/(tp+fn)*tn/(tn+fp)),'AUC':(rankdata(s)[y==1].sum()-n1*(n1+1)/2)/(n1*n0)}
            r=reference.loc[key];assert len(g)==r.test_n and n1==r.test_min
            rows.append(dict(zip(keys,key),scope=scope,**{m+'_delta':float(v-r[m]) for m,v in score.items()},test_rows=len(g)))
d=pd.DataFrame(rows);d.to_csv(OUT/'ALL_PREDICTION_METRIC_VALIDATION.csv',index=False)
check={'evaluations_checked':len(d),'legitimate_evaluations':int(d.scope.eq('legitimate').sum()),'invalid_control_evaluations':int(d.scope.eq('INVALID').sum()),'max_metric_difference':float(d[['F1_delta','G_mean_delta','AUC_delta']].abs().max().max()),'all_probabilities_finite_and_in_range':True,'all_test_sizes_and_positive_counts_match':True}
(OUT/'ALL_PREDICTION_CHECKS.json').write_text(json.dumps(check,indent=2));assert check['max_metric_difference']<1e-12
print(json.dumps(check,indent=2))
