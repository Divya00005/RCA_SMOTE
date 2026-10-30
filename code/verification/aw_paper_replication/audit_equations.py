
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
import sys,json,hashlib,platform,importlib.metadata as md,ctypes,contextlib,io,re
from pathlib import Path
from replication_common import *
from equation_aw_smote import EquationAWSMOTE,MaskAlignedAWSMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
records=[]
for r in SPEC:
    X,y,info=load_dataset(r['name']);xo,yo,io_=load_dataset(r['name'],'onehot')
    keys=[x.tobytes() for x in X];cross=[]
    for tr,te in StratifiedKFold(n_splits=5,shuffle=True,random_state=42).split(X,y):cross.append(len({keys[i] for i in tr}&{keys[i] for i in te}))
    records.append(dict(dataset=r['name'],cohort=r['cohort'],paper_n=r['paper_n'],actual_n=len(y),paper_d=r['paper_features'],raw_d=info['raw_features'],ordinal_d=X.shape[1],onehot_d=xo.shape[1],nominal_count=len(info['nominal']),nominal_attributes='|'.join(info['nominal']),n_majority=info['n_maj'],n_minority=info['n_min'],paper_IR=r['paper_IR'],actual_IR=info['IR'],IR_matches_round2=round(info['IR'],2)==r['paper_IR'],archive_matches_old=r['matches_existing_archive'],duplicate_feature_rows=len(X)-len(set(keys)),max_crossfold_duplicate_vectors=max(cross)))
pd.DataFrame(records).to_csv(OUT/'TABLE1_TABLE5_DATASET_AUDIT.csv',index=False)
eq=[
 (1,'p7','majority count among Nk in D','_phase2_quota_allocation:86-97; _phase3:120-134','DIFFERENCE + AMBIGUITY','Stage 2 searches original D rather than Algorithm 1 cleaned D. Stage 3 uses reordered D and original y mask: confirmed misalignment. Query self is included in k=5; paper self convention unspecified.'),
 (2,'p7','cluster representative = max wb','_phase2:89-100','MATCH conditional on Eq1','Maximum implemented; inherits neighbour-universe or mask errors.'),
 (3,'p7','Wbar = W/sum W','_phase2:101-104','MATCH + UNSPECIFIED extension','Normalization correct when denominator positive; code uses uniform if all weights zero, not specified by formula.'),
 (4,'p8','floor(Wbar*N+0.5), residual to max Wbar','_phase2:105-110','CONFIRMED DIFFERENCE','NumPy round is ties-to-even; residual goes to first max rounded quota, possibly different from max normalized weight. Both source and prose can create negative allocations in small-budget cases.'),
 (5,'p8','wb normalized inside cluster','_phase3:126-142','FORMULA MATCH, INPUT BUG','Normalization/fallback works numerically but wb uses misaligned mask and wrong D universe relative to Algorithm 1; zero-sum uniform fallback unspecified.'),
 (6,'p8','mean Euclidean distance to all other cluster rows','_phase3:144-155','MATCH for size>1','Loops exclude own row index and average Euclidean norms; singleton is skipped rather than quota redistributed as prose says.'),
 (7,'p8','rho=1/(dbar+delta)','_phase3:157-159','MATCH, delta unspecified','Code delta=1e-8; paper calls it a small constant without numeric value.'),
 (8,'p8','Wd=rho/sum rho','_phase3:161-164','MATCH','Positive finite densities normalize correctly.'),
 (9,'p9','alpha=sigmoid(gamma*(Wb-theta))','_phase3:167-183','MATCH conditional on Wb','Defaults theta=.2,gamma=5 match. alpha_fixed is an extra ablation branch, not default AW.'),
 (10,'p9','Wi=alpha*Wb+(1-alpha)*Wd','_phase3:187','MATCH conditional on inputs','Linear fusion implemented.'),
 (11,'p9','Pi=Wi/sum Wi','_phase3:188-191','MATCH mathematically','Extra epsilon normalization followed by exact renormalization cancels algebraically (possible floating error only). Seed allocation uses multinomial draws; paper does not specify deterministic vs multinomial allocation.'),
 (12,'p9','xnew=xi+lambda*(xk-xi), lambda in (0,1)','_phase3:194-200','FORMULA MATCH, DOMAIN DIFFERENCE','Code chooses any distinct cluster row uniformly (not explicit k-nearest subset) and lambda uniform(.2,.8). RandomNeighbor subset is unspecified; interpolation domain difference confirmed.')]
pd.DataFrame(eq,columns=['equation','paper_page','specification','author_location','classification','assessment']).to_csv(OUT/'EQUATION_VS_CODE.csv',index=False)
# Locations are verified against the downloaded upstream file, without local attribution headers.
locations={1:'Stage 2:84-96; Stage 3:114-133',2:'94-96',3:'98-102',4:'103-108',5:'128-138',6:'140-150',7:'151-152',8:'154-157',9:'160-173',10:'177',11:'178-190',12:'192-198'}
comparison=pd.read_csv(OUT/'EQUATION_VS_CODE.csv');comparison['author_location']=comparison.equation.map(locations);comparison.to_csv(OUT/'EQUATION_VS_CODE.csv',index=False)
# Source-order invariance diagnostic, no classifier fitting.
rng=np.random.RandomState(123);xx=np.r_[rng.normal([-2,0],.2,(30,2)),np.c_[np.arange(10),np.zeros(10)]];yy=np.r_[np.zeros(30,int),np.ones(10,int)];perm=np.r_[np.arange(30,40),np.arange(30)];probes={}
for name,cls in [('pinned',ADWSMOTE),('mask_only',MaskAlignedAWSMOTE)]:
    with contextlib.redirect_stdout(io.StringIO()):a,b=cls(random_state=42).fit_resample(xx,yy);c,d=cls(random_state=42).fit_resample(xx[perm],yy[perm]);aa,bb=cls(random_state=42).fit_resample(xx,yy)
    probes[name]=dict(deterministic=bool(np.array_equal(a,aa) and np.array_equal(b,bb)),same_synthetic_after_class_block_permutation=bool(np.array_equal(a[len(xx):],c[len(xx):])),generated=len(a)-len(xx))
probes['rounding_example']={'shares':[.25,.75],'N':2,'code_initial':np.round(np.array([.25,.75])*2).astype(int).tolist(),'paper_initial':np.floor(np.array([.25,.75])*2+.5).astype(int).tolist(),'code_final':[0,2],'paper_final':[1,1]}
probes['residual_target_example']={'weights':[4,5,6],'N':2,'initial':[1,1,1],'code_final':[0,1,1],'paper_final':[1,1,0]}
probes['radius']={'coordinates':[0,1,4,10],'author_distances':NearestNeighbors(n_neighbors=2).fit(np.array([0,1,4,10])[:,None]).kneighbors(np.array([0,1,4,10])[:,None])[0][:,-1].tolist()}
(OUT/'EQUATION_PROBES.json').write_text(json.dumps(probes,indent=2))
# Preserve exact author console log per-dataset scores and full-dataset class counts.
log=(OUT/'sources'/'authors_results.txt').read_text(encoding='utf-8');ds=None;clf=None;rr=[]
rename={'Raw':'None','ADW':'AW-SMOTE','SV_SMOTEWB':'SMOTEWB','SV_SYMPROD':'SYMPROD','SV_AMSCO':'AMSCO'}
for line in log.splitlines():
    m=re.search(r'处理数据集:\s*(\S+)',line)
    if m:ds=m.group(1)
    m=re.search(r'分类器:\s*(DT|kNN|SVM|GBoost)',line)
    if m:clf=m.group(1)
    m=re.search(r'(\S+): F1=([0-9.]+), G-mean=([0-9.]+), AUC=([0-9.]+)',line)
    if m and ds and clf:rr.append(dict(dataset=ds,classifier=clf,method=rename.get(m.group(1),m.group(1)),F1=float(m.group(2)),G_mean=float(m.group(3)),AUC=float(m.group(4))))
pd.DataFrame(rr).to_csv(OUT/'AUTHORS_LOG_DATASET_SCORES.csv',index=False)
class Mem(ctypes.Structure):_fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total','available','page_total','page_available','virtual_total','virtual_available','extended']]
m=Mem();m.length=ctypes.sizeof(m);ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
env=dict(platform=platform.platform(),python=sys.version,executable=sys.executable,logical_cores=os.cpu_count(),physical_cores_previously_verified=16,RAM_bytes=m.total,available_RAM_bytes=m.available,workers=4,packages={d.metadata['Name']:d.version for d in md.distributions()},classifiers={k:v.get_params() for k,v in classifiers().items()},CV=dict(n_splits=5,shuffle=True,random_state=42,numeric_seed_is_assumption=True),sampler_seed=42,author_source_sha256=hashlib.sha256((OUT/'sources'/'adw.py').read_bytes()).hexdigest())
(OUT/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2,default=str))
print(json.dumps(dict(datasets=len(records),all_counts_match=all(r['paper_n']==r['actual_n'] and r['paper_d']==r['raw_d'] for r in records),IR_disagreements=[r for r in records if not r['IR_matches_round2']],probes=probes,author_log_rows=len(rr),available_RAM_GiB=m.available/2**30),indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)))
