"""Equation-oriented diagnostic, never a replacement for the pinned author baseline.

Nondegenerate Eqs 1-12 and Algorithm 1 under explicit policies in CONFIGURATION.md.
Unspecified zero-denominator/all-noise/negative-quota cases fail rather than invent a formula.
"""

# Release path setup; algorithm logic is unchanged.
import sys as _release_sys
from pathlib import Path as _ReleasePath
for _release_parent in _ReleasePath(__file__).resolve().parents:
    if (_release_parent / "code" / "_project.py").is_file():
        _release_sys.path.insert(0, str(_release_parent / "code"))
        break
from _project import ROOT as PROJECT_ROOT, CODE

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import cdist
from replication_common import ADWSMOTE

class UnspecifiedPaperCase(ValueError):pass

class EquationAWSMOTE:
    def __init__(self,random_state=42,radius_other_rank=1,boundary_excludes_self=True,remove_noise_from_training=True):
        self.seed=random_state;self.radius_other_rank=radius_other_rank;self.boundary_excludes_self=boundary_excludes_self;self.remove_noise_from_training=remove_noise_from_training;self.last_info_={}
    def fit_resample(self,X,y):
        X=np.asarray(X,float);y=np.asarray(y);labs,counts=np.unique(y,return_counts=True)
        if len(labs)!=2 or counts[0]==counts[1]:raise UnspecifiedPaperCase('binary unequal classes required')
        minority=labs[np.argmin(counts)];im=np.flatnonzero(y==minority);N=int(counts.max()-counts.min());S=X[im];rng=np.random.RandomState(self.seed)
        k=self.radius_other_rank
        if len(S)<=k:raise UnspecifiedPaperCase('radius neighbour rank infeasible')
        dist=NearestNeighbors(n_neighbors=k+1).fit(S).kneighbors(S)[0][:,-1];eps=float(np.percentile(dist,65))
        if eps<=0:raise UnspecifiedPaperCase('epsilon=0: paper does not define fallback')
        labels=DBSCAN(eps=eps,min_samples=2).fit_predict(S)
        ids=[im[labels==j] for j in np.unique(labels) if j!=-1 and np.sum(labels==j)>1]
        if not ids:raise UnspecifiedPaperCase('all noise/no valid clusters: paper supplies no rule')
        kept=np.r_[np.flatnonzero(y!=minority),np.concatenate(ids)];D=X[kept];dy=y[kept];position={int(i):j for j,i in enumerate(kept)}
        K=5;n_query=K+int(self.boundary_excludes_self)
        if len(D)<n_query:raise UnspecifiedPaperCase('boundary neighbour rank infeasible')
        nn=NearestNeighbors(n_neighbors=n_query).fit(D);boundary=[]
        for cluster in ids:
            indices=nn.kneighbors(X[cluster],return_distance=False);b=[]
            for i,nb in zip(cluster,indices):
                if self.boundary_excludes_self:nb=nb[nb!=position[int(i)]][:K]
                b.append(int((dy[nb]!=minority).sum()))
            boundary.append(np.asarray(b,float))
        W=np.array([b.max() for b in boundary]);self.last_info_=dict(eps=eps,noise=int((labels==-1).sum()),N=N,kept_minority=sum(map(len,ids)),cluster_sizes=[len(c) for c in ids],boundary_weight=W.tolist())
        if W.sum()==0:raise UnspecifiedPaperCase('Eq 3 zero total boundary weight; uniform fallback is not specified')
        share=W/W.sum();quota=np.floor(share*N+.5).astype(int);quota[np.argmax(share)]+=N-int(quota.sum())
        self.last_info_['quota']=quota.tolist()
        if (quota<0).any():raise UnspecifiedPaperCase('Eq 4 prose residual correction produces a negative quota')
        synthetic=[]
        for ix,b,n in zip(ids,boundary,quota):
            if n==0:continue
            if b.sum()==0:raise UnspecifiedPaperCase('Eq 5 denominator zero')
            C=X[ix];wb=b/b.sum();dbar=cdist(C,C).sum(axis=1)/(len(C)-1);rho=1/(dbar+1e-8);wd=rho/rho.sum();alpha=1/(1+np.exp(-5*(wb-.2)));fused=alpha*wb+(1-alpha)*wd;p=fused/fused.sum()
            allocations=np.bincount(rng.choice(len(C),size=int(n),p=p),minlength=len(C))
            for i,n_i in enumerate(allocations):
                for _ in range(int(n_i)):
                    partner=rng.randint(len(C))
                    while partner==i:partner=rng.randint(len(C))
                    lam=rng.uniform(0,1);synthetic.append(C[i]+lam*(C[partner]-C[i]))
        syn=np.asarray(synthetic).reshape(-1,X.shape[1]);base_idx=np.sort(kept) if self.remove_noise_from_training else np.arange(len(y))
        self.last_info_['generated']=len(syn)
        return np.vstack([X[base_idx],syn]),np.r_[y[base_idx],np.repeat(minority,len(syn))]

class MaskAlignedAWSMOTE(ADWSMOTE):
    """Only align Stage-3 labels; retain author radius, quotas, RNG order and lambda range."""
    def _phase3_intracluster_sampling(self,clusters,n_j,S_maj,y,minority_label):
        if not clusters:return np.empty((0,S_maj.shape[1]))
        D=np.vstack([np.vstack(clusters),S_maj]);aligned_majority=np.r_[np.zeros(sum(map(len,clusters)),bool),np.ones(len(S_maj),bool)];knn=NearestNeighbors(n_neighbors=self.k_neighbors).fit(D);out=[]
        for C,n in zip(clusters,n_j):
            if len(C)<2 or n<=0:continue
            b=np.array([aligned_majority[knn.kneighbors([x],return_distance=False)[0]].sum() for x in C],float);wb=b/b.sum() if b.sum()>0 else np.ones(len(C))/len(C)
            # Same per-pair arithmetic and RNG sequencing as released implementation.
            dbar=np.array([np.mean([np.linalg.norm(x-z) for j,z in enumerate(C) if j!=i]) for i,x in enumerate(C)])
            rho=1/(dbar+1e-8);wd=rho/rho.sum() if rho.sum()>0 else np.ones(len(C))/len(C);alpha=1/(1+np.exp(-self.gamma*(wb-self.theta))) if self.alpha_fixed is None else np.full_like(wb,self.alpha_fixed);w=alpha*wb+(1-alpha)*wd;w=w/(w.sum()+1e-8);p=w/w.sum();ni=np.zeros(len(C),int)
            for _ in range(int(n)):ni[self.rng.choice(len(C),p=p)]+=1
            for i,n_i in enumerate(ni):
                for _ in range(int(n_i)):
                    j=self.rng.randint(len(C))
                    while j==i:j=self.rng.randint(len(C))
                    out.append(C[i]+self.rng.uniform(.2,.8)*(C[j]-C[i]))
        return np.asarray(out).reshape(-1,D.shape[1])
