"""Bounded release checks: no classification benchmark or algorithm tuning."""
import os,sys,unittest
from pathlib import Path
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'code'))
import numpy as np
from common import evaluate
from rca_smote import RCASMOTE

class FixedClassifier:
    def predict(self,X):return np.array([0,0,1,0,1,1,1,0])
    def predict_proba(self,X):
        p=np.array([.1,.2,.7,.3,.9,.8,.6,.4]);return np.c_[1-p,p]

class ReleaseTests(unittest.TestCase):
    def test_subgroup_metric_and_auc_fixture(self):
        y=np.array([0,0,0,0,1,1,1,1]);s=np.array([-1,-1,-1,-1,0,0,0,1])
        m=evaluate(FixedClassifier(),np.zeros((8,2)),y,s)
        self.assertAlmostEqual(m['wBA'],.625)
        self.assertAlmostEqual(m['F1'],.75)
        self.assertAlmostEqual(m['Specificity'],.75)
        self.assertAlmostEqual(m['Recall'],.75)
        self.assertAlmostEqual(m['recall_smallest_sub'],0.)
        self.assertAlmostEqual(m['AUC'],14/16)
    def test_sampler_determinism_and_original_rows(self):
        rng=np.random.RandomState(137)
        X=np.r_[rng.normal(size=(80,4)),rng.normal(1.5,.4,size=(20,4))]
        y=np.r_[np.zeros(80,int),np.ones(20,int)]
        a,b=RCASMOTE(3).fit_resample(X,y);c,d=RCASMOTE(3).fit_resample(X,y)
        np.testing.assert_array_equal(a,c);np.testing.assert_array_equal(b,d)
        np.testing.assert_array_equal(a[:len(X)],X);np.testing.assert_array_equal(b[:len(y)],y)
        self.assertTrue(np.isfinite(a).all());self.assertEqual(int((b==0).sum()),int((b==1).sum()))

if __name__=='__main__':unittest.main()
