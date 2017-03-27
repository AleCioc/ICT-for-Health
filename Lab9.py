import numpy as np
import scipy
import pandas as pd

from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
#import seaborn as sns

def standardize (x):
    return (x-x.mean())/x.std()

def square_distance (x, xk):
    return np.linalg.norm(x - xk)**2

def cov_matrix (x):
    n = float(len(x))
    return 1.0/n * x.T.dot(x)

def load_data (full_classes=False):

    df = pd.read_csv("../Data/arrhythmia.csv", header=None)
    df = df.replace({"?": np.NaN}).dropna(axis=1, how="any")

    if not full_classes:
        df.ix[df.iloc[:, -1] > 1, df.columns[-1]] = 2
             
    df = df.loc[:,(df!=0).any()]
    
    df_notnorm = df.copy()
    df.iloc[:, :-1] = df.iloc[:, :-1].apply(standardize)
    return df_notnorm, df

df_notnorm, df = load_data(False)

def get_PCA (x):
                    
    Rx = cov_matrix(x)
    eigvals, U = np.linalg.eig(Rx)
    L = len(np.where(eigvals.cumsum() < eigvals.sum() * 0.9)[0])    
    U = U[:, :L]            
    z = x.dot(U)
    z = z/z.std()
    
    return pd.concat([z, df.iloc[:, -1]], axis=1)
        
def evaluate_performances (yhat, y):
    
    n_strike = float((yhat == y).sum())
    n_miss = float((yhat != y).sum())
    strike_rate = n_strike/(n_strike + n_miss)
    tp = float(((yhat >= 2) & (y >= 2)).sum())/float((y >= 2).sum())
    tn = float(((yhat == 1) & (y == 1)).sum())/float((y == 1).sum())
    fp = float(((yhat >= 2) & (y == 1)).sum())/float((y == 1).sum())
    fn = float(((yhat == 1) & (y >= 2)).sum())/float((y >= 2).sum())
    
    return {
            "strike_rate":strike_rate, 
            "sensitivity":tp, 
            "specificity":tn, 
            "false_positive":fp, 
            "false_negative":fn
           }
    
def run (self):
    pass

clf = svm.LinearSVC(C=2)
x = get_PCA(df.iloc[:, :-1])
y = df.iloc[:, -1]
clf = clf.fit(x, y)
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
scores = cross_val_score(clf, x, y, cv=cv)
print scores
yhat = cross_val_predict(clf, x, y, cv=100)
print metrics.accuracy_score(y, yhat)
print evaluate_performances(yhat, y)
