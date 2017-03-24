import numpy as np
import scipy
import pandas as pd

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

df_notnorm, df = load_data(True)

class Classifier (object):

    def __init__ (self, df):

        self.xks = df.groupby(df.columns[-1]).mean()
        self.y = df.iloc[:, :-1]
        self.c = df.iloc[:, -1]
        
    def evaluate_performances (self, prediction, true):
        
        n_strike = float((prediction == true).sum())
        n_miss = float((prediction != true).sum())
        strike_rate = n_strike/(n_strike + n_miss)
        tp = float(((prediction >= 2) & (true >= 2)).sum())/float((true >= 2).sum())
        tn = float(((prediction == 1) & (true == 1)).sum())/float((true == 1).sum())
        fp = float(((prediction >= 2) & (true == 1)).sum())/float((true == 1).sum())
        fn = float(((prediction == 1) & (true >= 2)).sum())/float((true >= 2).sum())
        
        return {
                "strike_rate":strike_rate, 
                "sensitivity":tp, 
                "specificity":tn, 
                "false_positive":fp, 
                "false_negative":fn
               }
        
    def run (self):
        pass
    
class MinDistClassifier (Classifier):

    def run (self):
        distances = pd.DataFrame(index = self.y.index, 
                                 columns = self.xks.index)
        for class_id in self.xks.index:
            distances[class_id] = self.y.apply(square_distance, 
                                          args=(self.xks.loc[class_id],), 
                                          axis=1)
        prediction = distances.idxmin(axis=1)
        performances = self.evaluate_performances(prediction, self.c)
        
        return distances, prediction, performances

class MinDistClassifierPCA (MinDistClassifier):
    
    def __init__ (self, df, p=0.999):

        def get_PCA (x):
                            
            self.Rx = cov_matrix(x)
            eigvals, U = np.linalg.eig(self.Rx)
            L = len(np.where(eigvals.cumsum() < eigvals.sum() * p)[0])    
            U = U[:, :L]            
            z = x.dot(U)
            z = z/z.std()
            
            return pd.concat([z, df.iloc[:, -1]], axis=1)
        
        self.pca_df = get_PCA(df.iloc[:, :-1])
        self.xks = self.pca_df.groupby(df.columns[-1]).mean()    
        self.y = self.pca_df.iloc[:, :-1]
        self.c = self.pca_df.iloc[:, -1]

class BayesianClassifierPCA_1 (MinDistClassifierPCA):
    
    def run (self):
        
        self.pi = self.c.value_counts()/float(len(self.c))
        distances = pd.DataFrame(index = self.y.index,
                                 columns = self.xks.index)
        for class_id in self.c.unique():
            distances[class_id] = self.y.apply(square_distance, 
                                          args=(self.xks.loc[class_id],), 
                                          axis=1)
        distances = distances - 2 * np.log(self.pi)            
        prediction = distances.idxmin(axis=1)
        performances = self.evaluate_performances(prediction, self.c)
        
        return distances, prediction, performances

class BayesianClassifierPCA_2 (MinDistClassifierPCA):
    
    def run (self):
        
        self.pi = self.c.value_counts()/float(len(self.c))
        distances = pd.DataFrame(index = self.y.index,
                                 columns = self.c.unique())
        for class_id in self.c.unique():
            print class_id
            zk = self.pca_df.loc[self.pca_df.iloc[:, -1] == class_id]
            zk = zk.iloc[:, :-1]
            dk = zk - zk.mean()
            Rxk = cov_matrix(dk)
#            print zk.shape
#            print zk.mean().shape
#            print Rxk.min().min()
#            Rxk.where((Rxk < 1e-20) & (Rxk > 0), 1e-20, inplace=True)
#            Rxk.where((Rxk > -1e-20) & (Rxk < 0), -1e-20, inplace=True)
#            print Rxk.shape
#            print np.linalg.det(Rxk)
#            print self.pi[class_id]
            Rxki = pd.DataFrame(np.linalg.pinv(Rxk))
#            print Rxki.min().min()
#            print ""
            for i in range(len(self.y)):
                d = self.y.iloc[i, :] - zk.mean()
                
                a = np.log(np.linalg.det(Rxk))
                b = d.dot(Rxki).dot(d.T)
                c = 2 * np.log(self.pi[class_id])
                
                if np.linalg.det(Rxk) == 0.0:
                    det = 1e-200
                else:
                    det = np.linalg.det(Rxk)
                    
                distances.ix[i, class_id] =\
                            np.log(det) +\
                            d.dot(Rxki).dot(d.T) -\
                            2 * np.log(self.pi[class_id])

        prediction = distances.idxmin(axis=1)
        performances = self.evaluate_performances(prediction, self.c)
        
        return distances, prediction, performances

mdc = MinDistClassifier(df)
min_dist, min_dist_pred, min_dist_perf = mdc.run()

mdc_pca = MinDistClassifierPCA(df)
min_dist_pca, min_dist_pca_pred, min_dist_pca_perf = mdc_pca.run()

bc1 = BayesianClassifierPCA_1(df)
min_dist_bc1, bc1_pred, bc1_perf = bc1.run()

bc2 = BayesianClassifierPCA_2(df, p=0.999)
min_dist_bc2, bc2_pred, bc2_perf = bc2.run()

plt.figure(figsize=(13,6))
index = np.arange(0,4,1)
xticks = ["Minimum distance",
          "Minimum distance with PCA", 
          "Bayesian 1 with PCA", 
          "Bayesian 2 with PCA"]
bars0 = [min_dist_perf["strike_rate"], 
        min_dist_pca_perf["strike_rate"], 
        bc1_perf["strike_rate"], 
        bc2_perf["strike_rate"]]
bars1 = [min_dist_perf["sensitivity"], 
        min_dist_pca_perf["sensitivity"], 
        bc1_perf["sensitivity"], 
        bc2_perf["sensitivity"]]
bars2 = [min_dist_perf["specificity"], 
        min_dist_pca_perf["specificity"], 
        bc1_perf["specificity"], 
        bc2_perf["specificity"]]
bars3 = [min_dist_perf["false_negative"], 
        min_dist_pca_perf["false_negative"], 
        bc1_perf["false_negative"], 
        bc2_perf["false_negative"]]
bars4 = [min_dist_perf["false_positive"], 
        min_dist_pca_perf["false_positive"], 
        bc1_perf["false_positive"], 
        bc2_perf["false_positive"]]

plt.bar(index, bars0, 0.14, label="Strike Rate", color="black")
plt.bar(index + 0.14, bars1, 0.14, label="Sensitivity", color="blue")
plt.bar(index + 0.28, bars2, 0.14, label="Specificity", color="red")
plt.bar(index + 0.42, bars3, 0.14, label="False negative", color="green")
plt.bar(index + 0.56, bars4, 0.14, label="False positive", color="orange")
plt.title("Classification methods comparison")
plt.xlabel("Method")
plt.ylabel("Rate")
plt.xticks(index + 0.35, xticks)
plt.legend(loc=0, framealpha=0.7)
