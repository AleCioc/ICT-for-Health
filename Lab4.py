import numpy as np
import scipy
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

def standardize (x):
    return (x-x.mean())/x.std()

def square_distance (x, xk):
    return np.linalg.norm(x - xk)**2

def distance (x, xk):
    return np.linalg.norm(x - xk)

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

class HardKMeans (object):

    def __init__ (self, df, nc, random_start=False):

        self.n_clusters = nc
        self.df = df
        self.y = df.iloc[:, :-1]
        self.c = df.iloc[:, -1]
        if not random_start:
            self.xks = df.groupby(df.columns[-1]).mean()
        else:
            self.xks = pd.DataFrame(np.random.randn(self.n_clusters, len(self.y.columns)), 
                                    columns = self.y.columns,
                                    index = range(1, self.n_clusters+1))

    def evaluate_performances (self, prediction, true):
        
        n_strike = float((prediction == true).sum())
        n_miss = float((prediction != true).sum())
        strike_rate = n_strike/(n_strike + n_miss)
        tp = float(((prediction >= 2) & (true >= 2)).sum())/float((true >= 2).sum())
        tn = float(((prediction == 1) & (true == 1)).sum())/float((true == 1).sum())
        fp = float(((prediction >= 2) & (true == 1)).sum())/float((true == 1).sum())
        fn = float(((prediction == 1) & (true >= 2)).sum())/float((true >= 2).sum())
        sse = 0.0
        for cluster_id in range(1, self.n_clusters+1):
            sse += self.y.loc[prediction == cluster_id].apply(square_distance, 
                                                          args=(self.xks.loc[cluster_id],), 
                                                          axis=1)
        
        return {
                "strike_rate":strike_rate, 
                "sensitivity":tp, 
                "specificity":tn, 
                "false_positive":fp, 
                "false_negative":fn,
                "sse":sse
               }
        
    def run (self, prob=False):

        N = float(len(self.df))

        self.dists = pd.DataFrame(index = self.y.index, 
                             columns = range(1, self.n_clusters+1))
        self.pis = pd.Series(1.0/float(len(range(1, self.n_clusters+1))), 
                        index = range(1, self.n_clusters+1))
        self.vars = pd.Series(1.0, index = range(1, self.n_clusters+1))
        
        iters = 0
        eps = 1e-3
        diff = 1e2
        diff_history = []
        
        while diff > eps:

            prev_xks = self.xks.copy()
            
            for cluster_id in range(1, self.n_clusters+1):
                self.dists[cluster_id] = self.y.apply(square_distance, 
                                              args=(self.xks.loc[cluster_id],), 
                                              axis=1)
            
            if prob:
                self.dists = self.dists - 2 * self.vars * np.log(self.pis)
            preds = self.dists.idxmin(axis=1)
            
            for cluster_id in range(1, self.n_clusters+1):
                wk = self.y[preds == cluster_id]
                Nk = float(preds[preds == cluster_id].size)
                self.xks.loc[cluster_id] = wk.mean()
                self.pis.loc[cluster_id] = Nk/N
                self.vars.loc[cluster_id] = \
                         (1.0/(Nk-1)/N) * np.sum(np.linalg.norm(wk - wk.mean())**2)

            diff = (self.xks-prev_xks).abs().sum().sum()
            diff_history += [diff]
            iters += 1
            
        return diff_history, preds, self.evaluate_performances(preds, self.c)

nc = 2

def plot_colds (p, edgecolor):
    
    index = np.arange(0,10,1)
    bars0 = []
    bars1 = []
    bars2 = []
    bars3 = []
    bars4 = []
    xticks = []
    for i in range(10):
        print i
        c_cold = HardKMeans(df, nc, random_start=True)
        xks_0_cold = c_cold.xks.copy()
        diff_history_cold, cluster_pred_cold, cluster_perf_cold = c_cold.run(p)
        xks_final_cold = c_cold.xks
        bars0 += [cluster_perf_cold["strike_rate"]]
        bars1 += [cluster_perf_cold["sensitivity"]]
        bars2 += [cluster_perf_cold["specificity"]]
        bars3 += [cluster_perf_cold["false_negative"]]
        bars4 += [cluster_perf_cold["false_positive"]]
        xticks += [str(i)]
        
    rects0 = plt.bar(index, bars0, 0.14, label="Strike Rate" if p else "", color="black", edgecolor=edgecolor, alpha=0.4)
    rects1 = plt.bar(index + 0.14, bars1, 0.14, label="Sensitivity" if p else "", color="blue", edgecolor=edgecolor, alpha=0.4)
    rects2 = plt.bar(index + 0.28, bars2, 0.14, label="Specificity" if p else "", color="red", edgecolor=edgecolor, alpha=0.4)
    rects3 = plt.bar(index + 0.42, bars3, 0.14, label="False negative" if p else "", color="green", edgecolor=edgecolor, alpha=0.4)
    rects4 = plt.bar(index + 0.56, bars4, 0.14, label="False positive" if p else "", color="orange", edgecolor=edgecolor, alpha=0.4)
    plt.title("Hard K-means - Different cold start comparison")
    plt.xlabel("Instance")
    plt.ylabel("Rate")
    plt.xticks(index + 0.35, xticks)
    plt.legend(loc=0, framealpha=0.7)

#fig, ax = plt.subplots(1,1, figsize=(13,6))
#plot_colds(False, "white")
#plot_colds(True, "black")

nc = 2

c_cold = HardKMeans(df, nc, random_start=True)
xks_0_cold = c_cold.xks.copy()
diff_history_cold, cluster_pred_cold, cluster_perf_cold = c_cold.run(True)
xks_final_cold = c_cold.xks

c_warm = HardKMeans(df, nc, random_start=True)
xks_0_warm = c_warm.xks.copy()
diff_history_warm, cluster_pred_warm, cluster_perf_warm = c_warm.run(True)
xks_final_warm = c_warm.xks

plt.figure(figsize=(13, 6))
plt.plot(diff_history_cold, marker="o", label="Cold start")
plt.plot(diff_history_warm, marker="+", label="Warm start")
plt.legend()

fig, axs = plt.subplots(1, nc, figsize=(13, 6))
plt.suptitle("xks gap histogram, hard K-means, warm start vs cold start")
axs_cold = (xks_final_cold - xks_0_warm).T.hist(ax=axs, bins=100, label="Cold start", alpha=0.4)
axs_warm = (xks_final_warm - xks_0_warm).T.hist(ax=axs, bins=100, label="Warm start", alpha=0.4)
plt.legend()
