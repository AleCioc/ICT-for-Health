import numpy as np
import pandas as pd

#import matplotlib.pyplot as plt
#import seaborn as sns

def standardize (x):
    return (x-x.mean())/x.std()
def square_distance (x, xk):
    return np.linalg.norm(x - xk)**2

df = pd.read_csv("../Data/arrhythmia.data", header=None)
df = df.replace({"?": np.NaN}).dropna(axis=1, how="any")
df.ix[df.iloc[:, -1] > 1, df.columns[-1]] = 2
df = df.loc[:,(df!=0).any(axis=0)]

class_ids = df.iloc[:, -1]
y = df.iloc[:, :-1].apply(standardize)
#
#y1 = y[class_ids == 1]
#y2 = y[class_ids == 2]
#x1 = y1.mean()
#x2 = y2.mean()
#dist1 = y.apply(square_distance, args=(x1,), axis=1)
#dist2 = y.apply(square_distance, args=(x2,), axis=1)
#
#xmeans = np.stack([x1.values, x2.values])
#eny = np.diag(y.dot(y.T))
#enx = np.diag(xmeans.dot(xmeans.T))
#dotprod = y.dot(xmeans.T)
#U, V = np.meshgrid(enx, eny)
#distances_visintin = U + V - 2*dotprod
#distances_visintin.columns = [1,2]
#

df.iloc[:, :-1] = df.iloc[:, :-1].apply(standardize)
xks = df.groupby(279).mean()
distances = pd.DataFrame(index = df.index, columns = xks.index)
for class_id in xks.index:
    distances[class_id] = df.iloc[:, :-1].apply(square_distance, 
                                                args=(xks.loc[class_id],), 
                                                axis=1)
    
prediction = distances.idxmin(axis=1)
n_strike = float((prediction == class_ids).sum())
n_miss = float((prediction != class_ids).sum())
strike_rate = n_strike/(n_strike + n_miss)

n_true_positive = float(((prediction == 2) & (class_ids == 2)).sum())
n_true_negative = float(((prediction == 1) & (class_ids == 1)).sum())
n_false_positive = float(((prediction == 2) & (class_ids == 1)).sum())
n_false_negative = float(((prediction == 1) & (class_ids == 2)).sum())
n_positive = float((class_ids == 2).sum())
n_negative = float((class_ids == 1).sum())

true_positive_rate = n_true_positive/n_positive
true_negative_rate = n_true_negative/n_negative
false_positive_rate = 1.0 - true_positive_rate
false_negative_rate = 1.0 - true_negative_rate

pi = df[279].value_counts()/float(len(df))

#def cov_matrix (x, n):
#    return 1.0/n * x.T.dot(x)
#
#def weights_PCR (x, y, U, A, N):
#    return 1.0/N * (U.dot(np.linalg.inv(A).dot(U.T))).dot((x.T).dot(y))
#
#N = float(len(self.X_train))
#RX = cov_matrix(self.X_train, N)
#self.eigvals, U = np.linalg.eig(RX)
#A = np.diag(self.eigvals)[:self.L-1, :self.L-1]
#U = U[:, :self.L-1]
