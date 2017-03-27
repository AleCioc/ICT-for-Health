import numpy as np
from scipy.io import arff
import pandas as pd
from sklearn.cluster import KMeans

import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

def standardize (x):
    return (x-x.mean())/x.std()

def square_distance (x, xk):
    return np.linalg.norm(x - xk)**2

def cov_matrix (x):
    n = float(len(x))
    return 1.0/n * x.T.dot(x)

def load_data ():
    
    df = pd.read_csv("../Data/chronic_kidney_disease.csv", skipinitialspace=True, header=None)
    df = df.replace({"?": np.NaN})
    df = df.replace({"\t?": np.NaN})
    df = df.replace({"\tyes": "yes"})
    df = df.replace({"\tno": "no"})
    df = df.replace({"ckd\t": "no"})
    for col in df:
        df[col] = df[col].fillna(df[col].mode().values[0])
    keylist = ["normal","abnormal","present","notpresent","yes","no","good","poor","ckd","notckd"]
    keymap = [0,1,0,1,0,1,0,1,1,0]
    df = df.replace(keylist, keymap)
    for col in [5,6,7,8,18,21,22,23]:
        df[(-df[col].isin([0,1]))] = df[col].mode().values[0]
        
    return df

df = load_data().astype(float)
true = df[24]
df = df.drop(24, axis=1)

#kmeans = KMeans(n_clusters=8, 
#                init='k-means++', 
#                n_init=10, 
#                max_iter=300, 
#                tol=0.0001, 
#                precompute_distances='auto', 
#                verbose=0, 
#                random_state=None, 
#                copy_x=True, 
#                n_jobs=1, 
#                algorithm='auto').fit(df.values)

from scipy.cluster.hierarchy import dendrogram, linkage

Z = linkage(df.apply(standardize).values, 
            method='single', 
            metric='euclidean')

plt.figure(figsize=(13, 6))
plt.title("Complete dendrogram")
dendrogram(Z)

plt.figure(figsize=(13, 6))
plt.title("Reduced dendrogram, Mathematica(TM) style")
dendrogram(Z, truncate_mode="mtica")

plt.figure(figsize=(13, 6))
plt.title("Reduced dendrogram, Last p method")
dendrogram(Z, truncate_mode="lastp")

from sklearn import cluster
from sklearn import tree
from sklearn import metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import ShuffleSplit

model = cluster.AgglomerativeClustering(n_clusters=2)
clusters = model.fit_predict(df.values)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
print metrics.silhouette_score(df, model.labels_, metric='sqeuclidean')

def get_PCA (x):
                    
    Rx = cov_matrix(x)
    eigvals, U = np.linalg.eig(Rx)
    A = np.diag(eigvals)
    z = x.dot(U).dot(np.linalg.inv(A))
    
    return z

df = df.apply(standardize)
#df = get_PCA(df.apply(standardize))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(df.values, true)

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
scores = cross_val_score(clf, df, true, cv=cv)
print scores

predicted = cross_val_predict(clf, df, true, cv=10)
print metrics.accuracy_score(true, predicted)

import pydotplus

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_ps("clf_tree.ps")

from IPython.display import Image
Image(filename='clf_tree.png')
