import numpy as np
from scipy.io import arff
import pandas as pd

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

df = load_data().drop(24, axis=1)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit(df.values)

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(df.values)
plt.figure()
dendrogram(Z)
plt.figure()
dendrogram(Z, truncate_mode="mtica")
plt.figure()
dendrogram(Z, truncate_mode="lastp")

df_class = load_data()

from sklearn import tree

X = df_class.loc[:, :23]
Y = df_class[24]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

import pydotplus

dot_data = tree.export_graphviz(clf, out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_ps("clf_tree.ps")

from IPython.display import Image
Image(filename='clf_tree.png')
