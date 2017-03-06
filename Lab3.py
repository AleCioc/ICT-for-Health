import numpy as np
import pandas as pd

import csv

#import matplotlib.pyplot as plt
#import seaborn as sns

def lab3 ():
    
    df = pd.DataFrame(list(csv.reader(open("arrhythmia.csv","rb"), delimiter=',')))

    for col in df:
        if "?" in df[col].unique():
           df.drop(col, axis=1, inplace = True)
           continue
        df[col] = df[col].astype("float64")
        if len(df[col].unique()) == 1:
            u = df[col].unique()
            if u[0] == 0:
                df.drop(col, axis=1, inplace = True)
                        
    df = pd.DataFrame(data=df.values, \
                      index=df.index, \
                      columns=range(0,len(df.columns)))
    
        
    last = len(df.columns) - 1
    df[last].ix[(df[last] > 1)] = 2
    
    new_df = ((df-df.mean())/df.std()).drop(last, axis=1)
    y = new_df.values

    y1 = new_df.ix[(df[last] == 1)]
    y2 = new_df.ix[(df[last] == 2)]
    
    x1 = y1.mean()
    x2 = y2.mean()
    
    xmeans = np.stack([x1.values, x2.values])
    print xmeans.shape
    eny = np.diag(y.dot(y.T))
    print eny.shape
    enx = np.diag(xmeans.dot(xmeans.T))
    print enx.shape
    dotprod = y.dot(xmeans.T)
    print dotprod.shape
    U,V = np.meshgrid(enx, eny)
    dist2 = U + V - 2*dotprod
    print dist2.shape
        
    return y, dist2
    
#lab0ex1()
#lab0ex2()
#lab1()
#lab2()

y, distances = lab3()
