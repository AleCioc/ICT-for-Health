import numpy as np

import pandas as pd

#import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use('ggplot')

df = pd.read_csv("../Data/parkinsons_updrs.data")
df.test_time = df.test_time.apply(np.abs)

df["day"] = df.test_time.astype(np.int64)
df = df.groupby(["subject#", "day"]).mean()

training_df = df.loc[df.index.get_level_values('subject#') <= 36, df.columns.difference(["day","age","sex","test_time"])]
testing_df = df.loc[df.index.get_level_values('subject#') > 36, df.columns.difference(["day","age","sex","test_time"])]

def standardize (x):
    return (x-x.mean())/x.std()

training_df_st = training_df.apply(standardize)
testing_df_st = testing_df.apply(standardize)

class LR (object):
    
    def __init__ (self, training_df_st, testing_df_st, y_col):

        self.y_col = y_col
        self.x_cols = training_df_st.columns.difference([y_col])
        
        self.train_df = training_df_st
        self.test_df = testing_df_st

        self.y_train = training_df_st[y_col].values
        self.X_train = training_df_st[self.x_cols].values

        self.y_test = testing_df_st[y_col].values
        self.X_test = testing_df_st[self.x_cols].values

        self.e_history = []

    def test (self):
        
        self.yhat_test = self.X_test.dot(self.a)
        self.e = self.yhat_test - self.y_test
        
        return self.e, self.yhat_test

    def plot (self):
        
        plt.figure()
        plt.plot(self.a, marker="o")
        plt.xticks(range(len(self.a)), self.x_cols, rotation='vertical')
        
        if len(self.e_history):
            plt.figure()
            plt.plot(np.array(self.e_history)[0:1000])
            
#        plt.figure()
#        plt.plot(self.yhat_train, self.yhat_train, color="black", linewidth=1)
#        plt.scatter(self.yhat_train, self.y_train, marker="o", color="red")
#        
#        plt.figure()
#        plt.plot(self.yhat_test, self.yhat_test, color="black", linewidth=1)
#        plt.scatter(self.yhat_test, self.y_test, marker="o", color="red")
#        
#        plt.figure()
#        plt.hist(self.yhat_train - self.y_train, bins=100)
#        
#        plt.figure()
#        plt.hist(self.yhat_test - self.y_test, bins=100)
                
        plt.figure()
        plt.plot(self.yhat_test)
        plt.plot(self.y_test)        

class PC_LR(LR):
    
    def test (self):
        
        self.yhat_test = self.X_test[:,:5].dot(self.a[:5])        
        return self.yhat_test
    
    def train(self):
        N = float(len(self.X_train))
        RX = 1.0/N * self.X_train.T.dot(self.X_train)
        eigvals, U = np.linalg.eig(RX)
        Lambda = np.diag(eigvals)
        self.a = 1.0/N * U.dot(np.linalg.inv(Lambda).dot(U.T).dot(self.X_train.T).dot(self.y_train))        

        return self.a

y_col = "Jitter(%)"

pcr = PC_LR(training_df_st, testing_df_st, y_col)
pcr_train_results = pcr.train()
pcr_test_results  = pcr.test()
pcr.plot()
