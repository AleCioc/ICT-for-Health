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
            plt.plot(np.array(self.e_history)[0:100])
            
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

class MSE_LR (LR):
    
    def train (self):
        self.a = np.dot(np.linalg.pinv(self.X_train), self.y_train)
        self.e = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
        self.yhat_train = self.X_train.dot(self.a)
        return self.a, self.e, self.yhat_train
            
class GD_LR (LR):
    
    def train (self):
        
        def _gradient (X, y, a):
            return -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(a)
        
        learning_coefficient = 1.08e-4
        a_prev = np.random.rand(len(self.x_cols))
        self.a = np.random.rand(len(self.x_cols))
        self.e = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
        gradient = _gradient(self.X_train, self.y_train, self.a)
        iterations = 0
        max_iterations = 1e4
        
        while np.linalg.norm(self.a-a_prev) > 1e-8 and iterations < max_iterations:
            iterations += 1
            print iterations, np.linalg.norm(self.a-a_prev), self.e
            self.e_history += [self.e]                                           

            a_prev = self.a
            self.a = self.a - learning_coefficient * gradient
            self.e = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
            gradient = _gradient(self.X_train, self.y_train, self.a)        
        self.yhat_train = self.X_train.dot(self.a)
            
        return self.a, self.e, self.yhat_train

class SD_LR (LR):
    
    def train (self):
        
        def _gradient (X, y, a):
            return -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(a)
        def _hessian (X):
            return 4 * X.T.dot(X)
        
        a_prev = np.random.rand(len(self.x_cols))
        self.a = np.random.rand(len(self.x_cols))
        self.yhat_train = self.X_train.dot(a_prev)
        self.e = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
        gradient = _gradient(self.X_train, self.y_train, self.a)
        hessian = _hessian(self.X_train)
        iterations = 0
        max_iterations = 1e4
        
        while np.linalg.norm(self.a-a_prev) > 1e-8 and iterations < max_iterations:
            iterations += 1

            print iterations, np.linalg.norm(self.a-a_prev), self.e
            self.e_history += [self.e]                                           

            a_prev = self.a
            learning_coefficient = \
                (np.linalg.norm(gradient)**2)/(gradient.T.dot(hessian).dot(gradient))
            self.a = self.a - learning_coefficient * gradient
            self.e = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
            gradient = _gradient(self.X_train, self.y_train, self.a)        
        self.yhat_train = self.X_train.dot(self.a)
            
        return self.a, self.e, self.yhat_train

class SGD_LR (LR):
    
    def train (self):
                
        def _gradient (X, y, a):
            return -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(a)

        learning_coefficient = 1e-4
        a_prev = np.random.rand(len(self.x_cols))
        self.a = np.random.rand(len(self.x_cols))
        self.e = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
        iterations = 0
        max_iterations = 1e4
        
        batch_size = 20
        batch_prev = 0
        self.shuffled = training_df_st.sample(frac=1)
        
        while iterations < max_iterations:
            iterations += 1
            print iterations, np.linalg.norm(self.a-a_prev), self.e

            a_prev = self.a
            self.shuffled = self.shuffled.sample(frac=1)
            self.y_train = self.shuffled[y_col].values
            self.X_train = self.shuffled[self.x_cols].values
            
            for batch in range(batch_size, len(self.X_train), batch_size):
            
                X_batch = self.X_train[batch_prev:batch]
                y_batch = self.y_train[batch_prev:batch]
                batch_gradient = _gradient(X_batch, y_batch, self.a)
        
                self.a = self.a - learning_coefficient * batch_gradient
                batch_prev = batch
                self.e = np.linalg.norm(X_batch.dot(self.a) - y_batch)**2
                self.e_history += [self.e]

        self.yhat_train = self.X_train.dot(self.a)
            
        return self.a, self.e, self.yhat_train

#y_col = "Jitter(%)"
#
#mse = MSE_LR(training_df_st, testing_df_st, y_col)
#mse.train()
#mse.test()
#mse.plot()
#
#gd = GD_LR(training_df_st, testing_df_st, y_col)
#gd.train()
#gd.test()
#gd.plot()
#
#sd = SD_LR(training_df_st, testing_df_st, y_col)
#sd.train()
#sd.test()
#sd.plot()

#sgd = SGD_LR(training_df_st, testing_df_st, y_col)
#sgd_train_results = sgd.train()
#sgd_test_results  = sgd.test()
#sgd.plot()

plt.figure()
plt.plot(mse.y_test, color="green")
plt.plot(mse.yhat_test, marker="+", color="black", alpha=0.5)
plt.plot(gd.yhat_test, marker="x", color="red", alpha=0.5)
plt.plot(sd.yhat_test, marker="o", color="blue", alpha=0.5)
plt.plot(sgd.yhat_test, marker=".", color="yellow", alpha=0.5)
plt.legend()

plt.figure()
plt.plot(mse.yhat_test-mse.y_test, marker="+", color="black", alpha=0.5)
plt.plot(gd.yhat_test-mse.y_test, marker="x", color="red", alpha=0.5)
plt.plot(sd.yhat_test-mse.y_test, marker="o", color="blue", alpha=0.5)
plt.plot(sgd.yhat_test-mse.y_test, marker=".", color="yellow", alpha=0.5)
plt.legend()