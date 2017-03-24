import numpy as np

import pandas as pd

#import statsmodels.api as sm

import matplotlib
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
matplotlib.style.use('ggplot')

df = pd.read_csv("../Data/parkinsons_updrs.csv")
df.test_time = df.test_time.apply(np.abs)

df["day"] = df.test_time.astype(np.int64)
df = df.groupby(["subject#", "day"]).mean()

training_df = df.loc[df.index.get_level_values('subject#') <= 36, df.columns.difference(["day","age","sex","test_time"])]
testing_df = df.loc[df.index.get_level_values('subject#') > 36, df.columns.difference(["day","age","sex","test_time"])]

def standardize (x):
    return (x-x.mean())/x.std()

training_df_st = training_df.apply(standardize)
testing_df_st = testing_df.apply(standardize)

a_0 = np.random.rand(len(training_df_st.columns)-1)
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

        self.e_train = 0.0
        self.e_test = 0.0

        self.a = a_0

    def train (self):        
        pass

    def test (self):
        
        self.yhat_test = self.X_test.dot(self.a)
        self.e_test = np.linalg.norm(self.yhat_test - self.y_test)**2
        
class MSE_LR (LR):
    
    def train (self):
        
        self.a = np.dot(np.linalg.pinv(self.X_train), self.y_train)
        self.e_train = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
        self.yhat_train = self.X_train.dot(self.a)

class PC_LR(object):
    
    def __init__ (self, training_df_st, testing_df_st, y_col, L):

        self.L = L
        
        self.y_col = y_col
        self.x_cols = training_df_st.columns.difference([y_col])
        
        self.train_df = training_df_st
        self.test_df = testing_df_st

        self.y_train = training_df_st[y_col].values
        self.X_train = training_df_st[self.x_cols].values

        self.y_test = testing_df_st[y_col].values
        self.X_test = testing_df_st[self.x_cols].values

    def train(self):

        def cov_matrix (x, n):
            return 1.0/n * x.T.dot(x)

        def weights_PCR (x, y, U, A, N):
            return 1.0/N * (U.dot(np.linalg.inv(A).dot(U.T))).dot((x.T).dot(y))
        
        N = float(len(self.X_train))
        RX = cov_matrix(self.X_train, N)
        self.eigvals, U = np.linalg.eig(RX)
        A = np.diag(self.eigvals)[:self.L-1, :self.L-1]
        U = U[:, :self.L-1]
        self.a = weights_PCR(self.X_train, self.y_train, U, A, N)
        self.yhat_train = self.X_train.dot(self.a)
        self.e_train = np.linalg.norm(self.yhat_train - self.y_train)**2

        return self.a

    def test (self):
        
        self.yhat_test = self.X_test.dot(self.a)
        self.e_test = np.linalg.norm(self.yhat_test - self.y_test)**2

        return self.yhat_test

y_col = "Jitter(%)"

mse = MSE_LR(training_df_st, testing_df_st, y_col)
mse.train()
mse.test()

pcrF = PC_LR(training_df_st, testing_df_st, y_col, len(training_df_st.columns)+1)
pcrF.train()
pcrF.test()

L1 = len(np.where(pcrF.eigvals.cumsum() < pcrF.eigvals.sum() * 0.999)[0])
L2 = len(np.where(pcrF.eigvals.cumsum() < pcrF.eigvals.sum() * 0.97)[0])

pcrL1 = PC_LR(training_df_st, testing_df_st, y_col, L1)
pcrL1.train()
pcrL1.test()

pcrL2 = PC_LR(training_df_st, testing_df_st, y_col, L2)
pcrL2.train()
pcrL2.test()

plt.figure(figsize=(13,6))
plt.plot(mse.a, marker="o", label="MSE")
plt.plot(pcrF.a, marker="o", label="PCR - F")
plt.plot(pcrL1.a, marker="o", label="PCR - L1")
plt.plot(pcrL2.a, marker="o", label="PCR - L2")
plt.title("Regression coefficients")
plt.xlabel("Feature")
plt.ylabel("Weight")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.plot(pcrF.y_train, color="black", label="y true")
plt.plot(mse.yhat_train, label="MSE")
plt.plot(pcrF.yhat_train, label="PCR - F")
plt.plot(pcrL1.yhat_train, label="PCR - L1")
plt.plot(pcrL2.yhat_train, label="PCR - L2")
plt.title("Train prediction")
plt.xlabel("Sample index")
plt.ylabel("Normalized value")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.plot(pcrF.y_test, color="black", label="y true")
plt.plot(mse.yhat_test, label="MSE")
plt.plot(pcrF.yhat_test, label="PCR - F")
plt.plot(pcrL1.yhat_test, label="PCR - L1")
plt.plot(pcrL2.yhat_test, label="PCR - L2")
plt.title("Test prediction")
plt.xlabel("Sample index")
plt.ylabel("Normalized value")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.scatter(mse.yhat_train, mse.y_train, label="MSE", marker=".", alpha=0.4)
plt.plot(mse.y_train, mse.y_train, linewidth=0.2)
plt.scatter(pcrF.yhat_train, pcrF.y_train, label="PCR - F", marker="o", alpha=0.4)
plt.plot(pcrF.y_train, pcrF.y_train, linewidth=0.2)
plt.scatter(pcrL1.yhat_train, pcrL1.y_train, label="PCR - L1", marker="x", alpha=0.4)
plt.plot(pcrL1.y_train, pcrL1.y_train, linewidth=0.2)
plt.scatter(pcrL2.yhat_train, pcrL2.y_train, label="PCR - L2", marker="+", alpha=0.4)
plt.plot(pcrL2.y_train, pcrL2.y_train, linewidth=0.2)
plt.title("Train - y_true VS y_hat")
plt.xlabel("y_true")
plt.ylabel("y_hat")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.scatter(mse.yhat_test, mse.y_test, label="MSE", marker="o", alpha=0.4)
plt.plot(mse.y_test, mse.y_test, linewidth=0.2)
plt.scatter(pcrF.yhat_test, pcrF.y_test, label="PCR - F", marker="o", alpha=0.4)
plt.plot(pcrF.y_test, pcrF.y_test, linewidth=0.2)
plt.scatter(pcrL1.yhat_test, pcrL1.y_test, label="PCR - L1", marker="x", alpha=0.4)
plt.plot(pcrL1.y_test, pcrL1.y_test, linewidth=0.2)
plt.scatter(pcrL2.yhat_test, pcrL2.y_test, label="PCR - L2", marker="+", alpha=0.4)
plt.plot(pcrL2.y_test, pcrL2.y_test, linewidth=0.2)
plt.title("Test - y_true VS y_hat")
plt.xlabel("y_true")
plt.ylabel("y_hat")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.hist(mse.yhat_train-mse.y_train, bins=100, label="MSE", alpha=0.4)
plt.hist(pcrF.yhat_train-pcrF.y_train, bins=100, label="PCR - F", alpha=0.4)
plt.hist(pcrL1.yhat_train-pcrL1.y_train, bins=100, label="PCR - L1", alpha=0.4)
plt.hist(pcrL2.yhat_train-pcrL2.y_train, bins=100, label="PCR - L2", alpha=0.4)
plt.title("Train - Error histogram")
plt.xlabel("Error")
plt.ylabel("Occurrencies")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.hist(mse.yhat_test-mse.y_test, bins=100, label="MSE", alpha=0.4)
plt.hist(pcrF.yhat_test-pcrF.y_test, bins=100, label="PCR - F", alpha=0.4)
plt.hist(pcrL1.yhat_test-pcrL1.y_test, bins=100, label="PCR - L1", alpha=0.4)
plt.hist(pcrL2.yhat_test-pcrL2.y_test, bins=100, label="PCR - L2", alpha=0.4)
plt.title("Test - Error histogram")
plt.xlabel("Error")
plt.ylabel("Occurrencies")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
index = np.arange(0,4,1)
xticks = ["MSE"
          "PCR - F", 
          "PCR - L1", 
          "PCR - L2"]
train_errors = [mse.e_train, pcrF.e_train, pcrL1.e_train, pcrL2.e_train]
test_errors = [mse.e_test, pcrF.e_test, pcrL1.e_test, pcrL2.e_test]
plt.bar(index, train_errors, 0.35, label="Train", color="blue")
plt.bar(index + 0.35, test_errors, 0.35, label="Test", color="red")
plt.title("Squared errors")
plt.xlabel("Method")
plt.ylabel("Error")
plt.xticks(index + 0.35, xticks)
plt.legend(loc=1)
