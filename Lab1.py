import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
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
            
class GD_LR (LR):
    
    def train (self):
        
        def _gradient (X, y, a):
            return -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(a)
        
        learning_coefficient = 1.0e-4
        a_prev = np.ones(len(self.x_cols))
        self.e_train = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
        gradient = _gradient(self.X_train, self.y_train, self.a)
        iterations = 0
        max_iterations = 1e4
        self.e_history = []
        
        while np.linalg.norm(self.a-a_prev) > 1e-8 and iterations < max_iterations:
            iterations += 1
#            print iterations, np.linalg.norm(self.a-a_prev), self.e_train
            self.e_history += [self.e_train]                                           

            a_prev = self.a
            self.a = self.a - learning_coefficient * gradient
            self.e_train = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
            gradient = _gradient(self.X_train, self.y_train, self.a)        
        self.yhat_train = self.X_train.dot(self.a)
            
class SD_LR (LR):
    
    def train (self):
        
        def _gradient (X, y, a):
            return -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(a)
        def _hessian (X):
            return 4 * X.T.dot(X)
        
        a_prev = np.ones(len(self.x_cols))
        self.yhat_train = self.X_train.dot(a_prev)
        self.e_train = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
        gradient = _gradient(self.X_train, self.y_train, self.a)
        hessian = _hessian(self.X_train)
        iterations = 0
        max_iterations = 1e4
        self.e_history = []
        
        while np.linalg.norm(self.a-a_prev) > 1e-8 and iterations < max_iterations:
            iterations += 1

#            print iterations, np.linalg.norm(self.a-a_prev), self.e_train
            self.e_history += [self.e_train]                                           

            a_prev = self.a
            learning_coefficient = \
                (np.linalg.norm(gradient)**2)/(gradient.T.dot(hessian).dot(gradient))
            self.a = self.a - learning_coefficient * gradient
            self.e_train = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
            gradient = _gradient(self.X_train, self.y_train, self.a)        
        self.yhat_train = self.X_train.dot(self.a)
            
class SGD_LR (LR):
    
    def train (self):
                
        def _gradient (X, y, a):
            return -2 * X.T.dot(y) + 2 * X.T.dot(X).dot(a)

        learning_coefficient = 1.0e-4
        self.e_train = np.linalg.norm(self.X_train.dot(a_0) - self.y_train)**2
        iterations = 0
        max_iterations = 1e4
        self.e_history = [self.e_train]
        
        batch_size = 20
        self.shuffled = self.train_df.sample(frac=1)
        
        while iterations < max_iterations:
            iterations += 1
#            print iterations, np.linalg.norm(self.a-a_prev), self.e_train
                                            
            self.shuffled = self.shuffled.sample(frac=1)
            self.y_train = self.shuffled[self.y_col].values
            self.X_train = self.shuffled[self.x_cols].values

            batch_prev = 0
            for batch in range(batch_size, len(self.X_train), batch_size):
            
                X_batch = self.X_train[batch_prev:batch]
                y_batch = self.y_train[batch_prev:batch]
                batch_gradient = _gradient(X_batch, y_batch, self.a)
        
                self.a = self.a - learning_coefficient * batch_gradient
                batch_prev = batch

            self.e_train = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
            self.e_history += [self.e_train]
            
        self.yhat_train = self.train_df[self.x_cols].values.dot(self.a)
        self.e_train = np.linalg.norm(self.X_train.dot(self.a) - self.y_train)**2
            
y_col = "Jitter(%)"
#y_col = "motor_UPDRS"
##y_col = "total_UPDRS"
#
mse = MSE_LR(training_df_st, testing_df_st, y_col)
mse.train()
mse.test()
#
gd = GD_LR(training_df_st, testing_df_st, y_col)
gd.train()
gd.test()
#
sd = SD_LR(training_df_st, testing_df_st, y_col)
sd.train()
sd.test()
#
sgd = SGD_LR(training_df_st, testing_df_st, y_col)
sgd.train()
sgd.test()

plt.figure(figsize=(13,6))
plt.plot(mse.a, marker="o", label="minimum squares")
plt.plot(gd.a, marker="o", label="gradient descent")
plt.plot(sd.a, marker="o", label="steepest descent")
plt.plot(sgd.a, marker="o", label="stochastic gradient descent")
plt.xticks(range(len(mse.a)), mse.x_cols, rotation="vertical")
plt.title("Regression coefficients")
plt.xlabel("Feature")
plt.ylabel("Weight")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.plot(gd.e_history[:100], marker="o", label="gradient descent")
plt.plot(sd.e_history[:100], marker="o", label="steepest descent")
plt.plot(sgd.e_history[:100], marker="o", label="stochastic gradient descent")
plt.title("Error history - first 100 iterations")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.legend(loc=0)

plt.figure(figsize=(13,6))
plt.plot(mse.y_train, color="black", label="y true")
plt.plot(mse.yhat_train, label="minimum squares")
plt.plot(gd.yhat_train, label="gradient descent")
plt.plot(sd.yhat_train, label="steepest descent")
plt.plot(sgd.yhat_train, label="stochastic gradient descent")
plt.title("Train prediction")
plt.xlabel("Sample index")
plt.ylabel("Normalized value")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.plot(mse.y_test, color="black", label="y true")
plt.plot(mse.yhat_test, label="minimum squares")
plt.plot(gd.yhat_test, label="gradient descent")
plt.plot(sd.yhat_test, label="steepest descent")
plt.plot(sgd.yhat_test, label="stochastic gradient descent")
plt.title("Test prediction")
plt.xlabel("Sample index")
plt.ylabel("Normalized value")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
index = np.arange(0,4,1)
xticks = ["minimum square estimate", 
          "gradient descent", 
          "steepest descent", 
          "stochastic gradient descent"]
train_errors = [mse.e_train, gd.e_train, sd.e_train, sgd.e_train]
test_errors = [mse.e_test, gd.e_test, sd.e_test, sgd.e_test]
plt.bar(index, train_errors, 0.35, label="Train", color="blue")
plt.bar(index + 0.35, test_errors, 0.35, label="Test", color="red")
plt.title("Squared errors")
plt.xlabel("Method")
plt.ylabel("Error")
plt.xticks(index + 0.35, xticks)
plt.legend(loc=1)

plt.figure(figsize=(13,6))
plt.scatter(mse.yhat_train, mse.y_train, label="minimum square estimate", marker="o", color="red", alpha=0.4)
plt.plot(mse.y_train, mse.y_train, linewidth=0.2)
plt.scatter(gd.yhat_train, gd.y_train, label="gradient descent", marker="x", color="blue", alpha=0.4)
plt.plot(gd.y_train, gd.y_train, linewidth=0.2)
plt.scatter(sd.yhat_train, sd.y_train, label="steepest descent", marker=".", color="green", alpha=0.4)
plt.plot(sd.y_train, sd.y_train, linewidth=0.2)
plt.scatter(sgd.yhat_train, sgd.train_df[y_col].values, label="stochastic gradient descent", marker="+", color="grey", alpha=0.4)
plt.plot(sgd.y_train, sgd.y_train, linewidth=0.2)
plt.title("Train - y_true VS y_hat")
plt.xlabel("y_true")
plt.ylabel("y_hat")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.scatter(mse.yhat_test, mse.y_test, label="minimum square estimate", marker="o", color="red", alpha=0.4)
plt.plot(mse.y_test, mse.y_test, linewidth=0.2)
plt.scatter(gd.yhat_test, gd.y_test, label="gradient descent", marker="x", color="blue", alpha=0.4)
plt.plot(gd.y_test, gd.y_test, linewidth=0.2)
plt.scatter(sd.yhat_test, sd.y_test, label="steepest descent", marker=".", color="green", alpha=0.4)
plt.plot(sd.y_test, sd.y_test, linewidth=0.2)
plt.scatter(sgd.yhat_test, sgd.y_test, label="stochastic gradient descent", marker="+", color="grey", alpha=0.4)
plt.plot(sgd.y_test, sgd.y_test, linewidth=0.2)
plt.title("Test - y_true VS y_hat")
plt.xlabel("y_true")
plt.ylabel("y_hat")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.hist(mse.yhat_train-mse.y_train, bins=100, label="minimum square estimate", alpha=0.4)
plt.hist(gd.yhat_train-gd.y_train, bins=100, label="gradient descent", alpha=0.4)
plt.hist(sd.yhat_train-sd.y_train, bins=100, label="steepest descent", alpha=0.4)
plt.hist(sgd.yhat_train-sgd.train_df[y_col].values, bins=100, label="stochastic gradient descent", alpha=0.4)
plt.title("Train - Error histogram")
plt.xlabel("Error")
plt.ylabel("Occurrencies")
plt.legend(loc=2)

plt.figure(figsize=(13,6))
plt.hist(mse.yhat_test-mse.y_test, bins=100, label="minimum square estimate", alpha=0.4)
plt.hist(gd.yhat_test-gd.y_test, bins=100, label="gradient descent", alpha=0.4)
plt.hist(sd.yhat_test-sd.y_test, bins=100, label="steepest descent", alpha=0.4)
plt.hist(sgd.yhat_test-sgd.y_test, bins=100, label="stochastic gradient descent", alpha=0.4)
plt.title("Test - Error histogram")
plt.xlabel("Error")
plt.ylabel("Occurrencies")
plt.legend(loc=2)
