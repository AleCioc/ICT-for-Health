import collections
import itertools

import numpy as np

import scipy as sp
import pandas as pd

import statsmodels.api as sm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import seaborn as sns

class DiscreteEnsemble (object):
    
    def __init__ (self, name, x):
        
        self.name = name
        self.x = pd.Series(x)
        self.n = len(self.x.values)

        self.occ = self.x.value_counts().sort_index()
        self.norm_occ = self.occ / self.n
        self.norm_cum = self.norm_occ.cumsum()
        self.Dx = collections.OrderedDict()
        for key in sorted(set(self.x.unique())):
            self.Dx[key] = self.norm_occ.ix[key]
    
            
class ContinuousEnsemble (object):
    
    def __init__ (self, name, x):
        
        self.name = name
        self.x = pd.Series(x)
        self.n = len(self.x.values)

        self.kde = sm.nonparametric.KDEUnivariate(np.array(self.x.values))
        self.kde.fit()
        self.pdf = np.zeros(len(self.kde.icdf))
        for i in range(0,len(self.kde.icdf)):
            v = self.kde.icdf[i]
            self.pdf[i] = self.kde.evaluate(v)
        
    def cdf (self, v):
        for i,j in itertools.izip(list(self.kde.icdf),list(self.kde.cdf)):
            if v < i:
                return j
            
    def plot(self, interval = None):
        
        if interval != None:
            fig, (ax1,ax2) = plt.subplots(1,2, figsize = (18,6))
            ax1.set_title("P(" + self.name + ")")
            ax2.set_title("F(" + self.name + ")")
            ax1.plot(self.kde.icdf, self.pdf)
            ax2.plot(self.kde.icdf, self.kde.cdf)
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=70)
        else:
            fig, (ax1,ax2) = plt.subplots(1,2, figsize = (18,6))
            sns.distplot(self.x,ax=ax1)
            sns.distplot(self.x.values,ax=ax2, hist_kws=dict(cumulative=True),kde_kws=dict(cumulative=True))
    def show (self):
        print "Ensamble name: " + str(self.name)

def lab0ex1 ():
        
    TH = np.arange(1,20,0.1)

    n_samples = 1000    
    bilirubin_sick = ContinuousEnsemble("Dy", pd.Series(np.random.normal(15, 8, n_samples)))
    bilirubin_healty = ContinuousEnsemble("Dn", pd.Series(np.random.normal(10, 4, n_samples)))
    #bilirubin_sick.plot()
    #bilirubin_healty.plot()

    sensitivity = np.array([1.0 - bilirubin_sick.cdf(float(th)) for th in np.nditer(TH)]).reshape(len(TH),1)
    specificity = np.array([bilirubin_healty.cdf(float(th)) for th in np.nditer(TH)]).reshape(len(TH),1)
    false_negative = 1.0 - sensitivity
    false_positive = 1.0 - specificity
    #print sensitivity.shape
    #print specificity.shape
    #print false_negative.shape
    #print false_positive.shape

    incidence = np.arange(0.01,1.0,0.01)
    #print incidence.shape
    
    P_Dy = incidence
    P_Dn = 1.0 - incidence
    #print P_Dy
    #print P_Dn
        
    P_Tp = np.multiply(sensitivity,P_Dy) + np.multiply(false_negative,P_Dn)
    P_Tn = np.multiply(false_positive,P_Dy) + np.multiply(specificity,P_Dn)
    #print P_Tp.shape
    #print P_Tn.shape
    
    P_Dy_given_Tp = np.divide(np.multiply(sensitivity,P_Dy), P_Tp)
    P_Dn_given_Tn = np.divide(np.multiply(specificity,P_Dn), P_Tn)
    #print P_Dy_given_Tp.shape
    #print P_Dn_given_Tn.shape
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize = (12,6))
    
    ax1.set_title("Sensitivity vs specificity")
    ax2.set_title("False negative vs false positive probabilities")    
    
    sensitivity_plot, = ax1.plot(TH,sensitivity,label="Specificity")
    specificity_plot, = ax1.plot(TH,specificity,label="Sensitivity")
    ax1.legend(handles=[sensitivity_plot,specificity_plot],loc=0)
    
    false_negative_plot, = ax2.plot(TH,false_negative, label="False negative")
    false_positive_plot, = ax2.plot(TH,false_positive, label="False positive")
    ax2.legend(handles=[false_negative_plot,false_positive_plot],loc=0)

    plt.show()
    
    fig = plt.figure(figsize = (12,6))
    ax1 = fig.add_subplot(1,2,1,projection='3d')
    ax2 = fig.add_subplot(1,2,2,projection='3d')
    x = incidence
    y = TH
    #print x.shape
    #print y.shape
    x,y = np.meshgrid(x,y)
    z1 = P_Dy_given_Tp
    #print z1.shape
    z2 = P_Dn_given_Tn
    ax1.set_title("P(true_positive)")
    ax2.set_title("P(true_negative)")    
    ax1.plot_surface(x,y,z1, rstride=8, cstride=8, alpha=0.3)
    ax1.contourf(x,y,z1, zdir='z', offset=0, cmap=plt.cm.coolwarm)
    ax2.plot_surface(x,y,z2, rstride=8, cstride=8, alpha=0.3)
    ax2.contourf(x,y,z2, zdir='z', offset=0, cmap=plt.cm.coolwarm)
    
    plt.show()

def lab0ex2 ():

    #zF[int(j)][int(i)] = (np.exp(-1.0/l)-np.exp(-x/l)) / (np.exp(-1/l)-np.exp(-20.0/l))
#    fig = plt.figure(figsize = (18,6))
#    ax1 = fig.add_subplot(1,2,1)
#    ax2 = fig.add_subplot(1,2,2)
#    ax1.set_title("f(x|" + r"$\lambda$)")
#    ax2.set_title(r"f($\lambda$|x)")

    def Z(y):
        return (np.exp(-1.0/y) - np.exp(-20.0/y))
    def F2D(x,y):
        (np.exp(-1.0/y)-np.exp(-x/y)) / Z(y)
    def f2D(x,y):
        return (np.exp(-x/y) / Z(y)) / y
    def f3D(x,y):
        return (np.exp(-np.multiply(x,y)) / Z(y)) / y
    def Zinv(y):
        return (np.exp(-20.0/y) - np.exp(-1.0/y))
    def Finv(u,y):
        return -u * np.log( Zinv(u) * (np.exp(-1.0/u) / Zinv(u) + y) )
    def Mean(y):
        return y + (np.exp(-1.0/y)-np.exp(-20.0/y)) / Z(y) - np.mean([1.5,2,3,4,5,12])


    x = np.array(np.arange(1.0,10.0,0.1))
    params = np.array([2.0,5.0,10.0])
    
    fig = plt.figure(figsize = (12,6))
    
    ax1 = fig.add_subplot(1,2,1)
    plots = list()
    for param in np.nditer(params):
        p, = ax1.plot(x,f2D(x,param),label= "$\lambda$ = " + str(param))
        plt.grid()
        plots.append(p)
    ax1.set_title("P (x | $\lambda$)")
    ax1.legend(handles=plots)
    ax1.grid(True)
    
    x = np.array([3.0,5.0,12.0])
    params = np.arange(1.0,100.0,0.1)

    ax2 = fig.add_subplot(1,2,2)
    plots = list()
    for i in np.nditer(x):
        p, = ax2.plot(params,f2D(i,params),label= "x = " + str(i))
        plt.grid()
        plots.append(p)
    ax2.set_title("P ($\lambda$ | x)")
    ax2.set_xscale('log')
    ax2.legend(handles=plots)
    ax2.grid(True)

    fig.set_tight_layout(True)
        
    x_range = np.array([1.5,2,3,4,5,12])
    y_range = np.arange(1.0,20.0,0.1)    
    x = np.array(x_range).reshape(len(x_range),1)
    y = np.array(y_range).reshape(1,len(y_range))

    L = 1
    for i in np.nditer(x):
        L *= f2D(i,y)
    y = y.reshape(len(y_range),1)
    L = L.reshape(len(y_range),1)

    #print y.shape
    #print L.shape
    fig = plt.figure(figsize = (6,6))
    ax1 = fig.add_subplot(1,1,1)
    ax1.plot(y, L)
    fig.set_tight_layout(True)

    x = sp.optimize.broyden1(Mean, 1, f_tol=1e-14)
    print x
    
    plt.show()

#lab0ex1()
lab0ex2()
