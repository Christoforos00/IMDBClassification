import re,os,glob,json,math
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix

def gradientAscent(X,Y,threshold,maxIterations,h,_lambda):
    w =  random.randint(low=-10,high=10, size=(1,X.shape[1]))
    iterations = 0
    while True :
        s0=0 
        s1=0
        for i in range(X.shape[0]):
            x = X[i:i+1]   
            y = Y[i:i+1]    
            l = L(w,x,y) - _lambda*np.sum(np.power(w,2))
            s0 = s1
            s1 = l
            w = w +  h * ( derivative(w,x,y)  - 2*_lambda*(w) )
            
        iterations+=1    
        if ( abs(s0-s1) < threshold or iterations > maxIterations):
            return w
    
    return w




def accuracy(w,X,y):
    return metrics.accuracy_score(y, predictP(w,X) )

def P(w,X):
    return sigmoid(X.dot(w.T) )

def sigmoid(x):
    return 1/(1+np.exp(-x))

def L(w,x,y):
    p = P(w,x)
    return y.dot( np.log(p) ) + (1-y).dot(np.log(1-p)) 

def derivative(w,X,Y):
    p = P(w,X)
    return  (Y-p).dot(X)   # (X.T.dot(Y-p)    )

def predictP(w,X):
    p = P(w,X)
    return np.array([int( prob>0.5 ) for prob in p ])
