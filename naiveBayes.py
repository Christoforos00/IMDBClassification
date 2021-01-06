import re,os,glob,json,math
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix


    
def bayesArr(X_train,y_train,X_test):
    prob = [0,0]
    ycounts = [y_train[y_train==0].shape[0] , y_train[y_train==1].shape[0]  ]  # \y=0\ , \y=1\
    ysize = y_train.shape[0] 
    featuresNum = X_train.shape[1]
    y_test = []
    countx0y0 = [ np.sum( (X_train[:,f]==0) & ((y_train==0)[:,0])) for f in range(featuresNum) ]  #(X(i) =0 ^ Y=0)
    countx0y1 = [ np.sum( (X_train[:,f]==0) & ((y_train==1)[:,0])) for f in range(featuresNum) ]  #(X(i) =0 ^ Y=1)
    countx0y = np.array([countx0y0,countx0y1 ])

    for x in X_test:
        for c in [0,1]:                                             #find P(y=0|x) and PP(y=0|x)
            prob[c]= np.log( ycounts[c] / ysize)+1
            for f in range(featuresNum):
                count = countx0y[c][f]                            #if x[f]==0
                if (x[f]==1):
                    count = ycounts[c] - count                
                prob[c] += np.log ( (count +1 )/( ycounts[c] + 2) )
         
        
        y_test.append( int( prob[0]< prob[1] )   )
    return np.array(y_test) 
