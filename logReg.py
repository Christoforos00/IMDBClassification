import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt 


def gradientAscent(X,Y,threshold,maxIterations,h,_lambda):
    w =  random.randint(low=-2,high=2, size=(1,X.shape[1]))
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



def P(w,X):
    return sigmoid(X.dot(w.T) )

def sigmoid(x):
    return 1/(1+np.exp(-x))

# loss function 
def L(w,x,y):
    p = P(w,x)
    return y.dot( np.log(p) ) + (1-y).dot(np.log(1-p)) 

def derivative(w,X,Y):
    p = P(w,X)
    return  (Y-p).dot(X)  

 #use the probabilities to return the predictions
def predict(w,X):           
    p = P(w,X)
    return np.array([int( prob>0.5 ) for prob in p ])

#the accuracy% of the prediction       
def accuracy(w,X,y):                                 
    return metrics.accuracy_score(y, predict(w,X) )
    
 #returns the data for test and train accuracy,precision,recall,f1,  (takes ~30 secs)
def getMetrics(X_train,y_train,X_test,y_test):       
    test=[]                                              
    train=[]
    precision = []
    recall = []
    f1 = []
    for i in range(1,X_train.shape[0],3000):
        w = gradientAscent(X_train[:i],y_train[:i],0.0001,1,0.05,0)
        logRegPredictions = predict(w,X_test)

        test.append(accuracy(w,X_test,y_test) )
        train.append(accuracy(w,X_train[:i],y_train[:i]) )
        calc = metrics.precision_recall_fscore_support(y_test,logRegPredictions)
        precision.append(calc[0] )
        recall.append(calc[1] )
        f1.append(calc[2] )    

    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)     
    
    plt.plot(range(len(test)),test, label="test Accuracy" )      #only plot the learning curves
    plt.plot(range(len(train)),train ,label="train Accuracy")   
    plt.legend()
    plt.show()
    
    return test,train,precision,recall,f1
