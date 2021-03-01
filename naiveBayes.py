import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt 

    
def predictBayes(X_train,y_train,X_test):
    prob = [0,0]
    ycounts = [y_train[y_train==0].shape[0] , y_train[y_train==1].shape[0]  ]  # \y=0\ , \y=1\
    ysize = y_train.shape[0] 
    featuresCount = X_train.shape[1]
    predictions = []
    countx0y0 = [ np.sum( (X_train[:,f]==0) & ((y_train==0)[:,0])) for f in range(featuresCount) ]  #(for f from 0 to featuresCount: count rows where X(f) =0 ^ Y=0)
    countx0y1 = [ np.sum( (X_train[:,f]==0) & ((y_train==1)[:,0])) for f in range(featuresCount) ]  #(for f from 0 to featuresCount: count rows where X(f) =0 ^ Y=1)
    countxy = np.array([ [countx0y0,[ycounts[0]-val for val in countx0y0]],[countx0y1,[ycounts[1]-val for val in countx0y1]] ] )  # [countx0y0.countx1y0,countx0y1,countx1y1] 

    for x in X_test:
        for c in [0,1]:                                             #find P(y=0|x) and P(y=1|x)
            prob[c] = np.log( ycounts[c] / ysize) + sum ([np.log ( (countxy[c][int(xf)][f]   +1 )/( ycounts[c] + 2) ) for f,xf in enumerate(x)])      
        predictions.append( int( prob[0]< prob[1] )   )
    return np.array(predictions) 
   
    
def getMetrics(X_train,y_train,X_test,y_test):       #returns the data for the metrics
    test=[]                                              #and learning curves  (takes 1 min)
    train=[]
    precision = []
    recall = []
    f1 = []
    for i in [1,1000,2000,6000,10000,30000]:
        bayesPredictions = predictBayes(X_train[:i],y_train[:i],X_test)
        test.append(metrics.accuracy_score(y_test,bayesPredictions) )
        train.append(metrics.accuracy_score(y_train[:i],predictBayes(X_train[:i],y_train[:i],X_train[:i])) )
        calc = metrics.precision_recall_fscore_support(y_test,bayesPredictions)
        precision.append(calc[0] )
        recall.append(calc[1] )
        f1.append(calc[2] )    

    precision = np.array(precision)
    recall = np.array(recall)
    f1 = np.array(f1)    


    plt.plot(range(len(test)),test, label="test Accuracy" )     #only plot the learning curves
    plt.plot(range(len(train)),train ,label="train Accuracy")   
    plt.legend()
    plt.show() 
    
    return test,train,precision,recall,f1
    

