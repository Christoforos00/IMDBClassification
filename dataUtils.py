import re,os,glob
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import logReg

def preProcessText (lines):                                               #preprocessing for a single file
    lines = [line.replace("<br /><br />", ' ') for line in lines]         #removing the html tags
    lines = [line.replace("'", '')  for line in lines]                    #sticking together words that are separated with apostrophes (he's -> hes)
    lines = [re.sub('[\W_]+', ' ', line)  for line in lines]              #keeping only letters,numbers and spaces
    lines = [line.lower() for line in lines]
    return lines
    
#preprocessing for the whole data set, we replace each txt file with the preprocessed one
def preProcessData(path):                                               
    folders = ["/train/pos", "/train/neg" ,"/test/pos", "/test/neg" ]
    for folder in folders:
        print("preprocessing folder: "+path+folder)
        for filename in glob.glob(os.path.join(path+folder, '*.txt')):
            with open(os.path.join(filename), 'r') as f: 
                lines = f.readlines()
                lines = preProcessText (lines)
            with open(os.path.join( filename), 'w') as f:
                f.writelines(lines) 
    print("All files have been preprocessed\n")        
    
def createWordDictionary(path):                                            #creating the dictionary of words based on the training comments                               
    wordDict = {}
    folders = ["/train/pos", "/train/neg" ]    
    for folder in folders:
        for filename in glob.glob(os.path.join(path+folder, '*.txt')):
            with open(os.path.join(filename), 'r') as f: 
                lines = f.readlines()
                for line in lines:
                    words = line.split()
                    for word in words:
                        if word in wordDict.keys():
                            wordDict[word] += 1
                        else:
                            wordDict[word] = 1
    
    sorted_dict = {}
    sorted_keys = sorted(wordDict, key=wordDict.get,reverse=True) 
    for w in sorted_keys:
        sorted_dict[w] = wordDict[w]   
                             
    return sorted_dict  

def splitDictionary(dictionary,n,m):                                      #getting a sub-dictionary which consists if
    return dict(list(dictionary.items())[n:m+n])                          #n words, starting from the m-th word
                
                
def textToArray(path,dictionary,n,m):                                            #merges all the comments into one numpy array
    dictionary = splitDictionary(dictionary,n,m)

    rows = []
    folders = ["/train/pos", "/train/neg" ,"/test/pos", "/test/neg" ]
    for folder in folders:
        for filename in glob.glob(os.path.join(path+folder, '*.txt')):
            with open(os.path.join(filename), 'r') as f: 
                text = f.readlines()[0]
                temp = np.zeros(len(dictionary)+1)
                temp[len(dictionary)] = int("pos" in folder)                   #last column is 1 if the row is a positive review
                split = text.split()
                i = 0
                for word in dictionary.keys():
                    if word in split:
                        temp[i]=1
                    i+=1
                rows.append(temp)
    return np.array(rows)                                            #returns (50000,m) numpy array
    

def createFiles(dictionary):
    for n in [0,100,1000,10000,20000]:
        for m in [100,500,1000,2000,4000]:
            print("Creating file for " + "m: " +str(m)+ ", n: " + str(n) )
            array = textToArray('aclImdb',dictionary,n,m)
            np.save('data/'+str(n)+"_"+str(m), array )  


def printAccuracyForEach_m_n():        
#testing different hyperparameters(m,n) using the Cross Validation data

    for i in [0,100,1000,10000,20000]:
        for j in [100,500,1000,2000,4000]:
            array = np.load('data/'+str(i)+"_"+str(j)+".npy")
            trainArray = array[:25000]
            valArray =array[25000:] 

            np.random.shuffle(trainArray)
            X_train = trainArray[:,:trainArray.shape[1]-1]
            y_train = trainArray[:,trainArray.shape[1]-1:]

            X_val = valArray[:,:valArray.shape[1]-1]
            y_val = valArray[:,valArray.shape[1]-1:]
            #we always use random_state=101, so that the cv and test data will not change
            #if we split the data again for the same (m,n) combination
            X_cv, X_test, y_cv, y_test= train_test_split(X_val, y_val, train_size=0.5,test_size=0.5, random_state=101)
            
            w = logReg.gradientAscent(X_train,y_train,0.0001,1,0.5,0)
            predictions = logReg.predict(w,X_cv)
            acc = metrics.accuracy_score(y_cv, predictions)
            print("n:"+str(i)+" m:"+str(j)+" Accuracy:",acc)

       
            
def printAccuracyForEach_h_iter(X_train,y_train,X_cv,y_cv):       #testing different hyperparameters(step/h and iterations)
    for h in [0.01,0.05,0.1 ,0.25, 0.5 ,0.75]:
        for ite in [1,2,3]:

            w = logReg.gradientAscent(X_train,y_train,0.0001,ite,h,0)
            predictions = logReg.predict(w,X_cv)

            print("h:"+str(h)+" iter:"+str(ite)+" Accuracy:",metrics.accuracy_score(y_cv, predictions))

