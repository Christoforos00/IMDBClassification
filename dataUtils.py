import re,os,glob,json,math
import numpy as np
from numpy import random
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt 

def preProcessText (lines):                                               #preprocessing for a single file
    lines = [line.replace("<br /><br />", ' ') for line in lines]
    lines = [line.replace("'", '')  for line in lines]                    #sticking together words that are separated with apostrophes (he's -> hes)
    lines = [re.sub('[\W_]+', ' ', line)  for line in lines]              #keeping only the alphanumerics
    lines = [line.lower() for line in lines]
    return lines

def preProcessData(path):                                               #preprocessing for the whole data set
    folders = ["/train/pos", "/train/neg" ,"/test/pos", "/test/neg" ]
    for folder in folders:
        print(path+folder)
        for filename in glob.glob(os.path.join(path+folder, '*.txt')):
            with open(os.path.join(filename), 'r') as f: 
                lines = f.readlines()
                lines = preProcessText (lines)
            with open(os.path.join( filename), 'w') as f:
                f.writelines(lines) 
        
def createWordDictionary(path):                                            #creating the dictionary of words out of the training data                               
    wordDict = {}
    folders = ["/train/pos", "/train/neg" ]     #,"/test/pos", "/test/neg" 
    for folder in folders:
        print(path+folder)
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
                            
    return wordDict  

def splitDictionary(dictionary,n,m):
    return dict(list(dictionary.items())[n:m+n])
                
                
def textToArray(path,dictionary,n,m):
    dictionary = splitDictionary(dictionary,n,m)

    rows = []
    folders = ["/train/pos", "/train/neg" ,"/test/pos", "/test/neg" ]
    for folder in folders:
        for filename in glob.glob(os.path.join(path+folder, '*.txt')):
            with open(os.path.join(filename), 'r') as f: 
                text = f.readlines()[0]
                i = 0
                temp = np.zeros(len(dictionary)+1)
                temp[len(dictionary)] = int("pos" in folder)                          #last column is 1 if the review is positive
                split = text.split()
                for word in dictionary.keys():
                    if word in split:
                        temp[i]=1
                    i+=1
                rows.append(temp)
    return np.array(rows)

