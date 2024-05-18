# Import the necessary libraries
import shutil
import numpy as np
import pandas as pd
import scipy as sc
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import os
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#https://machinelearningmastery.com/machine-learning-in-python-step-by-step/








#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()

#dataset.hist()
#pyplot.show()

# scatter plot matrix
#scatter_matrix(dataset)
#pyplot.show()
#FILE=os.path.join(os.path.dirname(__file__),"sectionimg/","data.csv")
#FILE2=os.path.join(os.path.dirname(__file__),"sectionimg/","data2.csv")
#NAMES=['numofcharacters','mocharheight','pixeldensity','uniquecolours','sectionheight','sectionwidth','class','parentPDF']
#NAMES2=['numofcharacters','mocharheight','pixeldensity','uniquecolours','sectionheight','sectionwidth','relitivelocation','class','parentPDF']
def predictmatrix(names,file,ignore,graph=False,size=0.2):
    dataset = read_csv(file, names=names)
    dataset.describe()
    # Split-out validation dataset
    array = dataset.values
    cutoff = len(names)-ignore
    x = array[:,:cutoff]
    y = array[:,cutoff]
    #print(x)
    #print(y)
    x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=size, random_state=1)
    #model = linear_model.LogisticRegression()#(gamma='auto')
    model=GaussianNB()
    model.fit(x_train, y_train)
    #predictions = model.predict(X_validation)
    #print(accuracy_score(Y_validation, predictions))
    #print(confusion_matrix(Y_validation, predictions))
    #print(classification_report(Y_validation, predictions))
    #print(X_validation)
    if graph:
        sns.pairplot(dataset)
        plt.show()
    return model #accuracy_score(Y_validation, predictions), confusion_matrix(Y_validation, predictions), classification_report(Y_validation, predictions)
    
def datacontroller(name):
    with open (os.path.join(os.path.dirname(__file__),"sectionimg\\data2.csv"),'r') as f:
        for line in f.readlines():
            if name+"\n" in line:
                return True
        return False
    
def removedata(name,file):
    backupfile=file.strip(".csv")+"backup.csv"
    shutil.copyfile(file, backupfile)
    try:
        with open(backupfile,'r') as f:
            lines=f.readlines()
        with open(file,'w') as f:
            for line in lines:
                if name+"\n" not in line:
                    f.write(line)
    # Move the file pointer to the beginning of the file
            #print("done")
        os.remove(file.strip(".csv")+"backup.csv")
        return 1
    except Exception as e:
        print(e)
        shutil.copyfile(file.strip(".csv")+"backup.csv", file)
        os.remove(file.strip(".csv")+"backup.csv")
        return 0    
        
if __name__ == '__main__':
    
    #file = os.path.join(os.path.dirname(__file__),"sectionimg/","data.csv")
    #names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    #names=['numofcharacters','mocharheight','pixeldensity','uniquecolours','sectionheight','sectionwidth','class','parentPDF']
    #predictmatrix(names,file,2,graph=True)
    #removedata("SSRN-id3052318","E:\Lvl4Project\ScientificEvidenceReviewApplication\server\sectionimg\data2.csv")
    print("done")