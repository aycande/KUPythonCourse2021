import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn import model_selection as ms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#read_csv
#np.array
#preprocessing
#no need for onehotencode, concatenate, fittransform
#featureselection
#train and test 
#cross validation for each
#gridsearch for each
#confusionmatrix for each

#--------------------
#SHIBA operations

shiba = pd.read_csv("/Users/denizaycan/Documents/datashiba.csv")


ys=np.array(shiba['open'])
ys=ys.astype('float')
xs = np.array(shiba[['Tweets', 'Google', 'BTCopen']])
xs=xs.astype('float')

sc = preprocessing.StandardScaler()
xs = sc.fit_transform(xs)

xtrain, xtest, ytrain, ytest = ms.train_test_split(xs,ys, test_size=0.33, random_state=42)

kneigh = KNeighborsRegressor()
dtree = DecisionTreeRegressor()
sv = svm.SVR()

K = ms.cross_val_score(kneigh, xtrain, ytrain, cv=5)
Kmean = ms.cross_val_score(kneigh, xtrain, ytrain, cv=5).mean()
Kstd = ms.cross_val_score(kneigh, xtrain, ytrain, cv=5).std()

D = ms.cross_val_score(dtree, xtrain, ytrain, cv=5)
Dmean = ms.cross_val_score(dtree, xtrain, ytrain, cv=5).mean()
Dstd = ms.cross_val_score(dtree, xtrain, ytrain, cv=5).std()

S= ms.cross_val_score(sv, xtrain, ytrain, cv=5)
Smean= ms.cross_val_score(sv, xtrain, ytrain, cv=5).mean()
Sstd= ms.cross_val_score(sv, xtrain, ytrain, cv=5).std()

print("Cross Validation Scores (K=5) Follow as: \nKNeighbors (Results, Mean, STD):", K, Kmean, Kstd, "\nDecisionTree (Results, Mean, STD):", D, Dmean, Dstd , "\nSupport Vector Machine (Results, Mean, STD):", S, Smean, Sstd )

#--------------------
#NEAR operations

near = pd.read_csv("/Users/denizaycan/Documents/datanear.csv")


yn=np.array(near['open'])
yn=yn.astype('float')
xn = np.array(near[['Tweets', 'Google', 'BTCopen']])
xn=xn.astype('float')

sc = preprocessing.StandardScaler()
xn = sc.fit_transform(xn)

xtrain, xtest, ytrain, ytest = ms.train_test_split(xn,yn, test_size=0.33, random_state=42)

kneigh = KNeighborsRegressor()
dtree = DecisionTreeRegressor()
sv = svm.SVR()

K = ms.cross_val_score(kneigh, xtrain, ytrain, cv=5)
Kmean = ms.cross_val_score(kneigh, xtrain, ytrain, cv=5).mean()
Kstd = ms.cross_val_score(kneigh, xtrain, ytrain, cv=5).std()

D = ms.cross_val_score(dtree, xtrain, ytrain, cv=5)
Dmean = ms.cross_val_score(dtree, xtrain, ytrain, cv=5).mean()
Dstd = ms.cross_val_score(dtree, xtrain, ytrain, cv=5).std()

S= ms.cross_val_score(sv, xtrain, ytrain, cv=5)
Smean= ms.cross_val_score(sv, xtrain, ytrain, cv=5).mean()
Sstd= ms.cross_val_score(sv, xtrain, ytrain, cv=5).std()



print("Cross Validation Scores (K=5) Follow as: \nKNeighbors (Results, Mean, STD):", K, Kmean, Kstd, "\nDecisionTree (Results, Mean, STD):", D, Dmean, Dstd , "\nSupport Vector Machine (Results, Mean, STD):", S, Smean, Sstd )


