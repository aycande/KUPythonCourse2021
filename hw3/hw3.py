"""
Created on Fri Jan 28 19:08:08 2022

@author: denizaycan
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, ShuffleSplit,GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, confusion_matrix


cses= pd.read_csv("../Documents/cses4_cut.csv")
cses.head()
#    D2002        >>> D02     GENDER (binary) categoric
#    D2005        >>> D05     UNION MEMBERSHIP OF RESPONDENT (binary) categoric
#    D2012        >>> D12     SOCIO ECONOMIC STATUS ordered
#    D2022        >>> D21b    NUMBER OF CHILDREN IN HOUSEHOLD UNDER AGE 18 ordered

# Extracting variables 
y = cses['voted'].to_numpy()
y = y.astype(float)


# Extracting gender and union membership as categorical variables
catx = cses[['D2002', 'D2005']].to_numpy()
catx = catx.astype(float)

for i in range(3,10):
	catx[catx==i] = np.nan
    
#Extracting socioeconomic status and number of children as ordered variables
#Range of values differs between them so first process seperately then merge with concatenate

orderx1 = cses['D2012'].to_numpy()
orderx1 = orderx1.astype(float)
for i in range (7,10):
	orderx1[orderx1==i] = np.nan
orderx1= orderx1.reshape(-1,1)

orderx2 = cses[ 'D2022'].to_numpy()
orderx2 = orderx2.astype(float)
for i in range (90,100):
	orderx2[orderx2==i] = np.nan 
orderx2= orderx2.reshape(-1,1)

orderx = np.concatenate((orderx1, orderx2), axis=1)

#Imputation of data to exclude missing values

imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
catx = imp.fit_transform(catx)
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
orderx = imp.fit_transform(orderx)

#Simplifying the variable names
x1= catx
x2= orderx

#Preprocessing to prepare data
#Scaling the ordered variables and use OneHotEncoder for categorical variables
sc = preprocessing.StandardScaler()
x2 = sc.fit_transform(x2)

OHE = preprocessing.OneHotEncoder()
OHE.fit(x1)
x1enc = OHE.transform(x1).toarray()

#Merging datas after cleaning 
X = np.concatenate((x1enc, x2), axis=1)

#Variable selection for optimizing the variable we work with
X = SelectKBest(f_classif, k=6).fit_transform(X, y)

#Cross validation with Shuffle Split. Randomly into 5 subgroups, k=5r
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)  

#After several trials I wanted to see the mean and the standard deviation seperately to differentiate the scores 

#Decision Tree
DT= DecisionTreeClassifier()
DTCmean=cross_val_score(DT, X, y, cv=cv).mean()
DTCstd=cross_val_score(DT, X, y, cv=cv).std()
DTC=cross_val_score(DT, X, y, cv=cv)

#Support Vector Machine
SVM = SVC(probability = True)
SVMAmean=cross_val_score(SVM, X, y, cv=cv).mean()
SVMAstd=cross_val_score(SVM, X, y, cv=cv).std()
SVMA=cross_val_score(SVM, X, y, cv=cv)

#Logistic Regression
LR = LogisticRegression()
LCVmean=cross_val_score(LR, X, y, cv=cv).mean()
LCVstd=cross_val_score(LR, X, y, cv=cv).std()
LCV=cross_val_score(LR, X, y, cv=cv)


#K-Nearest Neighbors
KNN = KNeighborsClassifier()
KCVmean=cross_val_score(KNN, X, y, cv=cv).mean()
KCVstd=cross_val_score(KNN, X, y, cv=cv).std()
KCV=cross_val_score(KNN, X, y, cv=cv)

print("Cross Validation Scores (K=5) Follow as: \nDecision Tree (Mean, STD, Results):", DTCmean, DTCstd, DTC, "\nLogistic Regression(Mean, STD, Results):", LCVmean, LCVstd,LCV , "\nSupport Vector Machine(Mean, STD, Results):", SVMAmean, SVMAstd, SVMA, "\nKNeighbors(Mean, STD, Results):", KCVmean,KCVstd, KCV)

#However their scores did not vary marginally. 
#The scores of SVM and LG are the highest. Therefore I pursued analysis with them further

#Train-test-split operations
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=44)

#Support Vector Machine
#Optimizing with GridSearch

SVM.fit(X_train, y_train)
SVpredict = SVM.predict(X_test)
print("SVM: Accuracy score with default settings is ", accuracy_score(y_test, SVpredict))

SVM = SVC()
parameters = {'C': [1, 10],
              'gamma': [0.001, 0.01, 1]}
grid_SVM = GridSearchCV(estimator=SVM, param_grid=parameters)
grid_SVM.fit(X_train, y_train)

#Summarizing the results
print("SVM: Best optimized score: ",grid_SVM.best_score_)
print("SVM: Best parameters: ", grid_SVM.best_params_)

#Training SVM model and creating the Confusion Matrix
#Improving the accuracy score with CM
SVM = SVC(C= 1, gamma= 0.001)
SVM.fit(X_train, y_train)
SVMpredict = SVM.predict(X_test)
print("SVM: Accuracy score with best settings is ", accuracy_score(y_test, SVpredict))

#Unfortunately there is no score difference between the default and best settings. There may be an overfitting problem. Therefore, it is not necessary to use best settings further.

CM_SVM = confusion_matrix(y_test, SVMpredict)
sns.heatmap(CM_SVM, square=True, annot=True, fmt='d', cbar=False, cmap="YlGnBu")
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title("Confusion Matrix for Support Vector Machine")
plt.show()

#LOGISTIC REGRESSION

LR = LogisticRegression()
LR.fit(X_train, y_train)
LRpredict = LR.predict(X_test)
print("LR: Accuracy score with default settings ", accuracy_score(y_test, LRpredict))

parameters = {'C':[1, 10]}
grid_LR = GridSearchCV(estimator=LR, param_grid=parameters)
y_pred = grid_LR.fit(X_train, y_train).predict(X_test)
# summarize the results of the grid search

print("LR: Best optimized score: ",grid_LR.best_score_)
print("LR: Best parameters: ", grid_LR.best_params_)

#Training the LR model and creating the Confusion Matrix
#Improving the accuracy score with CM

LR= LogisticRegression(C= 1, max_iter=1000)
LR.fit(X_train, y_train)
LRpredict = LR.predict(X_test)
print("LR: Accuracy score with best settings is ", accuracy_score(y_test, LRpredict))

#Unfortunately there is no score difference between the default and best settings. There may be an overfitting problem. Therefore, it is not necessary to use best settings further.
#In fact, the scores are equal to SVM scores too.

CM_LR = confusion_matrix(y_test, y_pred)
sns.heatmap(CM_LR/np.sum(CM_LR), square=True, annot=True, fmt='.2%', cbar=False, cmap='BuPu')
plt.title("Confusion Matrix for Logistic Regression")
plt.xlabel("True Value")
plt.ylabel("Predicted Value")
plt.show()


