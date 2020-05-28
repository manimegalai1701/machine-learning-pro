import sys
print('Python: {}'.format(sys.version))
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
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
from sklearn import model_selection

#downloading dataset
url="http://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names=['sepal-len','sepal-width','petal-len','petal-width','class']
dataset=read_csv(url,names=names)

#dimensions of dataset
print(dataset.shape)
dataset.head(20)
dataset.describe()
dataset.groupby('class').size()

#plot
dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
plt.show()

dataset.hist()
plt.show()

scatter_matrix(dataset)
plt.show()

array=dataset.values
X=array[:,0:4]
y=array[:,4]
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=1)

models=[]
models.append(('LR', LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results=[]
names=[]
for name, model in models:
  kfold=StratifiedKFold(n_splits=10,random_state=1)
  cv_results=cross_val_score(model,X_train,Y_train,cv=kfold,scoring='accuracy')
  results.append(cv_results)
  names.append(name)
  print('%s: %f (%f)' %(name,cv_results.mean(),cv_results.std()))
  
plt.boxplot(results,labels=names)
plt.title('Alg comparsion')
plt.show()

model=SVC(gamma='auto')
model.fit(X_train,Y_train)
pred=model.predict(X_test)

print(accuracy_score(Y_test,pred))
print(confusion_matrix(Y_test,pred))
print(classification_report(Y_test,pred))
