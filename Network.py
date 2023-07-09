import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#v_seti = pd.read_csv('NF-BoT-IoT.csv')
v_seti = pd.read_csv('output.csv')
#v_seti = pd.read_csv('data.csv')


"""
print(v_seti.head())
#len(v_seti)
print(v_seti.shape)
"""
v_seti["IPV4_SRC_ADDR"] = v_seti["IPV4_SRC_ADDR"].str.replace('.','')
v_seti["IPV4_DST_ADDR"] = v_seti["IPV4_DST_ADDR"].str.replace('.','')

v_seti["IPV4_SRC_ADDR"] = v_seti["IPV4_SRC_ADDR"].astype("int64")
v_seti["IPV4_DST_ADDR"] = v_seti["IPV4_DST_ADDR"].astype("int64")

#v_seti["IPV4_DST_ADDR"] = v_seti["IPV4_DST_ADDR"] / 100
#v_seti["IPV4_SRC_ADDR"] = v_seti["IPV4_SRC_ADDR"] / 100
m=v_seti.pop("Attack")
print(v_seti.head())
#len(v_seti)
print(v_seti.shape)
print(v_seti.dtypes)


X=v_seti
y=X.pop("Label")
X.head()
print(X.shape)

print(y.head())
print(y.unique())
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=105)

print(X_train.shape)
print(X_test.shape)
knn=KNeighborsClassifier(n_neighbors=1)
print(knn.fit(X_train, y_train))
print(knn.score(X_test, y_test))
y_pred = knn.predict(X_test)

from sklearn import svm
svmm = svm.SVC(kernel='linear')
svmm.fit(X_train, y_train)
svmm.score(X_test, y_test)


from sklearn.metrics import confusion_matrix
from sklearn import metrics 
cm=metrics.confusion_matrix(y_test, y_pred)
print(cm)

import seaborn as sns
sns.heatmap(cm, annot=True)

import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
 
color = 'white'
matrix = plot_confusion_matrix(knn, X_test, y_test, cmap=plt.cm.Blues)
matrix.ax_.set_title('Confusion Matrix', color=color)
plt.xlabel('Predicted Label', color=color)
plt.ylabel('True Label', color=color)
plt.gcf().axes[0].tick_params(colors=color)
plt.gcf().axes[1].tick_params(colors=color)
plt.show()