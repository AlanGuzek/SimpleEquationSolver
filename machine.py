from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd
import numpy as np
from dataset import get_dataset
from sklearn.datasets import load_digits


data = get_dataset()
X = data
y = X.drop(X.columns[['Values']], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y)


svm = SVC(C=2, kernel='poly')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(accuracy_score(y_test, y_pred))
