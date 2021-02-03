# Making a model that will recognize digits and arithmetic signs from pixel-dataframe

# Importing
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd
import numpy as np


# Preparing Data
data = pd.read_pickle("signs_data.pkl")
pca = PCA(n_components=1200)
scaler = StandardScaler()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# X = pca.fit_transform(X, y)
scaler.fit(X=X)
# X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)


# Making model using SVM

svm = SVC(C=0.023, kernel='poly')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print("SVC accuracy = " + str(accuracy_score(y_test, y_pred)))


# Making model using NNC
# nnc = MLPClassifier(activation='relu', solver='sgd', learning_rate='adaptive')
# # nnc = MLPClassifier(solver='lbfgs', max_iter=3000)
# nnc.fit(X_train, y_train)
# y_pred = nnc.predict(X_test)
# print("NN accuracy = " + str(accuracy_score(y_test, y_pred)))
