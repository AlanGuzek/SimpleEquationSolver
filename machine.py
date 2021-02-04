# Making a model that will recognize digits and arithmetic signs from pixel-dataframe

# Importing
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import BernoulliRBM as RBM

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import check_image as ci

import pickle
import pandas as pd
import numpy as np


# Preparing Data
data = pd.read_pickle("signs_data.pkl")
pca = PCA(n_components=600)
scaler = StandardScaler()
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
# X = pca.fit_transform(X, y)
scaler.fit(X=X)
X = scaler.transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)


# Making model using SVM
# values: list = [0.005]
# for i in values:
#     svm = SVC(C=i, kernel='poly')
#     svm.fit(X_train, y_train)

# Making model using NNC
nnc = MLPClassifier(hidden_layer_sizes=(1000,), activation='relu', solver='adam', max_iter=600, verbose=True, n_iter_no_change=15)
# nnc = MLPClassifier(solver='lbfgs', max_iter=3000)
nnc.fit(X, y)

# # Making model using RBM
# rbm = RBM(n_components=50, learning_rate=0.1, n_iter=30).fit(X_train, y_train)

# Checking accuracy
model = nnc
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
pickle.dump(model, open('model2.pkl', 'wb'))


# Testing on another data:
print(ci.check_image(ci.get_model(), ci.get_binary_image(ci.image_path)))
