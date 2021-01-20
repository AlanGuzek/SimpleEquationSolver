from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


file = pd.read_csv('dataset.csv')
X, y = exec(str(file['Data'])), file['Value']
X_train, X_test, y_train, y_test = train_test_split(X, y)
pass