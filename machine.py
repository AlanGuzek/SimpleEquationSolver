from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
svm = SVC(kernel='poly', C=2)
svm.fit(X_train, y_train)
y_predict = svm.predict(X_test)
print(accuracy_score(y_predict, y_test))
svm.fit(X, y)
pickle.dump(svm, open("svm_model.sav", "wb"))
