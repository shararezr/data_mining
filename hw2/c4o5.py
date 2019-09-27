import numpy as np
from sklearn.tree import DecisionTreeClassifier

def c4o5_classifier(x_train  , y_train , x_test , y_test):
	
	clf = DecisionTreeClassifier(criterion = 'entropy')
	clf.fit(x_train , y_train)
	return clf.predict(x_test)
