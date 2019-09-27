import numpy as np
from id3 import Id3Estimator

def id3_classifier(x_train  , y_train , x_test , y_test):
	
	clf = Id3Estimator(prune = True)
	clf.fit(x_train , y_train)
	return clf.predict(x_test)
