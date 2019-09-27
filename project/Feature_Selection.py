import numpy as np
from sklearn.feature_selection import VarianceThreshold

def feature_selector(x_train , x_test):

	vt = VarianceThreshold (0.01)
	vt.fit(x_train)

	return vt.transform(x_train) , vt.transform(x_test)
