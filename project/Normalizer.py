import numpy as np
from sklearn.preprocessing import MinMaxScaler

def normalizer(x_train , x_test):

	minmax = MinMaxScaler()
	minmax.fit(x_train)

	return minmax.transform(x_train) , minmax.transform(x_test)
