from sklearn.naive_bayes import GaussianNB

def classifier(x_train  , y_train , x_test , y_test):
	y_train = y_train.astype(int)
	y_test = y_test.astype(int)
	x_train = x_train[:6000]
	y_train = y_train[:6000]
	x_test = x_test[:1600]
	y_test = y_test[:1600]
	clf = GaussianNB()
	clf.fit(x_train , y_train)
	return clf.score(x_test , y_test)
