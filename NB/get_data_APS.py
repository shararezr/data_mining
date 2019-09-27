import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

def get_file_content(file):

	xl_file = pd.read_csv(file)
	content = xl_file.values
	return content


def clean_data(train , test):

	y_train = train[: , 0]
	y_test = test[: , 0]
	y_train [y_train == 'neg'] = 0
	y_test[y_test == 'neg'] = 0 
	y_train [y_train == 'pos'] = 1
	y_test[y_test == 'pos'] = 1
	
	x_train = train[:,1:]
	x_test = test[:,1:]
	x_train[x_train == 'na'] = np.NaN
	x_train = x_train.astype(float)
	x_test[x_test == 'na'] = np.NaN
	x_test = x_test.astype(float)

	imp_mean = SimpleImputer()
	imp_mean.fit(x_train , y_train)
	x_train = imp_mean.transform(x_train)
	x_test = imp_mean.transform(x_test)


	return x_train, x_test, y_train, y_test


def get_data():
	train_content = get_file_content("../aps_failure_training_set.csv")
	test_content = get_file_content("../aps_failure_test_set.csv")
	return clean_data(train_content , test_content)
