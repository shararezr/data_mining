import numpy as np
from id3_classifier import id3_classifier
from get_data_phishing import get_data
from sklearn.model_selection import train_test_split
from print_result import print_result

def main():

	X , y = get_data()
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
	y_pred = id3_classifier(X_train , y_train , X_test , y_test)
	print_result(y_test , y_pred)



if __name__ == '__main__':
	main()