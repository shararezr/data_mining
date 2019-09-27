import numpy as np

from sklearn.model_selection import KFold
from Classifier import classifier
from get_data_APS import get_data
from Feature_Extractor import feature_extractor

def main():
	
	X_train, X_test, y_train, y_test = get_data()
	X_train , X_test = feature_extractor(X_train , X_test) #comment this line to remove PCA
	print(classifier(X_train , y_train , X_test , y_test))

if __name__ == '__main__':
	main()