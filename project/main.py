import numpy as np

from sklearn.model_selection import StratifiedKFold
from ensemble_classifier import ensemble_classifier
from meta_classifier import meta_classifier
from data import get_data
from Feature_Extractor import feature_extractor
from Normalizer import normalizer
from Feature_Selection import feature_selector

def main():
	x_data , y_data , y_data_labels= get_data()
	
	# readmission : 0 , HbA1C : 1 , Diag1 : 2
	y_data = y_data[2]
	print("Data is loaded...")
	
	number_of_folds = 5
	avg_accuracy = 0
	cnt = 1


	kf = StratifiedKFold(n_splits=number_of_folds , shuffle  = True)
	
	for  train_index , test_index in kf.split(x_data, y_data):

		X_train, X_test = x_data[train_index], x_data[test_index]
		y_train, y_test = y_data[train_index], y_data[test_index]
		
		X_train , X_test = feature_extractor(X_train , X_test)  #commnet this line to remove PCA
		X_train, X_test = feature_selector(X_train, X_test)
		X_train, X_test = normalizer(X_train, X_test)
		X_train, X_test = ensemble_classifier(X_train, y_train, X_test, y_test)
		X_train , X_test = feature_extractor(X_train , X_test)  #commnet this line to remove PCA
		X_train, X_test = feature_selector(X_train, X_test)
		X_train, X_test = normalizer(X_train, X_test)

		accuracy = meta_classifier(X_train, y_train, X_test, y_test)
		
		avg_accuracy += accuracy
		print("Accuracy on iteration " + str(cnt) + " = " , accuracy )
		cnt+=1

	avg_accuracy/=number_of_folds

	print("Average accuracy of all iterations  = " , avg_accuracy)


if __name__ == '__main__':
	main()