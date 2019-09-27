import numpy as np
from sklearn.metrics import accuracy_score , confusion_matrix


def print_result(y_true , y_pred):
	print("\naccuracy = {:.3f}".format(accuracy_score(y_true , y_pred)))
	
	cm = confusion_matrix(y_true , y_pred , [-1 , 0 , 1])
	print("\nconfusion matrix : \n" + str(cm))
	
	FP = cm.sum(axis=0) - np.diag(cm)
	FN = cm.sum(axis=1) - np.diag(cm)
	TP = np.diag(cm)
	TN = cm.sum() - (FP + FN + TP)

	recall = TP / (TP + FN)
	precision = TP / (TP + FP)
	fmeasure = 2 * recall * precision / (precision + recall)

	print("\n recall : ")
	print(recall)

	print("\n precision : ")
	print(precision)

	print("\n f-measure : ")
	print(fmeasure)
