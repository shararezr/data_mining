import numpy as np
from sklearn.decomposition import PCA


def feature_extractor(x_train , x_test):

	pca = PCA (n_components = 0.99 ,  svd_solver = "full")
	pca.fit(x_train)

	return pca.inverse_transform(pca.transform(x_train)) , pca.inverse_transform(pca.transform(x_test))	
