import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from meta_classifier import meta_classifier
from sklearn.pipeline import Pipeline
from sklearn.utils.random import sample_without_replacement
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier,GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from Feature_Extractor import feature_extractor
from Feature_Selection import feature_selector
from copy import deepcopy

def ensemble_classifier(x_train  , y_train , x_test , y_test):

	
	ensemble_clfs = [
					SVC(C=1.0, gamma='auto', probability = True),
					SVC(C=10.0, gamma='auto', probability = True),
					GaussianNB(),
					LogisticRegression(penalty='l2', C=1.0, solver='lbfgs'),
					LogisticRegression(penalty='l2', C=10, solver='lbfgs'),
					LogisticRegression(penalty='l2', C=100.0, solver='lbfgs'),
					GradientBoostingClassifier(),
					AdaBoostClassifier(n_estimators = 50),
					AdaBoostClassifier(n_estimators = 80,learning_rate = 0.1),
					MLPClassifier(hidden_layer_sizes=(32,16), max_iter=200),
					MLPClassifier(hidden_layer_sizes=(128,32), max_iter=200),
					MLPClassifier(hidden_layer_sizes=(256,128,64,16), max_iter=200),
					RandomForestClassifier(n_estimators=10, max_depth=10),
					RandomForestClassifier(n_estimators=50, max_depth=5),
					RandomForestClassifier(n_estimators=10, criterion = 'entropy', max_depth=10),
					RandomForestClassifier(n_estimators=50, criterion = 'entropy', max_depth=5)	,
					RandomForestClassifier(n_estimators=10, max_depth=5),
					GradientBoostingClassifier(learning_rate=0.1, n_estimators=50),
					GradientBoostingClassifier(learning_rate=0.01, n_estimators=80),
					KNeighborsClassifier(n_neighbors=4),
					KNeighborsClassifier(n_neighbors=16),
					KNeighborsClassifier(n_neighbors=64),					
					]
	
	#ensemble_clfs += deepcopy(ensemble_clfs)

	fit(x_train, y_train, x_test, y_test, ensemble_clfs)
	print("All classifiers are trained ")

	x_train = np.concatenate((x_train, predict(x_train, ensemble_clfs)), axis = 1)
	x_test = np.concatenate((x_test, predict(x_test, ensemble_clfs)), axis = 1)
	return x_train, x_test

def fit(x_train, y_train, x_test, y_test, clfs):
	res = []
	for clf in clfs:
		sampled_x,sampled_y = balanced_subsample(x_train , y_train , 0.1)
		clf.fit(sampled_x,sampled_y)
		y = clf.predict(x_test)

def predict(x, clfs):        
	res = []
	for clf in clfs:
		res.append(clf.predict_proba(x))

	res = np.concatenate(res , axis = 1)
	return res

def balanced_subsample(x,y,subsample_size=1.0):

    class_xs = []
    min_elems = None

    for yi in np.unique(y):
        elems = x[(y == yi)]
        class_xs.append((yi, elems))
        if min_elems == None or elems.shape[0] < min_elems:
            min_elems = elems.shape[0]

    use_elems = min_elems
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs:
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)

        xs.append(x_)
        ys.append(y_)

    xs = np.concatenate(xs)
    ys = np.concatenate(ys)

    return xs,ys