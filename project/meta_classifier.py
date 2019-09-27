import numpy as np 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
import keras
from keras.callbacks import EarlyStopping
from sklearn.linear_model import LogisticRegression
from keras.utils import to_categorical
from keras.utils import plot_model


def meta_classifier(x_train  , y_train , x_test , y_test):
	number_of_classes = len(np.unique(y_train))
	y_train = to_categorical(y_train, num_classes=number_of_classes)
	y_test = to_categorical(y_test, num_classes=number_of_classes)

	clf = Sequential([
	    Dense(1024, activation = 'relu', input_shape=(len(x_train[0]),)),
	    Dropout(0.65),
	    Dense(512, activation = 'relu'),
	    Dropout(0.65),
	    Dense(128, activation = 'relu'),
	    Dense(32, activation = 'relu'),
	    Dense(number_of_classes, activation = 'sigmoid')
	])

	clf.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

	clf.fit(x_train , y_train, epochs=200, verbose = 2, batch_size = 512 
		,validation_split= 1/10, callbacks = [EarlyStopping(patience = 25)])

	return clf.evaluate(x_test , y_test)[1]

