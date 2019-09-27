from scipy.io import arff
from io import StringIO
import numpy as np

def get_file_content():
	
	file = open("PhishingData.arff", "r")
	content = file.read()
	file.close()
	return content

def clean_data(data):
	x_data, y_data = np.zeros((len(data), len(data[0])-1)) , np.zeros(len(data))

	for i in range(len(data)):
		for j in range(len(data[0])-1):
			x_data[i][j] = int(data[i][j])
	for i in range(len(data)):
		y_data[i] = int(data[i][-1])

	return x_data , y_data

def get_data():

	content = get_file_content()
	f = StringIO(content)
	data, meta = arff.loadarff(f)
	return clean_data(data)
