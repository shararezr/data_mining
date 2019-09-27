import numpy as np 
import matplotlib.pyplot as plt
from get_data import get_data


def main():
	x_data , y_data = get_data()
	plt.figure()
	plt.boxplot(np.concatenate((x_data , y_data.reshape((len(y_data) , 1))) , axis = 1))
	plt.show()

if __name__ == '__main__':
	main()
