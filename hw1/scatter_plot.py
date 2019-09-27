import numpy as np 
import matplotlib.pyplot as plt
from get_data import get_data


def plot(x_data , y_data):
	
	fig = plt.figure()
	
	for i in range(9):
		for j in range(9):
			ax = fig.add_subplot(9 , 9 , j+i*9+1)
			
			g1 = ([x_data[k][i] for k in range(len(x_data)) if(y_data[k] == -1)],
			[x_data[k][j] for k in range(len(x_data)) if(y_data[k] == -1)] )
			g2 = ([x_data[k][i] for k in range(len(x_data)) if(y_data[k] == 0)],
			[x_data[k][j] for k in range(len(x_data)) if(y_data[k] == 0)] )
			g3 = ([x_data[k][i] for k in range(len(x_data)) if(y_data[k] == 1)],
				[x_data[k][j] for k in range(len(x_data)) if(y_data[k] == 1)])
			
			data = (g1 , g2 , g3)
			colors = ("red", "green", "blue")
			groups = ("-1", "0", "1")
			

			for data, color, group in zip(data, colors, groups):
				x, y = data
				ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
				 
	
	plt.show()




def main():
	x_data , y_data = get_data()
	plot(x_data , y_data)

if __name__ == '__main__':
	main()
