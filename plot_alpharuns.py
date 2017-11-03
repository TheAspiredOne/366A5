#avery tan, altan, 1392212


import numpy as np
import matplotlib.pyplot as plt

x=[0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0]

data = np.load('alphaexp.npy')
n='\n'
# print data,n, len(data),n, data[0],n
plt.xlabel("Alpha")
plt.ylabel("Avg Number of Steps per Episode")
plt.title("Dyna Maze - with Dyna-Q with variable values of alpha, epsilon = 0.13\n ")
plt.xticks([0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0])
plt.plot(x,data)
plt.show()