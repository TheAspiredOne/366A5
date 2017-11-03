#avery tan, altan, 1392212


import numpy as np
import matplotlib.pyplot as plt

x = np.arange(51)

data0 = np.load('dyna_q_agent0.npy')
data1 = np.load('dyna_q_agent5.npy')
data2 = np.load('dyna_q_agent50.npy')
n='\n'
# print data,n, len(data),n, data[0],n
plt.xlabel("Episode")
plt.ylabel("Steps per Episode")
plt.title("Dyna Maze - with Dyna-Q\n ")
plt.plot(x,data0, label = "n = 0")
plt.plot(x,data1, label = "n = 5")
plt.plot(x,data2, label = "n = 50")
plt.legend()
plt.show()