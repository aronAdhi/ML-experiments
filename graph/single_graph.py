import numpy as np
import matplotlib.pyplot as plt

x = np.arange(10)
y = x**2 + 3*x + 6

#plt.plot(x,y,marker='o')
plt.plot(x,y,marker='o')

plt.title("Graph Title")
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.xlim(0,50)
plt.ylim(0,50)
plt.grid(True)

plt.show()