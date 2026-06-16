import numpy as np

x = np.arange(-5,6,1)
y = np.arange(-3,4,1)

X,Y = np. meshgrid(x,y)

Z = X + Y

import matplotlib.pyplot as plt

x_line = np.linspace(-5, 5, 100)
y_line = np.zeros_like(x_line)
z_line = x_line**2 + y_line**2

ax = plt.plot_3d(X, Y, Z)
ax.plot(x_line, y_line, z_line)
plt.show()