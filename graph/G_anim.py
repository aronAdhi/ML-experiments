import matplotlib.pyplot as plt
import numpy as np

X = np.linspace(-2,5,100)
Y = X*2 + 3

fix, ax = plt.subplots()
plt.ion()

for i in range(10):
    ax.clear()
    Y = X*i + 3
    ax.plot(X, Y)
    ax.set_xlim(-3,6)
    ax.set_ylim(-20,50)
    plt.pause(.5)

plt.ioff()
plt.show()
