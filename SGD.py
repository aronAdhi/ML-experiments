import numpy as np
import matplotlib.pyplot as plt
import math
import random

def give_ax():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    return ax


w_act = 2
b_act = 3
weights = np.array([-10,15])
biases = np.array([17,-30])
learning_rate = 0.01

#x_ = np.random.uniform(-2,2,100)
x_ = np.ones(100)
y_ = x_*w_act + b_act
ax = give_ax()
x_g = np. linspace(-5,5,100)
y_g = np.linspace(-5,5,100)
X,Y = np.meshgrid(x_g,y_g)
choosen_x = 3
choosen_y = 3*w_act + b_act
Z = (choosen_y - X*choosen_x - Y)**2
#ax.plot_surface(X,Y,Z)

for weight in weights:
    for bias in biases:
        combo_color = (random.random(), random.random(), random.random())
        for x,y in zip(x_,y_):
            y_pred = weight*x + bias
            error = (y - y_pred)**2
            ax.scatter3D(weight, bias, error,color=combo_color)
            weight += 2*(y - y_pred)*x*learning_rate
            bias += 2*(y - y_pred)*learning_rate

print("Weight = ",weight)
print("Bias = ",bias)    
ax.set_xlabel("Weight")
ax.set_ylabel("Bias")
ax.set_zlabel("error")
plt.show()
    