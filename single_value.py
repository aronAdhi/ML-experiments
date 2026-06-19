import numpy as np
import matplotlib.pyplot as plt
import math

size = 100
weight = 2
bias = 3
x = np.ones(size)
x *= 2
y = weight*x + bias
data = np.c_[x,y]
errors = []

w = 0
b = 0
lr = 0.01

for d in data:
    x_val = d[0]
    y_val = d[1]
    
    y_pred = w * x_val + b
    error = (y_val - y_pred)**2
    errors.append(error)
    
    w += 2*(y_val - y_pred)*x_val*lr
    b += 2*(y_val - y_pred)*lr
    
plt.plot(errors)
plt.show()    