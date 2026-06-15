import numpy as np
import matplotlib.pyplot as plt
import math

def create_dataset(m=2, b=3):
    #x = np.random.uniform(-2, 2, 500)
    x = np.linspace(-2,2,500)
    y = m*x + b
    data = np.c_[x,y]
    np.random.shuffle(data)
    return data
    
def train(data):
    weight = 5
    bias = 3
    learning_rate = 0.01
    weights = []
    biases = []
    
    for tuple in data:
        x = tuple[0]
        y_pred = weight*x + bias
        y_act = tuple[1]
        
        error = (y_pred - y_act)**2
        w_del = 2*(y_pred - y_act)*x*learning_rate
        b_del = 2*(y_pred - y_act)*learning_rate
        
        weight -= w_del
        bias -= b_del
        weights.append(weight)
        biases.append(bias)
    
    return weight, bias, weights, biases

data = create_dataset()
print(data[:5])

m,b, W, B = train(data)
print("weight = ",m)
print("bias = ",b)

plt.plot(W)
plt.plot(B)
plt.show()