import numpy as np
import matplotlib.pyplot as plt
import math

def e(x):
    return math.exp(x)

def sig(x):
    return 1/(1+e(-x))

#create data
student_id = np.array([1,2,3,4])
sleep = np.array([0.1,0.3,0.7,0.8])
study = np.array([0.2,0.4,0.8,0.9])
result = np.array([0,0,1,1])

data = np.c_[student_id,sleep,study,result]

weight_1 = 0.1
weight_2 = 0.1
bias = 0.5
epochs = 2000
lr = 0.1
Loss = []

for _ in range(epochs):
    for d in data:
        sleep = d[1]
        study = d[2]
        result = d[3]
        
        z = weight_1*sleep + weight_2*study + bias
        d_w1 = sleep 
        d_w2 = study
        d_b = 1
        
        y_pred = sig(z)
        L = (y_pred - result)**2
        Loss.append(L)
        
        weight_1 -= (y_pred - result) * sig(z)*(1-sig(z)) * d_w1*lr
        weight_2 -= (y_pred - result) * sig(z)*(1-sig(z)) * d_w2*lr
        bias -= (y_pred - result) * sig(z)*(1-sig(z)) * d_b*lr
        
        
plt.plot(Loss)
plt.show()