import numpy as np
import matplotlib.pyplot as plt
import math

def create_dataset(m=2, b=3):
    x = np.arange(100)
    y = m*x + b
    data = np.c_[x,y]
    return data
    
def train(data):
    return 2,3

data = create_dataset()
print(data[:5])

m,b = train(data)
print("weight = ",m)
print("bias = ",b)