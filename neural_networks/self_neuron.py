import numpy as np
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, weight=1, bias=1, act='ReLU', lr=0.01):
        self.weight = weight
        self.bias = bias
        self.act = act
        self.lr = lr
        self.inp = None
        self.act_slp = None

    def pass_fn(self, inp):
        self.inp = inp
        z = self.weight*inp + self.bias
        h = self.act_fn(z)
        return h

    def act_fn(self, inp):
        if(self.act == 'ReLU'):
            h = np.maximum(0, inp)
            self.act_slp = np.where(h>0,1,0)
            return h
        if(self.act == 'Lin'):
            h = inp
            self.act_slp = np.ones_like(h)
            return h
    
    def back_prop(self, blame):
        self.weight -= np.mean(blame * self.act_slp * self.inp * self.lr)
        self.bias -= np.mean(blame * self.act_slp * self.lr)


class N_Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, n):
        self.layers.append(n)

    def forward(self, inp):
        out = inp
        for layer in self.layers:
            out = layer.pass_fn(out)
        return out
    
    def back_prop(self, inp, out):
        temp = inp
        for layer in self.layers:
            temp = layer.pass_fn(temp)
        blame = temp - out
        for layer in reversed(self.layers):
            layer.back_prop(blame)
            blame *= layer.act_slp * layer.weight





n1 = Neuron(3,4,'Lin',0.05)
n2 = Neuron(3,4,'Lin',0.05)

X = np.linspace(-2,2,30)
Y = 2*X + 3


weights = []
biases = []
epochs = 500
fig, (ax,wt,bi) = plt.subplots(1,3,figsize=(15,5))
plt.ion()

for epoch in range(epochs):
    h = n1.pass_fn(X)
    y_pred = n2.pass_fn(h)
    weights.append(n1.weight)
    biases.append(n1.bias)
    e = y_pred - Y
    n2.back_prop(e)
    n1.back_prop(e*n2.act_slp*n2.weight)


    ax.clear()
    ax.plot(X, Y)
    ax.plot(X, y_pred)
    ax.set_title(f"epoch: {epoch}")

    wt.clear()
    wt.plot(weights)

    bi.clear()
    bi.plot(biases)
    #print(weights)

    plt.pause(.1)

plt.ioff()
plt.show()
