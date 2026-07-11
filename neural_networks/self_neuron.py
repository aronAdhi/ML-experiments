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
            #nxt_blm = blame * layer.act_slp * layer.weight
            nxt_blm = blame
            layer.back_prop(blame)
            blame = nxt_blm
        return temp
        

X = np.linspace(-2,2,30)
Y = 2*X + 7


weights = []
biases = []
error = []

epochs = 500
fig, (ax,e,bi) = plt.subplots(1,3,figsize=(15,5))
plt.ion()

nn = N_Network()
nn.add_layer(Neuron(1,2))
nn.add_layer(Neuron(1,2))
nn.add_layer(Neuron(1,2))

for epoch in range(epochs):
    y_pred = nn.back_prop(X,Y)
    er = np.mean(y_pred - Y)
    error.append(er)

    ax.clear()
    ax.plot(X, Y)
    ax.plot(X, y_pred)
    ax.set_title(f"epoch: {epoch}")

    e.clear()
    e.plot(error)

    plt.pause(.001)

plt.ioff()
plt.show()
