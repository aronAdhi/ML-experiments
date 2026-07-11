import numpy as np
import matplotlib.pyplot as plt

class Layer:
    def __init__(self, in_dim, out_dim, lr=0.01, act='ReLU'):
        # Small random initialization is safer than pure randn to prevent exploding gradients
        self.weights = np.random.randn(in_dim, out_dim) * 0.1
        self.bias = np.random.randn(1, out_dim)*0.01
        self.lr = lr
        self.act = act
        self.act_slp = None
        self.inp = None
        self.out = None  # Need to store pre-activation output for derivative

    def pass_fn(self, inp):
        self.inp = inp
        temp = np.dot(inp, self.weights)
        self.out = temp + self.bias
        ans = self.act_fn(self.out)
        return ans

    def act_fn(self, inp):
        if self.act == 'ReLU':
            self.act_slp = np.where(inp > 0, 1.0, 0.0)
            return np.maximum(0, inp)
        if self.act == 'Lin':
            self.act_slp = np.ones_like(inp)
            return inp

    def back_prop(self, blame):
        # 1. Apply activation derivative element-wise
        delta = blame * self.act_slp  
        
        # 2. Compute gradients
        # self.inp is (batch_size, in_dim), delta is (batch_size, out_dim)
        dW = np.dot(self.inp.T, delta)
        db = np.sum(delta, axis=0, keepdims=True)
        
        # 3. Compute next blame before updating weights: (batch_size, in_dim)
        next_blame = np.dot(delta, self.weights.T)
        
        # 4. Update parameters
        self.weights -= self.lr * dW
        self.bias -= self.lr * db
        
        return next_blame

class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, Lyr):
        self.layers.append(Lyr)

    def pass_fn(self, inp):
        ans = inp
        for layer in self.layers:
            ans = layer.pass_fn(ans)
        return ans

    def train_step(self, inp, outp):
        # Forward pass
        y_pred = self.pass_fn(inp)
        
        # Compute loss gradient (MSE derivative: 2 * (y_pred - y) / N)
        # We drop the constant 2 since it gets absorbed by the learning rate
        blame = (y_pred - outp) / inp.shape[0] 
        
        # Backward pass sequentially through layers
        for layer in reversed(self.layers):
            blame = layer.back_prop(blame)
            
        return y_pred

nn = Network()
# Using 'Lin' for the last layer since output Y can be negative and goes up to 7
nn.add_layer(Layer(1, 500, lr=0.01, act='ReLU'))
nn.add_layer(Layer(500, 500, lr=0.01, act='ReLU'))
nn.add_layer(Layer(500, 500, lr=0.01, act='ReLU'))
nn.add_layer(Layer(500, 1, lr=0.01, act='Lin'))

X = np.linspace(-2, 2, 100).reshape(-1, 1)
Y = np.cos(X*3)
epochs = 8000  # Increased slightly to show convergence clearly

fig, ax = plt.subplots()
plt.ion()

for epoch in range(epochs):
    y_pred = nn.train_step(X, Y)

    if epoch % 5 == 0:  # Speed up rendering
        ax.clear()
        ax.plot(X, Y, label='Target', color='black', linestyle='dashed')
        ax.plot(X, y_pred, label='Prediction', color='red')
        ax.set_title(f'Epoch: {epoch}')
        ax.legend()
        plt.pause(0.05)

plt.ioff()
plt.show()
