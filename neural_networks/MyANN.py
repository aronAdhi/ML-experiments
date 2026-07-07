import numpy as np

class Layer:
    def __init__(self, input_dim, output_dim, activation='relu'):
        """
        Represents a single dense layer in the network.
        """
        self.activation_name = activation
        
        # Initialize weights and biases randomly
        # Weights matrix shape: (input_features, output_neurons)
        self.weights = np.random.randn(input_dim, output_dim) * 0.01
        # Bias vector shape: (1, output_neurons)
        self.biases = np.zeros((1, output_dim))
        
    def _activate(self, Z):
        """Applies the chosen activation function."""
        if self.activation_name == 'relu':
            return np.maximum(0, Z)
        elif self.activation_name == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif self.activation_name == 'linear':
            return Z
        else:
            raise ValueError(f"Activation '{self.activation_name}' not supported yet!")

    def forward(self, X):
        """
        Computes the forward pass for this layer: A = activation(X . W + b)
        """
        self.input = X
        self.Z = np.dot(X, self.weights) + self.biases
        self.output = self._activate(self.Z)
        return self.output


class MyANN:
    def __init__(self):
        """
        A container to stack layers sequentially, similar to tf.keras.Sequential.
        """
        self.layers = []

    def add(self, layer):
        """Adds a layer to the network architecture."""
        self.layers.append(layer)

    def forward(self, X):
        """
        Passes the input through all layers sequentially (Forward Propagation).
        """
        current_input = X
        for layer in self.layers:
            current_input = layer.forward(current_input)
        return current_input

    def summary(self):
        """Prints out the architecture configuration."""
        print("=" * 45)
        print(f"{'Layer Type':<15} | {'Input Dim':<10} | {'Output Dim':<10} | {'Activation':<10}")
        print("=" * 45)
        for i, layer in enumerate(self.layers):
            print(f"Dense_{i+1:<9} | {layer.weights.shape[0]:<10} | {layer.weights.shape[1]:<10} | {layer.activation_name:<10}")
        print("=" * 45)
        
        
        
# 1. Initialize our custom model
model = MyANN()

# 2. Add layers dynamically
# Input layer has 4 features -> Hidden layer 1 has 8 neurons (ReLU)
model.add(Layer(input_dim=4, output_dim=8, activation='relu'))

# Hidden layer 1 (8 neurons) -> Hidden layer 2 has 4 neurons (ReLU)
model.add(Layer(input_dim=8, output_dim=4, activation='relu'))

# Hidden layer 2 (4 neurons) -> Output layer has 1 neuron (Sigmoid for binary classification)
model.add(Layer(input_dim=4, output_dim=1, activation='sigmoid'))

# 3. Print the architecture summary
model.summary()

# 4. Test a forward pass with dummy data (e.g., 2 samples with 4 features each)
dummy_data = np.random.randn(2, 4)
predictions = model.forward(dummy_data)

print("\nSample Input Data:\n", dummy_data)
print("\nGenerated Model Predictions:\n", predictions)