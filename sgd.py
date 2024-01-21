import numpy as np


class Node:
    def __init__(self):
        self.inputs = []
        self.output = None

    def forward(self):
        # Must implement forward pass
        raise NotImplementedError
    
    def backward(self):
        # Implement backpropagation in the subclasses
        pass


class Add(Node):
    def forward(self, x, y):
        self.inputs = [x, y]
        self.output = x + y
        return self.output
    
    def backward(self, output_gradient):
        input_gradient = output_gradient
        biases_gradient = np.sum(output_gradient, axis=0)
        return input_gradient, biases_gradient


class MatMul(Node):
    def forward(self, x, w):
        self.inputs = [x, w]
        self.output = np.dot(x, w)
        return self.output
    
    def backward(self, output_gradient):
        # Reshape the output_gradient to 2D if it's a 1D array
        if output_gradient.ndim == 1:
            output_gradient = output_gradient.reshape(1, -1)

        input_gradient = np.dot(output_gradient, self.inputs[1].T)
        weights_gradient = np.dot(self.inputs[0].T, output_gradient)

        # Ensure the gradients have the correct shape        
        if input_gradient.shape[0] == 1:
            input_gradient = input_gradient.flatten()

        return input_gradient, weights_gradient


class ReLU(Node):
    def forward(self, x):
        self.inputs = [x]
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, output_gradient):
        input_gradient = output_gradient * (self.inputs[0] > 0)
        return input_gradient


class SimpleDenseLayer:
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        # Using Xavier initialization
        stddev = np.sqrt(2 / (input_size + output_size))
        self.weights = np.random.normal(0, stddev, (input_size, output_size))
        self.biases = np.random.randn(output_size)

        self.matmul = MatMul()
        self.add = Add()
        self.relu = ReLU()

    def forward(self, x):
        x = self.matmul.forward(x, self.weights)
        x = self.add.forward(x, self.biases)
        x = self.relu.forward(x)
        return x
    
    def backward(self, output_gradient, learning_rate):
        relu_gradient = self.relu.backward(output_gradient)
        print(relu_gradient, output_gradient)
        add_gradient, biases_gradient = self.add.backward(relu_gradient)
        print(add_gradient, biases_gradient)
        input_gradient, weights_gradient = self.matmul.backward(add_gradient)

        # Update weights and biases
        # print('here', self.weights, learning_rate, weights_gradient)
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * biases_gradient

        return input_gradient
    

class NeuralNetwork:
    def __init__(self, layer_sizes):
        """
        Initialize the neural network with the given layer sizes.

        :param layer_sizes: A list of integers where each represents the
        number of neurons in that layer.
        """
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(SimpleDenseLayer(layer_sizes[i], layer_sizes[i+1]))

    def forward_pass(self, x):
        """
        Perform a forward pass through the network.

        :param x: Input tensor
        :return: Output tensor after passing through the network
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward_pass(self, output_gradient, learning_rate=0.01):
        gradient_norm = np.linalg.norm(output_gradient)
        max_norm = 5.0

        if gradient_norm > max_norm:
            output_gradient = output_gradient * max_norm / gradient_norm

        i = 0
        for layer in reversed(self.layers):
            print('here: ', i)
            output_gradient = layer.backward(output_gradient, learning_rate)
            i += 1
        
        return output_gradient


def compute_loss_and_gradient(predictions, targets):
    """
    This function utilizes binary cross-entropy loss for
    a classification problem.
    """
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)

    loss = -np.mean(targets * np.log(predictions) + (1 - targets) * np.log(1 - predictions))
    # derivative of loss function with resp. to predictions to get gradient
    loss_gradient = -(targets / predictions) + (1 - targets) / (1 - predictions)
    return loss, loss_gradient


def train(network, data, targets, epochs, learning_rate, batch_size):
    for epoch in range(epochs):
        # Create mini-batches
        permutation = np.random.permutation(len(data))
        data_shuffled = data[permutation]
        targets_shuffled = targets[permutation]

        for i in range(0, len(data), batch_size):
            batch_data = data_shuffled[i:i + batch_size]
            batch_targets = targets_shuffled[i:i + batch_size]

            predictions = network.forward_pass(batch_data)

            # Compute loss and gradient
            loss, loss_gradient = compute_loss_and_gradient(predictions, batch_targets)

            network.backward_pass(loss_gradient, learning_rate)

    return network


layer_sizes = [3, 5, 4, 5]
nn = NeuralNetwork(layer_sizes)

# Example input
with open('dataset.txt', 'r') as file:
    items = file.readlines()
    training_data = np.array([item.split(',') for item in [i.strip() for i in items]], dtype=float)

with open('labels.txt', 'r') as file:
    items = file.readlines()
    training_targets = np.array([item.split(',') for item in [i.strip() for i in items]], dtype=float)

training_targets += 1
# I noticed that a large epoch and larger training rate gives a lot of
# NaNs and that the individual gradient elements were increasing in size.
# I researched and found that this might be a phenomenon called "exploding gradients"
# and applied some tuning and improvements to the loss function and weight randomization.
trained_nn = train(nn, training_data, training_targets, epochs=10, learning_rate=0.0001, batch_size=32)
