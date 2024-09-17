from layer import Layer
import numpy as np

class Dense(Layer):
    """The Dense Layer is a fully (or deeply) connected layer where all neurons are connected to all neurons in the next layer."""
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) # initialize random weights with output and input size to enable broadcasting
        self.bias = np.random.randn(output_size, 1) # initialize random biases with the size of output (for each neuron)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias # Y = W * X + B
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T) # dE_dW = dE_dY * X^t

        # gradient descent
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient

        return np.dot(self.weights.T, output_gradient) # returning the gradient of the error with respect to the input dE_dX = W^t * dE_dY