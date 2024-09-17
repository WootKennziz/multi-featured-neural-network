from activation_class import Activation
from layer import Layer
import numpy as np

class Linear(Activation):
    """Linear Activation Function. Returns the input without changes."""
    def __init__(self):
        linear = lambda x: x
        linear_diff = lambda x: x
        super().__init__(linear, linear_diff)

class ReLU(Activation):
    """ReLU Activation Function. Returns the input if bigger than 0, otherwise returns 0. Usually used to eliminate vanishing gradients issues."""
    def __init__(self):
        relu = lambda x: np.where(x > 0, x, 0)
        relu_diff = lambda x: np.where(x > 0, 1, 0)
        # relu = lambda x: 0 if x <= 0 else x
        # relu_diff = lambda x: 0 if x <= 0 else 1
        super().__init__(relu, relu_diff)


class Sigmoid(Activation):
    """Sigmoid Activation Function. Returns a value between 0 and 1"""
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_diff = lambda x: sigmoid(x) * (1 - sigmoid(x))
        super().__init__(sigmoid, sigmoid_diff)

class Softmax(Layer): # more than just lambda functions, so much easier to just inherit from Layer directly
    """Softmax Activation Function. Returns vector with total sum of 1."""
    def forward(self, input):
        self.output = np.exp(input) / np.sum(np.exp(input))
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        return np.dot((np.identity(np.size(self.output)) - self.output.T) * self.output, output_gradient)


class Tanh(Activation):
    """Tanh Activation Function. Returns a value between -1 and 1."""
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_diff = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_diff)




valid_activations = [Linear, ReLU, Sigmoid, Softmax, Tanh]