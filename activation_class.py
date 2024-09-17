from layer import Layer
import numpy as np

class Activation(Layer):
    """A \"Layer\" which takes an input, does an activation and outputs the same size again. 
    This is a parent class for different activation functions found in activation_functions.py"""

    def __init__(self, activation, activation_diff):
        self.activation = activation # normal activation function
        self.activation_diff = activation_diff # derived version of activation function

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_diff(self.input))