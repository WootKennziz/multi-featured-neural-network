class Layer:
    """A Layer is an individual column of neurons connecting the previous and next layer (or input/output). 
    This is a parent class used to create different types of layers including dense and activation layers."""

    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input): # takes an input from the previous layer and returns an output
        pass

    def backward(self, output_gradient, learning_rate): 
        # takes an output gradient from the layer in front and 
        # returns an input gradient for the previous layer (in terms of how forward propagation looks)
        pass