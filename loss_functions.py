import numpy as np

def mse(y_true, y_pred):
    """MSE (Mean Squared Error) is a loss function which squares the differences between true and predicted values, 
    sums them up and divides them over size of input"""
    return np.mean(np.power(y_true - y_pred, 2))

def mse_diff(y_true, y_pred):
    """The derivative of MSE (Mean Squared Error) used for backpropagation."""
    return 2 * (y_pred - y_true) / np.size(y_true)