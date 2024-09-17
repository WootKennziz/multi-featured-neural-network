import time

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
from tqdm.notebook import tqdm_notebook

from activation_functions import valid_activations
from dense_layer import Dense
from loss_functions import mse, mse_diff

class NeuralNetwork:
    def __init__(self):

        self.network_struct = []

        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None
        self.Y_pred = None
        
        self.depth = None
        self.learning_rate = None
        self.no_neurons = None
        self.epochs = None

        # for trained
        self.network_is_trained = False
        self.trained_error_history = np.array([])
        self.total_parameters_amount = 0
        self.total_train_time = 0

        # for tested TODO
        self.network_is_tested = False
        self.network_accuracy_error = None
        self.network_accuracy_mean_squared_error = None
        # self.network_accuracy_percent = None


    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    def init_config(self, no_neurons: list, activations: list, learning_rate: int | float):
        """Initializes configuration."""

        self.network_is_trained = False
        self.network_is_tested = False
        self.total_parameters_amount = 0
        
        self._validate_config(no_neurons, activations, learning_rate)

        self.learning_rate = learning_rate
        self.depth = len(no_neurons)
        self.no_neurons = no_neurons
        self.activations = [activation.__name__ for activation in activations]

        # check if you have fitted data, for first-layer compatability
        if self.X_train is None or self.Y_train is None:
            raise RuntimeError("X and Y train not initialized. Did you forget to run model.fit(X, Y)?")

        # add classes to network struct which is the order the neural network runs in
        for i in range(len(no_neurons)):
            if i == 0: # first layer
                self.network_struct.append(Dense(self.X_train.shape[1], no_neurons[i])) # start with size of X_train columns
            else:
                self.network_struct.append(Dense(no_neurons[i-1], no_neurons[i])) # add with input of last layer to desired output

            self.network_struct.append(activations[i]()) # add activation function between each layer, can be Linear to not apply anything
            
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """Fits.. or assigns input to be accessed in training."""
        self.network_is_trained = False
        self.network_is_tested = False
        self.total_parameters_amount = 0

        self.X_train = X
        self.Y_train = Y

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    def show_stats(self):
        """Shows stats for current neural network configuration

        * Network dimension
        * Network activation functions
        * Network learning rate
        * Network is trained
        * Network is tested

        After training
        * Network train time
        * Network total number of epochs
        * Network total amount of parameters
        * Network loss history exists

        After testing
        * Network Mean Absolute Error
        * Network Mean Squared Error
        # * Network Mean Absolute Percentage Error (%)

        """

        print(
            f"""Current Neural Network Configuration:
              
        Network dimension (layers x neurons): {self.depth}x{self.no_neurons}
        Network activation functions: {self.activations}
        Network learning rate: {self.learning_rate}
        Network is trained: {"Yes" if self.network_is_trained else "No"}
        Network is tested: {"Yes" if self.network_is_tested else "No"}"""
        )

        if self.network_is_trained:
            print(
                f"""
        Network train time: {round(self.total_train_time, 2)}s
        Network total amount of epochs: {self.epochs}
        Network total amount of parameters: {self.total_parameters_amount}
        Network loss history exists: {"Yes" if self.trained_error_history.all() else "No"}
"""
            )

        if self.network_is_tested:
            print(
                f"""
        Network Trained Mean Absolute Error: {self.network_accuracy_error}
        Network Trained Mean Squared Error: {self.network_accuracy_mean_squared_error}
"""
            )
        # Disabled because Y value can be close to zero meaning this gets exponentially higher than expected
        # Network Mean Absolute Percentage Error (%): {round(self.network_accuracy_percent, 2)}%

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    def train(self, epochs: int, print_live_error: bool = False):
        """Train on fitted data"""

        self.network_is_trained = False
        self.network_is_tested = False

        if type(epochs) != int:
            raise TypeError(
                f"non-valid type of epochs `self.epochs`. Type {type(epochs)} were given."
            )

        self.epochs = epochs

        # start timer
        self.total_train_time = 0
        training_start = time.time()

        for i in tqdm_notebook(range(epochs), desc=f"Training..."):
            error = 0
            for x, y in zip (self.X_train, self.Y_train):
                # forward
                output = x
                for layer in self.network_struct:
                    output = layer.forward(output)


                # error
                error += mse(y, output)

                # backward
                grad = mse_diff(y, output)
                for layer in reversed(self.network_struct):
                    grad = layer.backward(grad, self.learning_rate)

            
            error /= len(self.X_train)
            self.trained_error_history = np.append(self.trained_error_history, error)

            if print_live_error:
                print(f"{i + 1}/{epochs}, error={error}")


        training_end = time.time()
        self.total_train_time = training_end - training_start
        self._check_parameters_amount()
        self.network_is_trained = True

    #
    #
    #
    #
    #
    #
    #
    #
    #
    #

    def test(self, X: np.ndarray, Y: np.ndarray):
        """Test provided data on trained network"""
        self.X_test = X
        self.Y_test = Y

        self.Y_pred = np.array([])
        error = 0
        for x, y in zip (self.X_test, self.Y_test):
            # forward
            output = x
            for layer in self.network_struct:
                output = layer.forward(output)

            self.Y_pred = np.append(self.Y_pred, output)

            error += mse(y, output)

        error /= len(self.X_train)

        self.network_is_tested = True

        # self.network_accuracy_percent = mean_absolute_percentage_error(self.Y_test[:, 0, 0], self.Y_pred)
        self.network_accuracy_error = mean_absolute_error(self.Y_test[:, 0, 0], self.Y_pred)
        self.network_accuracy_mean_squared_error = error



    def show_test_results(self, title: str = None):
        if self.X_test.shape[1] == 2:
            # 3D scatter
            fig = plt.figure(figsize = (10, 7.5))
            ax = plt.axes(projection = "3d")

            ax.scatter3D(self.X_test[:, 0, 0], self.X_test[:, 1, 0], self.Y_test, color="blue", label="Y true")
            ax.scatter3D(self.X_test[:, 0, 0], self.X_test[:, 1, 0], self.Y_pred, color="orange", label="Y pred")
            plt.title(title if title is not None else "Comparing Y True and Prediction")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid()
            plt.show()



        elif self.X_test.shape[1] == 1:
            plt.figure(figsize=(10, 7.5))
            plt.scatter(self.X_test, self.Y_test, color="blue", linewidth=2, label="Y True")
            plt.scatter(self.X_test, self.Y_pred, color="orange", linewidth=2, label="Y Pred")
            plt.title(title if title is not None else "Comparing Y True and Prediction")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.legend()
            plt.grid()
            plt.show()
        
        else:
            print(f"Could not plot data with dimensions {self.X_test.shape}")
    #
    #
    #
    #
    #
    #
    #
    #
    #

    def show_trained_error(self, title: str = None):
        """Show trained error graph."""

        err_x = np.arange(0, self.epochs, 1)

        plt.figure(figsize=(10, 7.5))
        plt.plot(err_x, self.trained_error_history, color="red", linewidth=2, label="Error Line")
        plt.title(title if title is not None else "Error (loss) over all epochs.")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid()
        plt.show()

    def print_trained_error(self):
        for i in range(1, len(self.trained_error_history) + 1):
            print(f"{i}/{self.epochs}, error={self.trained_error_history[i - 1]}")

    #
    #
    #
    #
    #
    #
    #
    #
    #
    # internal functions

    def _check_parameters_amount(self):
        """Check how many total parameters exist in the system."""
        self.total_parameters_amount = 0

        
        for i in range(len(self.no_neurons)):
            if i == 0:
                self.total_parameters_amount += (self.X_train.shape[1] * self.no_neurons[i]) + self.no_neurons[i]
            else:
                self.total_parameters_amount += (self.no_neurons[i - 1] * self.no_neurons[i]) + self.no_neurons[i]


    def _validate_config(self, no_neurons: list, activations: list, learning_rate: float | int):
        """Validating configuration to make sure everything """

        # check learning rate
        if type(learning_rate) != float and type(learning_rate) != int:
            raise TypeError(
                f"non-valid type of learning_rate `self.learning_rate`. Type {type(learning_rate)} were given."
            )
            
        # check correct size and type no_neurons and activations
        if type(no_neurons) != list:
            raise TypeError(
                f"non-valid type of no_neurons `self.no_neurons`. Type {type(no_neurons)} were given."
            )
        
        if type(activations) != list:
            raise TypeError(
                f"non-valid type of no_neurons `self.no_neurons`. Type {type(activations)} were given."
            )
        
        if len(no_neurons) != len(activations):
            raise IndexError(f"Incompatible length of no_neurons and activations. Length {len(no_neurons)} and {len(activations)}.")


        # check no neurons values
        for i in range(len(no_neurons)):
            if type(no_neurons[i]) != int:
                raise TypeError(f"non-valid type of no_neurons `self.no_neurons` at index {i}. Type {type(no_neurons[i])} were given.")
        

        # check valid activations
        for i in range(len(activations)):
            if activations[i] not in valid_activations:
                raise RuntimeError(f"non-valid activation function `{activations[i]}` at index {i}.")

