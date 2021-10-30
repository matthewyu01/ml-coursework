import numpy as np
import math

"""
 ==================================
 Problem 3: Neural Network Basics
 ==================================
    Generates a neural network with the following architecture:
        Fully connected neural network.
        Input vector takes in two features.
        One hidden layer with three neurons whose activation function is ReLU.
        One output neuron whose activation function is the identity function.
"""


def rectified_linear_unit(x):
    """ Returns the ReLU of x, or the maximum between 0 and x."""
    if x>0:
        return x
    else:
        return 0

def rectified_linear_unit_derivative(x):
    """ Returns the derivative of ReLU."""
    if x>0:
        return 1
    else:
        return 0

def vector_relu(x):
    x[x < 0] = 0
    return x

def output_layer_activation(x):
    """ Linear function, returns input as is. """
    return x

def output_layer_activation_derivative(x):
    """ Returns the derivative of a linear function: 1. """
    return 1

class NeuralNetwork(NeuralNetworkBase):

    def train(self, x1, x2, y):
        input_values = np.array([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.matmul(self.input_to_hidden_weights,input_values) + self.biases
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input)
        z_output = np.matmul(self.hidden_to_output_weights,hidden_layer_activation)
        activated_output = output_layer_activation(z_output)

        ### Backpropagation ###

        # Compute gradients  #TO DO
        output_layer_error = np.multiply(-1,np.subtract(y,activated_output)) * np.vectorize(output_layer_activation_derivative)(z_output) 
        m_var = np.multiply(self.hidden_to_output_weights.T, np.vectorize(rectified_linear_unit_derivative)(hidden_layer_weighted_input))
        hidden_layer_error = np.multiply(output_layer_error,m_var)

        bias_gradients = hidden_layer_error
        hidden_to_output_weight_gradients = np.matmul(output_layer_error,hidden_layer_activation.T)
        input_to_hidden_weight_gradients = hidden_layer_error * input_values.T

        # Use gradients to adjust weights and biases using gradient descent
        self.biases = self.biases - self.learning_rate*bias_gradients
        self.hidden_to_output_weights = self.hidden_to_output_weights - self.learning_rate*hidden_to_output_weight_gradients
        self.input_to_hidden_weights = self.input_to_hidden_weights - self.learning_rate*input_to_hidden_weight_gradients
        

class NeuralNetwork(NeuralNetworkBase):

    def predict(self, x1, x2):
        input_values = np.array([[x1],[x2]]) # 2 by 1

        # Calculate the input and activation of the hidden layer
        hidden_layer_weighted_input = np.matmul(self.input_to_hidden_weights,input_values) + self.biases
        hidden_layer_activation = np.vectorize(rectified_linear_unit)(hidden_layer_weighted_input)

        output = output_layer_activation(np.matmul(self.hidden_to_output_weights,hidden_layer_activation))
        
        return output

    # Run this to train your neural network once you complete the train method
    def train_neural_network(self):

        for epoch in range(self.epochs_to_train):
            for x,y in self.training_points:
                self.train(x[0], x[1], y)

    # Run this to test your neural network implementation for correctness after it is trained
    def test_neural_network(self):

        for point in self.testing_points:
            print("Point,", point, "Prediction,", self.predict(point[0], point[1]))
            if abs(self.predict(point[0], point[1]) - 7*point[0]) < 0.1:
                print("Test Passed")
            else:
                print("Point ", point[0], point[1], " failed to be predicted correctly.")
                return

x = NeuralNetwork()

x.train_neural_network()

# UNCOMMENT THE LINE BELOW TO TEST YOUR NEURAL NETWORK
# x.test_neural_network()
