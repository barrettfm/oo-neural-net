# Barrett Marsh
# Professor Mark Fleischer
# JHU EP
# Neural Networks Spring 2023

import numpy as np
import math
import sigmoid

# Perceptrons
# ------------------------------------------------------------------------------------------
# Class defining a perceptron, to be used as nodes in a
# feed-forward neural network.

# Every node j computes an *activity* value A_j = sum(w_{ij}x_i + theta_j),
# where theta_j is a bias parameter associated with node j. This activity value is sometimes
# referred to as the local field impinging on node j. The output *activation* value of node
# j is a function of the activity value y_j = f(A_j), where the function f(x) is a sigmoidal
# function 1/(1 + e^-x).

# Perceptron Delta Rule
# ------------------------------------------------------------------------------------------
# Goal is to set weights w_{ij} such that the output of node j is close to some desired
# output given a specified input. A simple rule based on minimizing some error value by
# using the method of steepest descent (MOSD) is what we use. The error for node j is
# e_j = d_j - y_j, where d_j is the desired output. For each of m output nodes j,
# 1 ... m in the output layer, define the error associated with each output node j
# using E_j = .5 * e_j^2 = .5 * (d_j - y_j)^2

# Using the concept of steepest descent, modify the weights in proportion to the gradient
# of the error. Thus:
# delta_w_{ij}(t+1) = w_{ij}(t+1) - w_{ij}(t) = eta * (delta_E_j)/(delta_w_{ij}(t))

# In vector form:
# (delta_w_{i1}, delta_w_{i2}, ..., delta_w_{in}) = eta*gradient(E)

# Thus we must evaluate using the chain rule for an output node (see notes),
# giving the *sensitivity* of the error to each weight w_{ij} in the network.


class Perceptron:
    def __init__(self, init_weights, bias, activation_function, eta=0.5, input_node=False, output_node=False):
        self.weights = init_weights
        self.bias = bias
        self.activation_function = activation_function
        self.eta = eta
        self.delta = 0
        self.vector_length = len(init_weights)

        self.delta_weights = [0]*self.vector_length
        self.delta_bias = 0
        self.activity_value = 0
        self.activation_value = 0

        self.index = 0
        self.error = 0

        self.input_node = input_node
        self.output_node = output_node

        # Input is a vector with length equal to that of the weights.
        self.input = [0]*self.vector_length

        # Points to next node (hidden or output).
        # If current node is output node, this remains None.
        self.next_node = None
        return

    # Functions supporting the activation function.
    def getActivation(self):
        self.calcActivation()
        return

    def calcActivation(self):
        self.activation_value = self.activation_function.calcActivation(
            self.activity_value)
        return

    # Set the index for keeping track of different nodes.
    def setIndex(self, index):
        self.index = index
        return

    def calcGradient(self, connecting_weight, delta_i=1, no_bias=False):
        # To get the change in overall error (E_j) over change in a given weight (w_ij),
        # -e_j*(1 - y_j)*y_j*x_i
        # Or, the error for the current node times one minus the node's output, times the node's
        # output times the weight's corresponding input. (One input per weight)

        # Should this be negated (see notes)?
        # scalar * scalar * scalar
        # 15:16 in March 9th office hours
        #   - You're calculating the delta using the original equation and
        #       not factoring in the weight.
        #   - If there were more output layer nodes, or if the next layer
        #       were a hidden layer node, you'd need to do the weighted
        #       sum of the deltas.
        if self.output_node:
            self.delta = self.error * (1 - self.activation_value) * self.activation_value
            self.delta_weights = self.eta * self.delta * np.array(self.input)
        elif self.input_node and len(connecting_weight) != 0:
            # Need to multiply by next node's deltas as well.
            self.delta = (1 - self.activation_value) * self.activation_value * delta_i * connecting_weight
            # self.delta = self.error * \
            #     (1 - self.activation_value) * self.activation_value
            self.delta_weights = self.eta * self.delta * np.array(self.input)
        else:
            # Need to multiply by next node's deltas as well.
            self.delta = (1 - self.activation_value) * self.activation_value * delta_i * connecting_weight
            self.delta_weights = self.eta * self.delta * np.array(self.input)

            # self.delta = self.error * \
            #     (1 - self.activation_value) * self.activation_value
            # self.delta_weights = self.eta * self.delta * \
            #     np.array(self.input) * connecting_weight
        # self.delta_weights = self.eta * self.delta * np.array(self.input) * connecting_weight
        # scalar * scalar
        if no_bias == False:
            self.delta_bias = self.eta * self.delta
        else:
            self.delta_bias = 0

        return

    def updateWeights(self):
        self.weights = self.weights + self.delta_weights
        self.bias = self.bias + self.delta_bias
        return

    def updateError(self, error):
        self.error = error
        return

    # Should this be part of the Perceptron class or the Layer class...?
    def feedForward(self, input):
        # Weight and bias given input; then calculate activation value.
        self.input = input
        self.activity_value = np.multiply(
            input, self.weights).sum() + self.bias
        self.getActivation()

        return self.activation_value


def calcError(prediction, desired_value):
    # Calculates error, assuming this is the output node
    # error = prediction - desired_value
    error = desired_value - prediction
    return error

# Code must be able to:
# - Show initial activation function value (no weight updates), weights = [0.24, 0.88], input = [0.8, 0.9],
#   desired output = 0.95, bias = 0, eta = 5.0

#   0.7279

#
# - Apply same initial conditions to 75 iterations of updating weights and bias. Show updated
#   activation function value.

#   0.9475

# - Using the same initial condition, with desired output = 0.15. Perform 30 iterations of updating
#   weights. Note: for the past two questions, get the activation output *after* updating weights (30/75)
#   times.

#   0.1501

# - Consider the bias a weights with an unchanging input value of 1. If we wanted to update this 'weight,'
#   we can apply MOSD to find dE/dTheta. If the initial condition is input = [2], activation/y = 0.3, and
#   desired output = 0.4, derive the value of dE/dTheta.

# -0.021


def main():
    iterations = int(input("Number of iterations: "))

    # Using self-made sigmoid class; made with the intent
    # of using different activation functions in the future
    sigmoid_activation = sigmoid.SigmoidActivation()

    # Values to mimic FFBP example from lecture
    # desired_value = 0.7
    # bias = 0
    # eta = 1.0
    # test_node = Perceptron([0.3, 0.3], bias, sigmoid_activation, eta)

    # Values for problem 1
    # desired_value = 0.95
    desired_value = 0.15
    bias = 0
    eta = 5.0
    test_node = Perceptron([0.24, 0.88], bias, sigmoid_activation, eta)

    for i in range(iterations):
        print(i)
        stopping_iter = 30

        if i == stopping_iter:
            print('--' + str(stopping_iter) + 'Iterations--')

        test_node.feedForward([0.8, 0.9])

        # Error must be updated *first*; otherwise, it'll be zero in the
        # gradient calculations.
        test_node.updateError(
            calcError(test_node.activation_value, desired_value))
        test_node.calcGradient()
        test_node.updateWeights()

        print("    - Activation Value: ", test_node.activation_value)
        print("    - Weights: ", test_node.weights)
        print("    - Prediction: ", test_node.activation_value)
        print("    - Error: ", test_node.error)

    return
