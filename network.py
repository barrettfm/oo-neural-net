# Barrett Marsh
# Professor Mark Fleischer
# JHU EP
# Neural Networks Spring 2023

import numpy as np
import math
import sigmoid
import perceptron
from copy import deepcopy

# Create a simple multi-layer neural network with two input nodes and one output node.

# I__1___
#  \/    \
#  /\     3__Output
# I__2___/

# Weights for node 1: [0.8, 0.5]
# Weights for node 2: [0.1, 0.2]
# Weights for node 3: [0.2, 0.7]
# Input: [1, 3]
# Eta: 0.1
# Desired Output = 0.95
# **Do not include or update biases**

# Goals
# ---------------------
# - Efficiently initialize a network with specified layers, layer sizes (shape), and node weights
# - Communicate data from node to node, ensuring that back-propagation happens correctly


class Layer:
    def __init__(self, shape, init_weights, bias, activation_function, eta=0.5, index=0):
        self.shape = shape
        self.weights = init_weights
        self.bias = bias
        self.activation_function = activation_function
        self.eta = eta
        self.nodes = {}
        self.activation_value = 0
        self.index = index
        self.output = 0

        # Create all nodes in the layer according to the layer shape
        #       - For layers, shape is a scalar
        for i in range(self.shape):
            self.nodes[i] = perceptron.Perceptron(
                self.weights[i], self.bias[i], self.activation_function, self.eta)
            self.nodes[i].index = i

        for key in self.nodes.keys():
            print(
                'L' + str(self.index) + ' Node ' + str(self.nodes[key].index) + ':', self.nodes[key].weights)

        return


class Network:
    def __init__(self, shape, init_weights, bias, activation_function, eta=0.5):
        self.shape = shape
        self.weights = init_weights
        self.bias = bias
        self.activation_function = activation_function
        self.eta = eta
        self.layers = {}

        for i in range(len(self.shape)):
            self.layers[i] = Layer(
                self.shape[i], self.weights[i], self.bias[i], self.activation_function, self.eta, i)
            # self.layers[i].index = i

        for key in self.layers.keys():
            print(
                'Layer ' + str(self.layers[key].index) + ':', self.layers[key].shape)

        # Ex. bias    = [ [1, 2], [0.5] ] for 2 input and 1 output node(s)
        # Ex. shape   = [2, 1] for 2 input and 1 output node(s)
        #       - For networks, shape is a list of all layer shapes
        # Ex. weights = [ [0.8, 0.2], [0.2, 0.7] ]
        #       - Weight shape must correspond to the *previous* layer (unless it's the input)
        #       - For example, a single output node will need two weights if it is receiving
        #           input from two hidden nodes.

        return

    def updateError(self, error):
        # Iterate over every layer and node to update the error
        # Not efficient but...makes sense to me logically
        for layer_idx in self.layers.keys():
            for node_idx in self.layers[layer_idx].nodes.keys():
                self.layers[layer_idx].nodes[node_idx].error = error[0]

        return

    def train(self, x, y, epochs):
        prev_layer_output = []
        layer_output = []

        for i in range(epochs):
            print(i)

            for layer_idx in self.layers.keys():
                if layer_idx != 0:
                    prev_layer_output = deepcopy(layer_output)
                else:
                    prev_layer_output = deepcopy(x)
                # Clear layer_output for use with the current layer
                layer_output = []

                for node_idx in self.layers[layer_idx].nodes.keys():
                    # Give input to the current node
                    # **How should we indicate first input vs. inputs
                    #   between layers?**
                    layer_input = prev_layer_output
                    activation_value = self.layers[layer_idx].nodes[node_idx].feedForward(
                        layer_input)
                    # Add current node's activation value to list of input for next layer
                    layer_output.append(activation_value)

            # Update error for the network
            error = np.array(y) - np.array(layer_output)
            # print('layer_output:', layer_output)
            # print('error:', error)
            self.updateError(error)

            for layer_idx in self.layers.keys():
                for node_idx in self.layers[layer_idx].nodes.keys():
                    # Error must be updated *first*; otherwise, it'll be zero in the
                    # gradient calculations.
                    # test_node.updateError(
                    #     calcError(test_node.activation_value, desired_value))
                    self.layers[layer_idx].nodes[node_idx].calcGradient()
                    self.layers[layer_idx].nodes[node_idx].updateWeights()

                    print("L" + str(layer_idx) + " Node " + str(node_idx) +
                          "------------------------------------")
                    print("    - Activation Value: ",
                          self.layers[layer_idx].nodes[node_idx].activation_value)
                    print("    - Weights: ",
                          self.layers[layer_idx].nodes[node_idx].weights)
                    print("    - Prediction: ",
                          self.layers[layer_idx].nodes[node_idx].activation_value)
                    print("    - Error: ",
                          self.layers[layer_idx].nodes[node_idx].error)

    def predict(self, x):
        for layer_idx in self.layers.keys():
            if layer_idx != 0:
                prev_layer_output = deepcopy(layer_output)
            else:
                prev_layer_output = deepcopy(x)
            # Clear layer_output for use with the current layer
            layer_output = []

            for node_idx in self.layers[layer_idx].nodes.keys():
                # Give input to the current node
                # **How should we indicate first input vs. inputs
                #   between layers?**
                layer_input = prev_layer_output
                activation_value = self.layers[layer_idx].nodes[node_idx].feedForward(
                    layer_input)
                # Add current node's activation value to list of input for next layer
                layer_output.append(activation_value)

        return (layer_output)

