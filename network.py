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
    def __init__(self, shape, init_weights, bias, activation_function, eta=0.5, index=0, input_layer=False, output_layer=False):
        self.shape = shape
        self.weights = init_weights
        self.bias = bias
        self.activation_function = activation_function
        self.eta = eta
        self.nodes = {}
        self.activation_value = 0
        self.index = index
        self.output = 0
        self.input_layer = input_layer
        self.output_layer = output_layer

        # Create all nodes in the layer according to the layer shape
        #       - For layers, shape is a scalar
        for i in range(self.shape):
            self.nodes[i] = perceptron.Perceptron(
                self.weights[i], self.bias[i], self.activation_function, self.eta, input_node=self.input_layer, output_node=self.output_layer)
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
        self.bigE = 0

        for i in range(len(self.shape)):
            input_layer = False
            output_layer = False
            if i == 0:
                input_layer = True
            if i == len(self.shape) - 1:
                output_layer = True
            self.layers[i] = Layer(
                self.shape[i], self.weights[i], self.bias[i], self.activation_function, self.eta, i, input_layer=input_layer, output_layer=output_layer)
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
        bigE = 0.5 * (error**2)
        self.bigE = bigE
        for layer_idx in self.layers.keys():
            for node_idx in self.layers[layer_idx].nodes.keys():
                self.layers[layer_idx].nodes[node_idx].error = error[0]
                self.layers[layer_idx].nodes[node_idx].bigE = bigE[0]

        return

    def train(self, x, y, epochs, no_bias=False, v=True):
        prev_layer_output = []
        layer_output = []

        for i in range(epochs):
            print(i)

            # Feed forward
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

            # Calculate Gradient for entire network (back prop)
            for layer_idx in list(reversed(self.layers.keys())):
                for node_idx in self.layers[layer_idx].nodes.keys():
                    # Error must be updated *first*; otherwise, it'll be zero in the
                    # gradient calculations.
                    # test_node.updateError(
                    #     calcError(test_node.activation_value, desired_value))

                    # Pass weight from 'later' layer to this function?
                    # if the node is an input node, we definitely need the following node's
                    #   weights.
                    # if the node is a hidden node, I'm not sure. If it's and output node,
                    #   there's no 'next' weight to pass in.
                    # if not self.layers[layer_idx].nodes[node_idx].output_node:
                    # self.layers[layer_idx].nodes[node_idx + 1]
                    # **********************************************************************
                    depth = len(self.layers.keys()) - 1
                    truelen = len(self.layers.keys())

                    # Get all of the weights connected to this guy
                    # Use layer_idx - 1 for the 'next' layer (as we're traversing in reverse)
                    # How to handle an output node?
                    connecting_weight = []
                    delta_i = []
                    if self.layers[layer_idx].nodes[node_idx].output_node != True:
                        for connected_node_idx in self.layers[layer_idx + 1].nodes.keys():
                            # Append the corresponding weight for each connected node
                            connecting_weight.append(
                                self.layers[layer_idx + 1].nodes[connected_node_idx].weights[node_idx])
                            delta_i.append(
                                self.layers[layer_idx + 1].nodes[connected_node_idx].delta)

                    self.layers[layer_idx].nodes[node_idx].calcGradient(
                        np.array(connecting_weight), np.array(delta_i), no_bias=no_bias)

                    # **********************************************************************
                    # if self.layers[layer_idx].index != depth:
                    #     # How to determine when a node is connected to another?
                    #     connecting_weight = self.layers[layer_idx +
                    #                                     1].nodes[node_idx].weights
                    # else:
                    #     connecting_weight = False
                    # self.layers[layer_idx].nodes[node_idx].calcGradient(connecting_weight,
                    #                                                     no_bias=no_bias)
                    # **********************************************************************

                    # Do not update weights here!!
                    # self.layers[layer_idx].nodes[node_idx].updateWeights()

                    # print("L" + str(layer_idx) + " Node " + str(node_idx) +
                    #       "------------------------------------")
                    # print("    - Activation Value: ",
                    #       self.layers[layer_idx].nodes[node_idx].activation_value)
                    # print("    - Weights: ",
                    #       self.layers[layer_idx].nodes[node_idx].weights)
                    # print("    - Prediction: ",
                    #       self.layers[layer_idx].nodes[node_idx].activation_value)
                    # print("    - Error: ",
                    #       self.layers[layer_idx].nodes[node_idx].error)
                    # print("    - Bias: ",
                    #       self.layers[layer_idx].nodes[node_idx].bias)

            # After gradient has been calculated for entire network, weights may be updated.
            for layer_idx in self.layers.keys():
                for node_idx in self.layers[layer_idx].nodes.keys():
                    self.layers[layer_idx].nodes[node_idx].updateWeights()
                    if v:
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
                        print("    - BigE: ",
                              self.layers[layer_idx].nodes[node_idx].bigE)
                        print("    - Bias: ",
                              self.layers[layer_idx].nodes[node_idx].bias)
                        print("    - Output? ",
                              self.layers[layer_idx].nodes[node_idx].output_node)
                        print("    - Input? ",
                              self.layers[layer_idx].nodes[node_idx].input_node)

    def trainM1_beta(self, x, y, batch_size, epochs, no_bias=False, v=True):
        prev_layer_output = []
        layer_output = []

        for i in range(epochs):
            print(i)
            # Iterate over the pairs using batch_size
            for k in range(0, len(x), batch_size):

                # Present the first input/output pair

                # Perform the back propagation technique (with weight updates)

                # Present the second input/output pair

                # Perform the back propagation technique (with weight updates)

                # Feed forward
                for layer_idx in self.layers.keys():
                    if layer_idx != 0:
                        prev_layer_output = deepcopy(layer_output)
                    else:
                        prev_layer_output = deepcopy(x[k])
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
                error = np.array(y[k]) - np.array(layer_output)
                # print('layer_output:', layer_output)
                # print('error:', error)
                self.updateError(error)

                # Calculate Gradient for entire network (back prop)
                for layer_idx in list(reversed(self.layers.keys())):
                    for node_idx in self.layers[layer_idx].nodes.keys():

                        depth = len(self.layers.keys()) - 1
                        truelen = len(self.layers.keys())

                        connecting_weight = []
                        delta_i = []
                        if self.layers[layer_idx].nodes[node_idx].output_node != True:
                            for connected_node_idx in self.layers[layer_idx + 1].nodes.keys():
                                # Append the corresponding weight for each connected node
                                connecting_weight.append(
                                    self.layers[layer_idx + 1].nodes[connected_node_idx].weights[node_idx])
                                delta_i.append(
                                    self.layers[layer_idx + 1].nodes[connected_node_idx].delta)

                        self.layers[layer_idx].nodes[node_idx].calcGradient(
                            np.array(connecting_weight), np.array(delta_i), no_bias=no_bias)

                # After gradient has been calculated for entire network, weights may be updated.
                for layer_idx in self.layers.keys():
                    for node_idx in self.layers[layer_idx].nodes.keys():
                        self.layers[layer_idx].nodes[node_idx].updateWeights()
                        if v:
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
                            print("    - BigE: ",
                                  self.layers[layer_idx].nodes[node_idx].bigE)
                            print("    - Bias: ",
                                  self.layers[layer_idx].nodes[node_idx].bias)
                            print("    - Output? ",
                                  self.layers[layer_idx].nodes[node_idx].output_node)
                            print("    - Input? ",
                                  self.layers[layer_idx].nodes[node_idx].input_node)
        return

    def trainM1(self, x, y, batch_size, epochs, no_bias=False, v=True):
        prev_layer_output = []
        layer_output = []

        for i in range(epochs):
            print('Epoch:', i)
            for k in range(0, len(x), batch_size):
                print('Batch:', x[k], y[k])
                checky1 = x[k]
                checky2 = y[k]
                # Feed forward per batch
                for layer_idx in self.layers.keys():
                    if layer_idx != 0:
                        prev_layer_output = deepcopy(layer_output)
                    else:
                        prev_layer_output = deepcopy(x[k])
                    # Clear layer_output for use with the current layer
                    layer_output = []

                    for node_idx in self.layers[layer_idx].nodes.keys():
                        layer_input = prev_layer_output
                        activation_value = self.layers[layer_idx].nodes[node_idx].feedForward(
                            layer_input)
                        # Add current node's activation value to list of input for next layer
                        layer_output.append(activation_value)

                # Update error for the network per batch
                error = np.array(y[k]) - np.array(layer_output)
                self.updateError(error)

                # Calculate Gradient for entire network (back prop); per batch
                for layer_idx in list(reversed(self.layers.keys())):
                    for node_idx in self.layers[layer_idx].nodes.keys():

                        depth = len(self.layers.keys()) - 1
                        truelen = len(self.layers.keys())

                        # Get all of the weights connected to this guy
                        # Use layer_idx - 1 for the 'next' layer (as we're traversing in reverse)
                        # How to handle an output node?
                        connecting_weight = []
                        delta_i = []
                        if self.layers[layer_idx].nodes[node_idx].output_node != True:
                            for connected_node_idx in self.layers[layer_idx + 1].nodes.keys():
                                # Append the corresponding weight for each connected node
                                connecting_weight.append(
                                    self.layers[layer_idx + 1].nodes[connected_node_idx].weights[node_idx])
                                delta_i.append(
                                    self.layers[layer_idx + 1].nodes[connected_node_idx].delta)

                        self.layers[layer_idx].nodes[node_idx].calcGradient(
                            np.array(connecting_weight), np.array(delta_i), no_bias=no_bias)

                # Update weights per batch
                for layer_idx in self.layers.keys():
                    for node_idx in self.layers[layer_idx].nodes.keys():
                        self.layers[layer_idx].nodes[node_idx].updateWeights()
                        if v:
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
                            print("    - BigE: ",
                                  self.layers[layer_idx].nodes[node_idx].bigE)
                            print("    - Bias: ",
                                  self.layers[layer_idx].nodes[node_idx].bias)
                            print("    - Output? ",
                                  self.layers[layer_idx].nodes[node_idx].output_node)
                            print("    - Input? ",
                                  self.layers[layer_idx].nodes[node_idx].input_node)

    def trainM2(self, x, y, batch_size, epochs, no_bias=False, v=True):
        prev_layer_output = []
        layer_output = []
        for k in range(0, len(x), batch_size):
            print('Batch:', x[k], y[k])
            for i in range(epochs):
                checky1 = x[k]
                checky2 = y[k]
                print('Epoch:', i)
                print('Input:', checky1)
                print('Expected Output:', checky2)
                # Feed forward per batch
                for layer_idx in self.layers.keys():
                    if layer_idx != 0:
                        prev_layer_output = deepcopy(layer_output)
                    else:
                        prev_layer_output = deepcopy(x[k])
                    # Clear layer_output for use with the current layer
                    layer_output = []

                    for node_idx in self.layers[layer_idx].nodes.keys():
                        layer_input = prev_layer_output
                        activation_value = self.layers[layer_idx].nodes[node_idx].feedForward(
                            layer_input)
                        # Add current node's activation value to list of input for next layer
                        layer_output.append(activation_value)

                # Update error for the network per batch
                error = np.array(y[k]) - np.array(layer_output)
                self.updateError(error)

                # Calculate Gradient for entire network (back prop); per batch
                for layer_idx in list(reversed(self.layers.keys())):
                    for node_idx in self.layers[layer_idx].nodes.keys():

                        depth = len(self.layers.keys()) - 1
                        truelen = len(self.layers.keys())

                        # Get all of the weights connected to this guy
                        # Use layer_idx - 1 for the 'next' layer (as we're traversing in reverse)
                        # How to handle an output node?
                        connecting_weight = []
                        delta_i = []
                        if self.layers[layer_idx].nodes[node_idx].output_node != True:
                            for connected_node_idx in self.layers[layer_idx + 1].nodes.keys():
                                # Append the corresponding weight for each connected node
                                connecting_weight.append(
                                    self.layers[layer_idx + 1].nodes[connected_node_idx].weights[node_idx])
                                delta_i.append(
                                    self.layers[layer_idx + 1].nodes[connected_node_idx].delta)

                        self.layers[layer_idx].nodes[node_idx].calcGradient(
                            np.array(connecting_weight), np.array(delta_i), no_bias=no_bias)

                # Update weights per batch
                for layer_idx in self.layers.keys():
                    for node_idx in self.layers[layer_idx].nodes.keys():
                        self.layers[layer_idx].nodes[node_idx].updateWeights()
                        if v:
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
                            print("    - BigE: ",
                                  self.layers[layer_idx].nodes[node_idx].bigE)
                            print("    - Bias: ",

                                  self.layers[layer_idx].nodes[node_idx].bias)
                            print("    - Output? ",
                                  self.layers[layer_idx].nodes[node_idx].output_node)
                            print("    - Input? ",
                                  self.layers[layer_idx].nodes[node_idx].input_node)
                print('Iteration ', i+1, ' complete.')

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
