# Barrett Marsh
# Professor Mark Fleischer
# JHU EP
# Neural Networks Spring 2023

import numpy as np
import math
from sigmoid import SigmoidActivation
from perceptron import Perceptron
from network import Network
from copy import deepcopy

# Create a neural network program implementing the same FFBP network as last time.
# however...

#   - initial weights for layer 1 nodes are [0.3, 0.3] and layer 2 nodes are [0.8, 0.8]
#   - eta = 1.0, no bias terms
#   - input = [1.0, 2.0], desired output = 0.7
#   - ^^^Ensure the above works^^^

# Then...

#   - Reenable bias training.
#   - Train with I/O pairs: [1,1] --> 0.9
#                           [-1,-1] --> 0.05

# Method 1
#   - Present the first I/O pair
#   - Perform backprop to update the weights
#   - Present the second I/O pair
#   - Perform backprop to update the weights
#       - ^^^The above is one cycle^^^
#   - Perform 15 cycles
#   - Present input values to the network and print out the total error
#       and final weights associated with each input/output pair

# Method 2
#   - Use one I/O pair for 15 iterations of FFBP
#   - Present second I/O pair, run FFBP to update weights for 15 iterations
#   - After training the second I/O pair, present the first I/O input, print
#       the total error and the final weights.
#   - Present the second pair and print the total error.

# Essentially, implement bespoke training loops for the two methods in
# network.py.


def main():
    sigmoid_activation = SigmoidActivation()

    # print("NORMAL TRAIN-----------------------------------------------------")

    # test_net = Network([2, 1],
    #                    [[[0.3, 0.3], [0.3, 0.3]], [[0.8, 0.8]]],
    #                    [[0, 0], [0]],
    #                    sigmoid_activation,
    #                    1.0)

    # test_net.train([1, 2], 0.7, 3, no_bias=True)

    # print(test_net.predict([1, 2]))

    print("BATCH TRAIN------------------------------------------------------")

    test_netb = Network([2, 1],
                        [[[0.3, 0.3], [0.3, 0.3]], [[0.8, 0.8]]],
                        [[0, 0], [0]],
                        sigmoid_activation,
                        1.0)

    test_netb.trainM1_beta([[1, 1], [-1, -1]], [[0.9], [0.05]],
                           1, 15, no_bias=False, v=False)

    predict1 = test_netb.predict([1, 1])
    predict2 = test_netb.predict([-1, -1])

    print('given [1,1]: ', predict1)
    print('BigE = ', 0.5 * (0.9 - predict1[0])**2)
    print('given [-1,-1]: ', predict2)
    print('BigE = ', 0.5 * (0.05 - predict2[0])**2)

    # print("ONLINE TRAIN-----------------------------------------------------")

    # test_netc = Network([2, 1],
    #                     [[[0.3, 0.3], [0.3, 0.3]], [[0.8, 0.8]]],
    #                     [[0, 0], [0]],
    #                     sigmoid_activation,
    #                     1.0)

    # test_netc.trainM2([[1, 1], [-1, -1]], [[0.9], [0.05]],
    #                   1, 14, no_bias=False, v=False)

    # predict1c = test_netc.predict([1, 1])
    # predict2c = test_netc.predict([-1, -1])

    # print('given [1,1]: ', predict1c)
    # print('BigE = ', 0.5 * (0.9 - predict1c[0])**2)
    # print('given [-1,-1]: ', predict2c)
    # print('BigE = ', 0.5 * (0.05 - predict2c[0])**2)

    # Current answers:
    # 0.7274
    # 0.0149
    # 0.5665
    # 0.1334
    # 0.4362
    # 0.1076
    # 0.2699
    # 0.02418 (rounded to 0.242)

    # Updated answers:
    # 0.7010
    # 0.01980
    # 0.3571
    # 0.04714

    #


main()
