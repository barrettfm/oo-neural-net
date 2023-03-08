# Barrett Marsh
# Professor Mark Fleischer
# JHU EP
# Neural Networks Spring 2023

import numpy as np
import math
import sigmoid
import perceptron
from copy import deepcopy

def main():
    sigmoid_activation = sigmoid.SigmoidActivation()
    # test_net = Network([1, 3, 2, 1],
    #                    [[[0.5, 0.5]], [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], [
    #                        [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5]]],
    #                    [[1, 1], [1, 1, 1], [1, 1], [1]],
    #                    sigmoid_activation,
    #                    0.5)

    test_net = Network([2, 1],
                       [[[0.8, 0.5], [0.1, 0.2]], [[0.2, 0.7]]],
                       [[0, 0], [0]],
                       sigmoid_activation,
                       0.1)

    test_net.train([1, 3], 0.95, 2)

    print(test_net.predict([1, 3]))

    return


main()