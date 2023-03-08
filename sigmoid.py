# Barrett Marsh
# Professor Mark Fleischer
# JHU EP
# Neural Networks Spring 2023

import numpy as np
import math

class SigmoidActivation:
    def __init__(self):
        self.name = 'sigmoid'
        return

    def calcActivation(self, activity_value):
        activation_value = 1 / (1 + math.e**(-activity_value))
        return activation_value

    def calcGradient(self, activity_value):
        gradient_value = (1 - self.calcActivation(activity_value))*self.calcActivation(activity_value)
        return gradient_value
