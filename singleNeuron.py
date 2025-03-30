import numpy as np


# Creating an Activation Function
def activationFunctionSigmoid(x):
    return 1 / (1 + np.exp(-x))


# Creating a class for single Neuron
class Neuron:
    def __init__(self, weight, bias):
        self.weight = weight
        self.bias = bias

    # defining a feed forward
    def feedForward(self, input):
        res = np.dot(input, self.weight) + self.bias
        return activationFunctionSigmoid(res)


"""
weight = np.array([2, 3])
bias = 2

neuron = Neuron(weight, bias)
output = neuron.feedForward(np.array([2, 3]))

print(output)
"""
