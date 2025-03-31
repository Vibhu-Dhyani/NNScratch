import numpy as np

from singleNeuron import Neuron


class singleLayeredNetwork:
    def __init__(self, weights, bias):
        # Initializing neurons for single layered CNN
        self.h1 = Neuron(weights, bias)
        self.h2 = Neuron(weights, bias)
        self.o1 = Neuron(weights, bias)

    def feedForward(self, input):
        h1Op = self.h1.feedForward(input)
        h2Op = self.h2.feedForward(input)
        o1Op = self.o1.feedForward(np.array([h1Op, h2Op]))

        return o1Op


"""
network = singleLayeredNetwork([3, 4], 2)
print(network.feedForward([10, 20])
"""
