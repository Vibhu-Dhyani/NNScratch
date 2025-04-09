import numpy as np


class RNN:
    def __init__(self, output_size, input_size, hidden_size=64):
        self.Why = np.random.randn(output_size, hidden_size) / 1000
        self.Whh = np.random.randn(hidden_size, hidden_size) / 1000
        self.Wxh = np.random.randn(hidden_size, input_size) / 1000
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, input):
        h = np.zeros((64, 1))

        # Caching for backprop
        self.hMap = {0: h}
        self.last_input = input

        for i, a in enumerate(input):
            h = np.tanh(self.Whh @ h + self.Wxh @ a + self.bh)
            self.hMap[i] = h
        y = self.Why @ h + self.by
        return y, h

    def backprop(self, dl_dy, learningRate = 2e-2):

        # dl_dWhy
        dl_dWhy = self.hMap[len(hMap)-1] * dl_dy

        # dl_dby
        dl_dby = dl_dy

        # dl_dWhh = dl_dy * sum(dy_ht) * dht_dWhh 
        # calc sum (dy_dht) = dy/dht+1 (1-ht^2)Whh
        dl_dht = self.Why
        for h in reversed(list(self.hMap.keys())) :
            
