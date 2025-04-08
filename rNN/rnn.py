import numpy as np


class RNN:
    def __init__(self, output_size, input_size, hidden_size=64):
        self.Why = np.random.randn(output_size, hidden_size) / 1000
        self.Whh = np.random.randn(hidden_size, hidden_size) / 1000
        self.Wxh = np.random.randn(hidden_size, input_size) / 1000
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self):
        pass
