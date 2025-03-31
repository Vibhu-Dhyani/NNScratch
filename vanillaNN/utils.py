import numpy as np


# Creating an Activation Function
def activationFunctionSigmoid(x):
    return 1 / (1 + np.exp(-x))


print(activationFunctionSigmoid(0.8))


# Derrivative of Sigmoid
def derrSigmoid(x):
    fx = activationFunctionSigmoid(x)
    return fx * (1 - fx)


# MSE Loss
def calculateLoss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
