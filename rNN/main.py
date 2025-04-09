import numpy as np
from createInputs import createInput
from dataset import train_data
from preProcesser import preProcessing
from rnn import RNN


def softmax(x):
    return np.exp(x) / sum(np.exp(x))


# Pre-Process and create a vocab and assing numbers to each word in vocab
idx_to_w, w_to_idx = preProcessing()


# Create our RNN
ourRnn = RNN(2, len(idx_to_w))


# Create Inputs for the train data
for x, y in train_data.items():
    input = createInput(x, len(idx_to_w), w_to_idx)
    target = int(y)

    y, h = ourRnn.forward(input)
    prob = softmax(y)

    # dl_dy
    dl_dy = prob
    dl_dy[target] -= 1

    ourRnn.backprop(dl_dy)
