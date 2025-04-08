from createInputs import createInput
from dataset import train_data
from preProcesser import preProcessing
from rnn import RNN
from tabulate import tabulate

# Pre-Process and create a vocab and assing numbers to each word in vocab
idx_to_w, w_to_idx = preProcessing()
print(idx_to_w)
inputs = []
# Create Inputs for the train data
for sentence in train_data.keys():
    inputs.append(createInput(sentence, len(idx_to_w), w_to_idx))

# for i in inputs:
#    print(tabulate(i, tablefmt="fancy_grid"))

# Create our rnn
ourRnn = RNN(2, len(idx_to_w))
print(tabulate(ourRnn.Whh, tablefmt="fancy_grid"))
