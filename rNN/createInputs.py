import numpy as np


def createInput(sentence, vocabsize, w_to_idx):
    # This is a good boy
    # 0     1 2 3    4
    # this should return
    # [1,0,0..] , [0,1,0,0..] , [..]
    input = []
    for word in sentence.split(" "):
        ip_vec = np.zeros((vocabsize, 1))
        ip_vec[w_to_idx[word]] = 1
        input.append(ip_vec)
    return input
