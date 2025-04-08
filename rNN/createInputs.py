import numpy as np
from preProcesser import preProcessing


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


idx_to_w, w_to_idx = preProcessing()
# print(idx_to_w)
print(w_to_idx)
vocabsize = len(idx_to_w)
print(createInput("i am very bad", vocabsize, w_to_idx))
