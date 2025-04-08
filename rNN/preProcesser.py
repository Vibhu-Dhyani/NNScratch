from dataset import train_data


# Define a Vocab
def preProcessing():
    vocab = list(set([w for text in train_data.keys() for w in text.split(" ")]))
    w_to_idx = {}
    idx_to_w = {}
    for i, a in enumerate(vocab):
        w_to_idx[a] = i
        idx_to_w[i] = a
    return idx_to_w, w_to_idx
