import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):

    if max_len is None:
        max_len = 0
        for s in seqs:
            max_len = max(len(s), max_len)

    x = np.full((len(seqs), max_len), pad_value)

    for i, seq in enumerate(seqs):
        x[i, :min(len(seq),max_len)] = seq[:max_len]

    return x