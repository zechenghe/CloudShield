import torch
import numpy as np

class SeqGenerator(object):

    def __init__(self, data):
        self.data = data
        self.l = self.data.shape[0]

    def next(self, Nbatch, subseqlen):

        # Split a long sequence into batches of subsequences for training
        batch = []
        startidx = np.random.randint(low = 0, high = self.l - subseqlen + 1, size = Nbatch)
        for i in range(subseqlen):
            batch.append(self.data[startidx + i, :])

        # Return an array with shape (TimeFrame, NBatch, NFeature)
        return np.array(batch)
