import numpy as np
from sklearn.manifold import TSNE
import random

class tSNE:

    # The elements to use for running tSNE


    # The constructor with all the standard default tSNE parameters plus default sample size of 5000
    # It just requires the data(numpy array) as the mandatory input.
    def __init__(self, data,t, n_samples=5000, labels=None, speakers=None, dialects=None, n_components=2, perplexity=30.0, early_exaggeration=4.0, learning_rate=1000.0,
                 n_iter=1000, n_iter_without_progress=30, min_grad_norm=1e-07, metric='euclidean', init='random', verbose=0, random_state=None, method='barnes_hut', angle=0.5 ):

        self.__train_data = 0
        self.__train_labels = 0
        self.__train_speakers = 0
        self.__train_dialects = 0
        self.__tSNE_output = 0
        # indices = np.arange(len(data))                              # Create an array of indices upto the length of the data
        # random.shuffle(indices)                                     # Shuffle the indices randomly for proper unbiased data
        self.__train_data = data[t]               # Select the random data of the desired sample size for running tSNE

        # Take the other information according to the random data selected if user wants it.
        if labels is not None:
            self.__train_labels = labels[t]
        if speakers is not None:
            self.__train_speakers = speakers[t]
        if dialects is not None:
            self.__train_dialects = dialects[t]

        # Run tSNE according to the data provided
        Y = TSNE(n_components, perplexity, early_exaggeration, learning_rate, n_iter, n_iter_without_progress, min_grad_norm, metric, init, verbose, random_state, method, angle)

        self.__tSNE_output = Y.fit_transform(self.__train_data)     # Fit the high dimensional data into the embedded space and assign the output.

    @property
    def train_data(self):
        return self.__train_data

    @property
    def train_labels(self):
        return self.__train_labels

    @property
    def train_speakers(self):
        return self.__train_speakers

    @property
    def train_dialects(self):
        return self.__train_dialects

    def output(self):
        return self.__tSNE_output

