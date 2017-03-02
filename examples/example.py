from dataset import Dataset
from tSNE_utils import tSNE
from plot_embedding import Plot_embedding as Pl
from time import time
import matplotlib.pyplot as plt
import numpy as np
import random

dataset1 = Dataset('dataset_mfcc.jl',f=1) # Get the dataset by making an object of the Dataset class

dataset2 = Dataset('dataset_chirplet.jl') # Get the dataset by making an object of the Dataset class

data1 = dataset1.data                                                            # Get the data for tSNE

labels1 = dataset1.labels                                                        # Get the labels

data2 = dataset2.data                                                            # Get the data for tSNE

labels2 = dataset2.labels                                                        # Get the labels

vowels = ['aa', 'ae', 'ah', 'eh', 'ih', 'iy', 'uh', 'uw']                      # Legends

color= ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

per= 30                                                                       # Set Perplexity(effective number of neighbours)

l_rate= 1000.0                                                                 # Set learning rate

n_samp = 10000                                                                 # Number of samples

indices = np.arange(len(data1))  # Create an array of indices upto the length of the data

for i in range(1):

    random.shuffle(indices)  # Shuffle the indices randomly for proper unbiased data

    t = indices[:n_samp]

    t0 = time()

    o1 = tSNE(data1,t,labels=labels1,method='exact')

    Y = o1.output()

    mfo= "Dataset-%i_MFCC_tSNEcoordinates.txt"%(i+1)

    np.savetxt(mfo, Y)

    title= "Visualizing Timit Data(Vowels) of %i samples in 2D with tSNE(MFCC)"% (n_samp)

    subtitle= "Dataset: %i Perplexity: %i Learning Rate: %i Time Taken: %i seconds" % (i+1, per, l_rate, (time() - t0))

    Pl(Y, o1.train_labels, vowels, color, title, subtitle)

    t1 = time()

    o2 = tSNE(data2, t, labels=labels2, method='exact')

    Y1 = o2.output()

    cro = "Dataset-%i_Chirplet_tSNEcoordinates.txt" % (i + 1)

    np.savetxt(cro, Y1)

    title1 = "Visualizing Timit Data(Vowels) of %i samples in 2D with tSNE(Chirplet)" % (n_samp)

    subtitle1 = "Dataset: %i Perplexity: %i Learning Rate: %i Time Taken: %i seconds" % (i+1, per, l_rate, (time() - t1))

    Pl(Y1, o2.train_labels, vowels, color, title1, subtitle1)


plt.show()