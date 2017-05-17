from tSNE_lib.dataset import Dataset
from tSNE_lib.tSNE_utils import tSNE
from tSNE_lib.plot_embedding import Plot_embedding as Pl
from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import urllib.request

file_name, headers = urllib.request.urlretrieve('http://workspace/TIMIT/features/dataset_mfcc.jl')

dataset = Dataset.import_Timit_Data(file_name,f=1)

os.remove(file_name)

data = dataset.data

labels = dataset.labels

vowels = ['aa', 'ae', 'ah', 'eh', 'ih', 'iy', 'uh', 'uw']

color= ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf']

per= 30
l_rate= 1000.0
n_samp = 100

indices = np.arange(len(data))
random.shuffle(indices)
chosen_sample = indices[:n_samp]

t0 = time()
o = tSNE(data,chosen_sample,labels=labels,method='exact')
Y = o.output()

np.savetxt("tSNEcoordinates.txt", Y)

title= "Visualizing Timit Data(Vowels) of %i samples in 2D with tSNE(MFCC)"% (n_samp)

Pl(Y, o.train_labels, vowels, color, "Vowels", title)

plt.show()
