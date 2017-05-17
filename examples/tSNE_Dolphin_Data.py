from tSNE_lib.dataset import Dataset
from tSNE_lib.tSNE_utils import tSNE
from tSNE_lib.plot_embedding import Plot_embedding as Pl
from time import time
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import urllib.request

#file_name, headers = urllib.request.urlretrieve('http://workspace/TIMIT/features/dataset_mfcc.jl') # if file is in the weblink

file ='/home/deva/Desktop/Dolphin Data/Dolphin_data'

# Provide file path, data columns in table and label column

dataset = Dataset.Dataset_in_Table(file,[2,5,7,16],[21])

#os.remove(file_name)

data = dataset.data

labels = dataset.labels

behavior_legends = ['Rest', 'Motion', 'Socialisation', 'Hunting', 'Harassment']

color= ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

per= 30
l_rate= 1000.0
n_samp = 300

indices = np.arange(len(data))
random.shuffle(indices)
chosen_sample = indices[:n_samp]

t0 = time()
o = tSNE(data,chosen_sample,labels=labels,method='exact')
Y = o.output()

np.savetxt("Dolphin_tSNEcoordinates.txt", Y)

title= "Visualizing Dolphin's Sound Data of %i samples in 2D with tSNE"% (n_samp)

subtitle = "Perplexity: %i Learning Rate: %i Time Taken: %i seconds" % (per, l_rate, (time() - t0))

Pl(Y, o.train_labels, behavior_legends, color, "Behavior", title, subtitle)

plt.show()
