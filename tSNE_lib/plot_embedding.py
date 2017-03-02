import matplotlib.pyplot as plt
import numpy as np


class Plot_embedding:

    # Constructor takes output to plot, labels for the data, labels for the legend to show and colors of the labels
    def __init__(self, Y, labels, legend_labels, colors, title= None, subtitle= None):

        # Feature Scaling
        y_min, y_max = np.min(Y, 0), np.max(Y, 0)
        Y = (Y - y_min) / (y_max - y_min)

        fig = plt.figure()
        plt.suptitle(title, fontsize=14, position=(0.435,0.980))
        ax = fig.add_subplot(111)

        for i in range(0, len(legend_labels)):                                                                 # In this data,labels are from index 0 to 7
            ind = np.where(labels == i)                                                                        # Take all the indexes where label is i.
            ax.scatter(Y[ind, 0], Y[ind, 1], c=colors[i], s=30, marker='x', alpha=0.8, label=legend_labels[i]) # Plot all the elements for the particular label
        plt.xticks([])                                                                                         # Remove the X axis labels
        plt.yticks([])                                                                                         # Remove the Y axis labels
        ax.legend(scatterpoints=1, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)                          # Display the legends
        plt.title(subtitle)
        plt.subplots_adjust(left=0.01, right=0.85, top=0.9, bottom=0.01)
