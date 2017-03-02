from sklearn import mixture
from sklearn.metrics.cluster import normalized_mutual_info_score
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

class Clustering:

    def __init__(self, Y, labels):

        n_components=5
        n_iter = 1000
        covariance_type = 'full'
        init_params='kmeans'
        alpha=1.0
        verbose=0
        dpgmm_model = mixture.BayesianGaussianMixture(n_components=n_components, covariance_type=covariance_type, tol=0.001, reg_covar=1e-06,
                                max_iter=n_iter, n_init=1, init_params=init_params, weight_concentration_prior_type='dirichlet_process',
                                weight_concentration_prior=None, mean_precision_prior=None, mean_prior=None, degrees_of_freedom_prior=None,
                                covariance_prior=None, random_state=None, warm_start=False, verbose=0, verbose_interval=10)

        dpgmm_model.fit(Y)

        res = dpgmm_model.predict(Y)
        absisse=Y[:,0]
        ordonnee=Y[:,1]
        l_index=labels

        leg=['Slow  Rest', 'Slow  Motion', 'Slow  Socialisation', 'Slow  Hunting','Slow  Harassment','Medium  Rest', 'Medium  Motion', 'Medium  Socialisation', 'Medium  Hunting','Medium  Harassment','Fast  Rest', 'Fast  Motion', 'Fast  Socialisation', 'Fast  Hunting','Fast  Harassment']
        colors = ['#e41a1c', '#4daf4a','#984ea3','#ffff33','#377eb8','#ff7f00','#a65628']
        leg1=['Slow  Rest', 'Slow  Socialisation', 'Slow  Hunting','Slow  Harassment','Medium  Motion','Medium  Harassment','Fast  Motion']
        legends = []
        for name, color_name in zip(leg1,colors):
            legends.append(mpatches.Patch(color=color_name, label=name))

        nmi = normalized_mutual_info_score(res,l_index)
        print(nmi)
        markers=['o','x','*','+',"3","v","^","<",">","D","|","d","_"]

        for x, y, z, k in zip(absisse, ordonnee, l_index, res):
            # plt.scatter(x, y, color=colors[c],label=leg[int(z)])
            c= colors[leg1.index(leg[int(z)])]
            plt.scatter(x, y, color=c,marker=markers[k],s=50)

        unique_clusters = np.unique(res)
        form_list = []
        text = []
        for cluster in unique_clusters:
            form,  = plt.plot([], markers[cluster] + "k")
            form_list.append(form)
            text.append("Cluster " + str(cluster))


        #plt.title()
        first_legend = plt.legend(handles=legends, loc=1)
        # Add the legend manually to the current Axes.
        ax = plt.gca().add_artist(first_legend)

        plt.legend(form_list, text, loc=2, numpoints=1)
        print(len(set(res))) # savoir le nombre de cluster trouvé dans res
        #plt.plot(absisse,ordonnee,'ro')
        plt.show()
