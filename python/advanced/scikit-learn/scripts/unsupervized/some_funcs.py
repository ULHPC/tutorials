import os
import datetime
# Library to generate plots
import matplotlib as mpl
# Define Agg as Backend for matplotlib when no X server is running
mpl.use('Agg')
import matplotlib.pyplot as plt
# Importing scikit-learn functions
from sklearn.cluster import  KMeans
from sklearn.metrics.pairwise import pairwise_distances_argmin
from matplotlib.cm import rainbow
# Import the famous numpy library
import numpy as np
# We import socket to have access to the function gethostname()
import socket
import time

# alias to the now function
now = datetime.datetime.now

# To know the location of the python script
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# We decorate (wrap) the kmeans function
# in order to add some pre and post-processing
def kmeans(nbClusters,X,profile):
    # We create a log for the clustering task
    file_path = os.path.join(os.getcwd(),
                             '{0}_C{1:06}'.format(profile,nbClusters))
    #logging will not work from the HPC engines
    #need to write into a file manualy.
    with open(file_path+".log", 'a+') as f:
        f.write('job started on {0}\n'.format(socket.gethostname()))
        f.write('new task for nbClusters='+str(nbClusters)+'\n')

    t0 = now()
    with open(file_path+".log", 'a+') as f:
        f.write('Start clustering at {0}\n'.format(t0.isoformat()))

    # Original scikit-learn kmeans 
    k_means = KMeans(init='k-means++', n_clusters=nbClusters, n_init=100)
    k_means.fit(X)

    # After clustering has been performed, we record information to 
    # the log file

    t1 = now()
    h = (t1-t0).total_seconds()//3600
    m = (t1-t0).total_seconds()//60 - h*60
    s = (t1-t0).total_seconds() -m*60 - h*60
    with open(file_path+".log", 'a+') as f:
        f.write('Finished at {0} after '
                '{1}h {2}min {3:0.2f}s\n'.format(t1.isoformat(),h,m,s))
        f.write('kmeans\n')
        f.write('nbClusters: {0}\n'.format(str(nbClusters)))

    # We sort the centers
    k_means_cluster_centers = np.sort(k_means.cluster_centers_, axis=0)
    # We assign the labels
    k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

    # The previous part is useful in order to keep the same color for
    # the different clustering

    t_batch = (t1 - t0).total_seconds()

    # We generate a plot in 2D
    colors = rainbow(np.linspace(0, 1, nbClusters))
    fig=plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for k, col in zip(range(nbClusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_cluster_centers[k]
        ax.plot(X[my_members, 0], X[my_members, 1], 'w',
            markerfacecolor=col, marker='.')
        ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            markeredgecolor='k', markersize=6)
        ax.set_title('KMeans')
        ax.set_xticks(())
        ax.set_yticks(())
    plt.text(-3.5, 1.8,  'clustering time: %.2fs\ninertia: %f' % (t_batch, k_means.inertia_))
    # We save the figure in png
    plt.savefig(file_path+".png")
    return (nbClusters,k_means.inertia_)
