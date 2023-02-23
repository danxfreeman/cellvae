# Import modules.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from umap import UMAP

def knee_plot(data, minPts=10, eps=None):
    nbrs = NearestNeighbors(n_neighbors=minPts).fit(data)
    dist, _ = nbrs.kneighbors(data)
    dist = dist[:, minPts-1]
    dist = np.sort(dist)
    plt.plot(dist)
    if eps:
        plt.axhline(y=eps, linewidth=1, linestyle='dashed', color='k')
    plt.xlabel('Sorted observations')
    plt.ylabel(f'Distance to {minPts}th neighbor')
    plt.show()

def dbscan(data, minPts, eps):
    cluster = DBSCAN(eps=eps, min_samples=minPts).fit(data)
    return cluster.labels_

def umap(data, n_neighbors=15, min_dist=0.1):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    umap = UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    data_umap = umap.fit_transform(data_scaled)
    return data_umap

def plot_umap(data, labels):
    fig, ax = plt.subplots()
    scatter = ax.scatter(data[:, 0], data[:, 1], c=labels, s=10, 
                         cmap='Spectral', linewidth=0.15, edgecolors='black')
    ax.legend(*scatter.legend_elements(), loc='upper right', title='Cluster')
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    ax.set_aspect(1)
    return fig
