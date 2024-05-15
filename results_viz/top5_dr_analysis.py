import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import collections

# Uses dimensionality reduction to visualize the top n styles in a 2D plot.
# Uses k-means clustering to color and label the most common style predictions in each cluster.

with open('gen_data/top5_style_ids_total.pkl', 'rb') as f:
    top5_style_ids_total = pickle.load(f)

with open('code/styles.pkl', 'rb') as f:
    styles = pickle.load(f)

top5_ids = np.array(top5_style_ids_total)

# t-SNE
tsne = TSNE(n_components=2, random_state=0)
top5_tsne = tsne.fit_transform(top5_ids)

# cluster
kmeans = KMeans(n_clusters=25, random_state=0).fit(top5_tsne) # change with top5_pca if desired
cluster_labels = kmeans.labels_


cluster_common_first_index = {}

# find most common #1 prediction
for cluster in range(kmeans.n_clusters):
    indices_in_cluster = np.where(cluster_labels == cluster)[0]
    
    ids_in_cluster = top5_ids[indices_in_cluster]
    first_indices = ids_in_cluster[:, 0]
    
    most_common_first_index = collections.Counter(first_indices).most_common(1)[0][0]
    cluster_common_first_index[cluster] = most_common_first_index

print("most common first index per cluster:", cluster_common_first_index)

most_common_indices = [cluster_common_first_index[i] for i in range(kmeans.n_clusters)]

centroids = np.array([top5_tsne[cluster_labels == i].mean(axis=0) for i in range(kmeans.n_clusters)])






# plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(top5_tsne[:, 0], top5_tsne[:, 1], c=cluster_labels, alpha=0.5, cmap='viridis')
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], styles[most_common_indices[i]], fontsize=12, ha='center', va='center', color='red')
plt.title('t-SNE of Art Styles + K-means Color Clustering')
plt.grid(True)
plt.show()
