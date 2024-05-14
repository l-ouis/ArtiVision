import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

with open('gen_data/top5_style_ids_total.pkl', 'rb') as f:
    top5_style_ids_total = pickle.load(f)

top5_ids = np.array(top5_style_ids_total)

# # for testing
# data = np.random.rand(100, 5)
# top5_ids = np.array(data)

# PCA
pca = PCA(n_components=2)
top5_pca = pca.fit_transform(top5_ids)

# t-SNE
tsne = TSNE(n_components=2, random_state=0)
top5_tsne = tsne.fit_transform(top5_ids)

# cluster
kmeans = KMeans(n_clusters=5, random_state=0).fit(top5_tsne) # change with top5_pca if desired
cluster_labels = kmeans.labels_

# plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(top5_tsne[:, 0], top5_tsne[:, 1], c=cluster_labels, alpha=0.5, cmap='viridis')
plt.title('t-SNE of Art Styles + K-means Color Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.colorbar(scatter, label='Cluster Label')
plt.show()
