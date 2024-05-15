import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Uses dimensionality reduction to visualize the top n styles in a 2D plot.
# Colors: correct predictions in green and incorrect predictions in red.

with open('gen_data/top5_style_ids_total.pkl', 'rb') as f:
    top5_style_ids_total = pickle.load(f)

with open('gen_data/top5_style_ids_result.pkl', 'rb') as f:
    top5_style_ids_result = pickle.load(f)

top5_ids = np.array(top5_style_ids_total)
top5_res = np.array(top5_style_ids_result)

print(len(top5_ids))
print(len(top5_res))

# PCA (poorer results)
pca = PCA(n_components=2)
top5_pca = pca.fit_transform(top5_ids)

# t-SNE
tsne = TSNE(n_components=2, random_state=0)
top5_tsne = tsne.fit_transform(top5_ids)

# creates a color map
colors = ['green' if res == 1 else 'red' for res in top5_res]

# Plotting t-SNE results with correct in green and incorrect in red
plt.figure(figsize=(8, 6))
scatter = plt.scatter(top5_tsne[:, 0], top5_tsne[:, 1], c=colors, alpha=0.5)
plt.title('t-SNE of Art Styles (Correct: Green, Incorrect: Red)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()


# cluster
kmeans = KMeans(n_clusters=25, random_state=0).fit(top5_tsne) # change with top5_pca if desired
cluster_labels = kmeans.labels_

# plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(top5_tsne[:, 0], top5_tsne[:, 1], c=cluster_labels, alpha=0.5, cmap='viridis')
plt.title('t-SNE of Art Styles + K-means Color Clustering')
plt.grid(True)
plt.show()
