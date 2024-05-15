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

# # PCA
# pca = PCA(n_components=2)
# top5_pca = pca.fit_transform(top5_ids)

# t-SNE
tsne = TSNE(n_components=2, random_state=0)
top5_tsne = tsne.fit_transform(top5_ids)

# cluster
kmeans = KMeans(n_clusters=25, random_state=0).fit(top5_tsne) # change with top5_pca if desired
cluster_labels = kmeans.labels_

# Calculate the most common first class in each cluster
cluster_to_styles = {i: [] for i in range(25)}  # 25 clusters
for idx, label in enumerate(cluster_labels):
    first_class = top5_style_ids_total[idx][0]
    cluster_to_styles[label].append(first_class)

most_common_first_classes = {}
for cluster, styles in cluster_to_styles.items():
    if styles:
        most_common_first_classes[cluster] = max(set(styles), key=styles.count)

for cluster, common_style in most_common_first_classes.items():
    cluster_points = top5_tsne[cluster_labels == cluster]
    centroid = np.mean(cluster_points, axis=0)
    plt.text(centroid[0], centroid[1], str(common_style), fontsize=9, ha='center', va='center', 
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='black', boxstyle='round,pad=0.5'))


# plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(top5_tsne[:, 0], top5_tsne[:, 1], c=cluster_labels, alpha=0.5, cmap='viridis')
plt.title('t-SNE of Art Styles + K-means Color Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
