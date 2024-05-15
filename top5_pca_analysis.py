import numpy as np
import pickle

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

import collections

with open('gen_data/top5_style_ids_total.pkl', 'rb') as f:
    top5_style_ids_total = pickle.load(f)

styles = [
    'Abstract Art', 'Abstract Expressionism', 'Academicism', 'Action painting', 
    'American Realism', 'Analytical Cubism', 'Analytical Realism', 'Art Brut', 
    'Art Deco', 'Art Informel', 'Art Nouveau (Modern)', 'Art Singulier', 'Automatic Painting', 
    'Baroque', 'Biedermeier', 'Byzantine', 'Cartographic Art', 
    'Chernihiv school of icon painting', 'Classical Realism', 'Classicism', 'Cloisonnism', 
    'Color Field Painting', 'Conceptual Art', 'Concretism', 'Confessional Art', 
    'Constructivism', 'Contemporary', 'Contemporary Realism', 'Coptic art', 
    'Costumbrismo', 'Cretan school of icon painting', 'Crusader workshop', 'Cubism', 
    'Cubo-Expressionism', 'Cubo-Futurism', 'Cyber Art', 'Dada', 'Digital Art', 
    'Divisionism', 'Documentary photography', 'Early Byzantine (c. 330–750)', 
    'Early Christian', 'Early Renaissance', 'Environmental (Land) Art', 'Ero guro', 
    'Excessivism', 'Existential Art', 'Expressionism', 'Fantastic Realism', 'Fantasy Art', 
    'Fauvism', 'Feminist Art', 'Fiber art', 'Figurative Expressionism', 'Futurism', 
    'Galicia-Volyn school', 'Geometric', 'Gongbi', 'Gothic', 'Graffiti Art', 
    'Hard Edge Painting', 'High Renaissance', 'Hyper-Mannerism (Anachronism)', 
    'Hyper-Realism', 'Impressionism', 'Indian Space painting', 'Ink and wash painting', 
    'International Gothic', 'Intimism', 'Japonism', 'Joseon Dynasty', 'Junk Art', 
    'Kinetic Art', 'Kitsch', 'Komnenian style (1081-1185)', 'Kyiv school of icon painting', 
    'Late Byzantine/Palaeologan Renaissance (c. 1261–1453)', 
    'Latin Empire of Constantinople (1204-1261)', 'Lettrism', 'Light and Space', 
    'Lowbrow Art', 'Luminism', 'Lyrical Abstraction', 'Macedonian Renaissance (867–1056)', 
    'Macedonian school of icon painting', 'Magic Realism', 'Mail Art', 
    'Mannerism (Late Renaissance)', 'Maximalism', 'Mechanistic Cubism', 'Medieval Art', 
    'Metaphysical art', 'Middle Byzantine (c. 850–1204)', 'Minimalism', 'Miserablism', 
    'Modernism', 'Modernismo', 'Mosan art', 'Moscow school of icon painting', 'Mozarabic', 
    'Muralism', 'Native Art', 'Naturalism', 'Naïve Art (Primitivism)', 'Neo-Byzantine', 
    'Neo-Concretism', 'Neo-Dada', 'Neo-Expressionism', 'Neo-Figurative Art', 'Neo-Geo', 
    'Neo-Impressionism', 'Neo-Minimalism', 'Neo-Orthodoxism', 'Neo-Pop Art', 'Neo-Rococo', 
    'Neo-Romanticism', 'Neo-Suprematism', 'Neo-baroque', 'Neoclassicism', 'Neoplasticism', 
    'New Casualism', 'New European Painting', 'New Ink Painting', 'New Medievialism', 
    'New Realism', 'New media art', 'Northern Renaissance', 'Nouveau Réalisme', 
    'Novgorod school of icon painting', 'Op Art', 'Orientalism', 'Orphism', 'Outsider art', 
    'P&D (Pattern and Decoration)', 'Performance Art', 'Photorealism', 'Pictorialism', 
    'Pointillism', 'Pop Art', 'Post-Impressionism', 'Post-Minimalism', 
    'Post-Painterly Abstraction', 'Postcolonial art', 'Poster Art Realism', 'Precisionism', 
    'Proto Renaissance', 'Pskov school of icon painting', 'Purism', 'Queer art', 
    'Rayonism', 'Realism', 'Regionalism', 'Renaissance', 'Rococo', 'Romanesque', 
    'Romanticism', 'Safavid Period', 'Severe Style', 'Shin-hanga', 'Site-specific art', 
    'Sky Art', 'Social Realism', 'Socialist Realism', 'Sots Art', 'Spatialism', 
    'Spectralism', 'Street Photography', 'Street art', 'Stroganov school of icon painting', 
    'Stuckism', 'Sumi-e (Suiboku-ga)', 'Superflat', 'Suprematism', 'Surrealism', 
    'Symbolism', 'Synchromism', 'Synthetic Cubism', 'Synthetism', 'Tachisme', 'Tenebrism', 
    'Tonalism', 'Transautomatism', 'Transavantgarde', 'Tubism', 'Ukiyo-e', 'Unknown', 
    'Verism', 'Viking art', 'Vladimir school of icon painting', 'Vologda school of icon painting', 
    'Yaroslavl school of icon painting', 'Yoruba', 'Zen'
]

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


cluster_common_first_index = {}

# find most common #1 prediction
for cluster in range(kmeans.n_clusters):
    indices_in_cluster = np.where(cluster_labels == cluster)[0]
    
    ids_in_cluster = top5_ids[indices_in_cluster]
    first_indices = ids_in_cluster[:, 0]
    
    most_common_first_index = collections.Counter(first_indices).most_common(1)[0][0]
    cluster_common_first_index[cluster] = most_common_first_index

print("Most common first index per cluster:", cluster_common_first_index)

most_common_indices = [cluster_common_first_index[i] for i in range(kmeans.n_clusters)]

centroids = np.array([top5_tsne[cluster_labels == i].mean(axis=0) for i in range(kmeans.n_clusters)])






# plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(top5_tsne[:, 0], top5_tsne[:, 1], c=cluster_labels, alpha=0.5, cmap='viridis')
for i, centroid in enumerate(centroids):
    plt.text(centroid[0], centroid[1], styles[most_common_indices[i]], fontsize=12, ha='center', va='center', color='red')
plt.title('t-SNE of Art Styles + K-means Color Clustering')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()
