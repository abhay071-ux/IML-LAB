import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Points
points = np.array([
    [1, 2],  # A
    [2, 1],  # B
    [4, 3],  # C
    [5, 4]   # D
])

# Perform hierarchical clustering
Z = linkage(points, method='single', metric='euclidean')

# Plot dendrogram
dendrogram(Z, labels=['A', 'B', 'C', 'D'])

plt.title("Dendrogram (Agglomerative Clustering)")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()