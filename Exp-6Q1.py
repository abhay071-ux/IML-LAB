import numpy as np
import matplotlib.pyplot as plt


points = np.array([[2,3],[3,4],[6,6],[7,7]])
labels = ['A','B','C','D']


c1 = np.array([2,3])
c2 = np.array([6,6])

iteration = 0

while True:
    iteration += 1
    cluster1 = []
    cluster2 = []

    print(f"\nIteration {iteration}")

  
    for i, p in enumerate(points):
        d1 = np.linalg.norm(p - c1)
        d2 = np.linalg.norm(p - c2)

        if d1 < d2:
            cluster1.append(p)
            print(labels[i], p, "-> Cluster 1")
        else:
            cluster2.append(p)
            print(labels[i], p, "-> Cluster 2")

    cluster1 = np.array(cluster1)
    cluster2 = np.array(cluster2)

    new_c1 = np.mean(cluster1, axis=0)
    new_c2 = np.mean(cluster2, axis=0)

    print("New C1:", new_c1)
    print("New C2:", new_c2)

   
    if np.array_equal(new_c1, c1) and np.array_equal(new_c2, c2):
        break

    c1 = new_c1
    c2 = new_c2

print("\nFinal Clusters:")
print("Cluster 1 Points:", cluster1)
print("Cluster 2 Points:", cluster2)

print("\nFinal Centroids:")
print("C1:", c1)
print("C2:", c2)

print("\nTotal Iterations:", iteration)


plt.scatter(cluster1[:,0], cluster1[:,1], color='blue', label="Cluster 1")
plt.scatter(cluster2[:,0], cluster2[:,1], color='red', label="Cluster 2")

plt.scatter(c1[0], c1[1], color='black', marker='X', s=200, label="Centroid 1")
plt.scatter(c2[0], c2[1], color='green', marker='X', s=200, label="Centroid 2")

for i, txt in enumerate(labels):
    plt.annotate(txt, (points[i][0]+0.05, points[i][1]+0.05))

plt.title("K-Means Clustering")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid()

plt.show()