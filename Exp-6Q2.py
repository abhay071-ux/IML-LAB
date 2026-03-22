import numpy as np
import matplotlib.pyplot as plt

points = np.array([2,4,10,12,3,20,30,11,25])

c1 = 2
c2 = 4

iteration = 0

while True:
    iteration += 1
    cluster1 = []
    cluster2 = []

    print("\nIteration", iteration)

    for p in points:
        d1 = abs(p - c1)
        d2 = abs(p - c2)

        if d1 <= d2:
            cluster1.append(p)
            print(p, "-> Cluster 1")
        else:
            cluster2.append(p)
            print(p, "-> Cluster 2")

    new_c1 = np.mean(cluster1)
    new_c2 = np.mean(cluster2)

    print("New C1:", new_c1)
    print("New C2:", new_c2)

    
    if np.isclose(new_c1, c1) and np.isclose(new_c2, c2):
        break

    c1 = new_c1
    c2 = new_c2


print("\nFinal Clusters")
print("Cluster 1:", cluster1)
print("Cluster 2:", cluster2)

print("\nFinal Centroids")
print("C1:", c1)
print("C2:", c2)

print("\nTotal Iterations:", iteration)

plt.scatter(points, [1]*len(points), color="blue", label="Data Points")

plt.scatter(c1, 1, color="red", marker="X", s=200, label="Centroid 1")
plt.scatter(c2, 1, color="green", marker="X", s=200, label="Centroid 2")

plt.yticks([])
plt.title("1D K-Means Clustering")
plt.xlabel("Data Values")
plt.legend()
plt.grid()

plt.show()