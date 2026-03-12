#experiment-4
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# Data
X = np.array([[1,1],
              [2,1],
              [2,3],
              [3,3]])

y = np.array([1,1,-1,-1])

# Train Hard-Margin SVM
model = svm.SVC(kernel='linear', C=1e6)
model.fit(X, y)

# Plot data points
plt.scatter(X[y==1][:,0], X[y==1][:,1], color='blue', label='Class +1')
plt.scatter(X[y==-1][:,0], X[y==-1][:,1], color='red', label='Class -1')

# Plot decision boundary
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = model.decision_function(xy).reshape(XX.shape)

# Decision boundary and margins
ax.contour(XX, YY, Z, levels=[-1, 0, 1],
           colors=['black', 'green', 'black'],
           linestyles=['--', '-', '--'])

# Highlight support vectors
ax.scatter(model.support_vectors_[:,0],
           model.support_vectors_[:,1],
           s=150, linewidth=1,
           facecolors='none',
           edgecolors='k',
           label='Support Vectors')

plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Hard Margin SVM')
plt.legend()
plt.show()

# Print results
print("Weight vector (w):", model.coef_)
print("Bias (b):", model.intercept_)
print("Margin:", 2/np.linalg.norm(model.coef_))