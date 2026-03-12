#EXPERIMENT-2
import numpy as np
import matplotlib.pyplot as plt
X = np.array([1, 2, 3]).reshape(-1, 1)
Y = np.array([
    [2, 3],
    [4, 5],
    [6, 7]
])

X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
B = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ Y
Y_pred = X_bias @ B
Error = Y - Y_pred

# -------------------- Single Graph: Regression Lines --------------------
plt.figure(figsize=(8,5))
plt.scatter(X, Y[:,0], color='blue', label='Actual y1')
plt.plot(X, Y_pred[:,0], 'b--', label='Predicted y1')
plt.scatter(X, Y[:,1], color='green', label='Actual y2')
plt.plot(X, Y_pred[:,1], 'g--', label='Predicted y2')
plt.xlabel('X')
plt.ylabel('Y values')
plt.title('Regression Lines for y1 and y2')
plt.legend()
plt.grid(True)
plt.show()

# -------------------- Single Bar Graph: Error Matrix --------------------
x_labels = np.arange(len(X))
plt.figure(figsize=(8,5))
plt.bar(x_labels - 0.15, Error[:,0], width=0.3, label='Error y1', color='blue')
plt.bar(x_labels + 0.15, Error[:,1], width=0.3, label='Error y2', color='green')
plt.xticks(x_labels, ['x=1','x=2','x=3'])
plt.xlabel('Data Points')
plt.ylabel('Error')
plt.title('Error (Residual) Bar Chart')
plt.legend()
plt.grid(True, axis='y')
plt.show()