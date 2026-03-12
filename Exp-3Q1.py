#experiment-3,question-1
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

# Given data
X = np.array([2,3,4]).reshape(-1, 1)
y = np.array([1, 2, 3])

# Ridge Regression model
alpha = 1.0
ridge = Ridge(alpha=alpha)
ridge.fit(X, y)

# Get model parameters
slope = ridge.coef_[0]
intercept = ridge.intercept_

# Print the answer
print("Ridge Regression Results")
print("------------------------")
print(f"Alpha (regularization): {alpha}")
print(f"Slope (w): {slope:.4f}")
print(f"Intercept (b): {intercept:.4f}")
print(f"Regression Equation: y = {slope:.4f}x + {intercept:.4f}")

# Predictions for plotting
X_line = np.linspace(1.5, 4.5, 100).reshape(-1, 1)
y_pred = ridge.predict(X_line)

# Plot
plt.scatter(X, y, color='blue', label='Given Data')
plt.plot(X_line, y_pred, color='red', label='Ridge Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ridge Regression with Answer')
plt.legend()
plt.grid(True)
plt.show()