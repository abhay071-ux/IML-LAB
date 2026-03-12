#experiment-3 question-2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from mpl_toolkits.mplot3d import Axes3D

# ===============================
# CASE 1: SINGLE FEATURE
# ===============================
print("\n===== CASE 1: SINGLE FEATURE RIDGE REGRESSION =====")

x = np.array([2, 3, 4]).reshape(-1, 1)
y = np.array([1, 2, 3])

model1 = Ridge(alpha=1.0)
model1.fit(x, y)

b0 = model1.intercept_
b = model1.coef_

y_pred1 = model1.predict(x)

# Print equation
equation1 = f"ŷ = {b0:.4f} + ({b[0]:.4f})x"
print("Ridge Regression Equation:")
print(equation1)

print("Predicted Output:", y_pred1)

# Plot
plt.scatter(x, y, color='blue', label='Actual Data')
plt.plot(x, y_pred1, color='red', label='Ridge Regression')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Ridge Regression (Single Feature)")
plt.legend()
plt.grid(True)
plt.show()


# ===============================
# CASE 2: TWO FEATURES
# ===============================
print("\n===== CASE 2: TWO FEATURE RIDGE REGRESSION =====")

X = np.array([[1, 2],
              [2, 3],
              [3, 4]])
y = np.array([1, 2, 3])

model2 = Ridge(alpha=1.0)
model2.fit(X, y)

b0 = model2.intercept_
b = model2.coef_

y_pred2 = model2.predict(X)

# Print equation
equation2 = f"ŷ = {b0:.4f} + ({b[0]:.4f})x1 + ({b[1]:.4f})x2"
print("Ridge Regression Equation:")
print(equation2)

print("Predicted Output:", y_pred2)

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Actual Data')
ax.plot(X[:, 0], X[:, 1], y_pred2, color='red', label='Ridge Regression')

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("y")
ax.set_title("Ridge Regression (Two Features)")
ax.legend()
plt.show()