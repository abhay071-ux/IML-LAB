import numpy as np
from sklearn.naive_bayes import GaussianNB

# dataset
temp = np.array([30,28,15,16,18,35]).reshape(-1,1)
play = np.array(["no","no","yes","yes","yes","no"])

# model
model = GaussianNB()
model.fit(temp, play)

# prediction for temp=20
pred = model.predict([[20]])

print("Prediction:", pred[0])