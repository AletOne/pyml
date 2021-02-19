from logistic_regression import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


np.random.seed(123)

X, y_true =  make_blobs(n_samples=1000, centers=2)

fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:,0], X[:,1], c=y_true)
plt.title("Dataset")
plt.xlabel("First Feature")
plt.ylabel("Second Feature")
plt.show()

y_true = y_true[:, np.newaxis]
X_train, X_test, y_train, y_test =train_test_split(X, y_true)

print(f'Shape X_train: {X_train.shape}')
print(f'Shape y_train: {y_train.shape}')
print(f'Shape X_test: {X_test.shape}')
print(f'Shape y_test: {y_test.shape}')

lr = LogisticRegression()
theta, bias, costs = lr.fit(X_train, y_train, n_iter=500, learning_rate=0.008)

fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(500), costs)
plt.title("Development of cost over training")
plt.xlabel("Number of iterations")
plt.ylabel("Cost")
plt.show()

y_p_train = lr.predict(X_train)
y_p_test = lr.predict(X_test)

print(f"train accuracy: {100 - np.mean(np.abs(y_p_train - y_train)) * 100}%")
print(f"test accuracy: {100 - np.mean(np.abs(y_p_test - y_test))}%")
