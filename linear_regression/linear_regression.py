import numpy as np
import matplotlib.pyplot as plt

class LinearRegression():
    def __init__(self):
        super().__init__()
        self.theta = None

    def fit(self, X, y):
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

    def predict(self, X):
        y_predict = X.dot(self.theta)
        return y_predict

def mean_squared_error(y_true, y_predict):
    mse = np.mean(np.power(y_predict - y_true, 2))
    return mse