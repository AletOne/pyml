import numpy as np

class LinearRegression():
    def __init__(self):
        super().__init__()
        self.theta = None

    def fit(self, X, y):
        temp = np.ones(X.shape[0])
        X = np.c_[temp, X]
        self.theta = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        print("Theta: ", self.theta)

    def predict(self, X):
        temp = np.ones(X.shape[0])
        X = np.c_[temp, X]
        y_predict = X.dot(self.theta)
        return y_predict

def mean_squared_error(y_true, y_predict):
    mse = np.mean(np.power(y_predict - y_true, 2))
    return mse