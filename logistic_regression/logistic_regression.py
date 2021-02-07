import numpy as np

class LogisticRegression():
    def __init__(self):
        super().__init__()
        self.theta = None
        self.bias = 0

    def sigmod(self, z):
        ans = 1 / (1 + np.exp(-z))
        return ans

    def fit(self, X, y, n_iter, learning_rate):
        num_of_sample, num_of_feature = X.shape
        self.theta = np.zeros((num_of_feature, 1))

        costs = []

        for i in range(n_iter):
            hx = self.sigmod(np.dot(X, self.theta) + self.bias)
            cost = -np.mean(np.multiply(y, np.log(hx)) + np.multiply(1-y, np.log(1-hx)))

            dTheta = (1 / num_of_sample) * (np.dot(X.T, (y - hx)))
            dBias = (1 / num_of_sample) * np.sum(y - hx)

            self.theta = self.theta + learning_rate * dTheta
            self.bias = self.bias + learning_rate * dBias

            costs.append(cost)

            if 1 % 100 == 0:
                print(f"Cost after iteration {i}, {cost}")
        return self.theta, self.bias, costs
    
    def predict(self, X):
        hx = self.sigmod(np.dot(X, self.theta) + self.bias)
        hx_labels = [1 if elem > 0.5 else 0 for elem in hx]
        return np.array(hx_labels)[:, np.newaxis]
        


    