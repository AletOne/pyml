import numpy as np
np.random.seed(13)

class SoftmaxRegression():
    def __init__(self):
        super().__init__()
        self.theta = None
        self.bias = None
        self.n_class = 0

    def softmax(self, Z):
        exp = np.exp(Z)
        sum_exp = np.sum(np.exp(Z), axis=1, keepdims=True)
        softmax = exp / sum_exp
        return softmax

    def one_hot(self, y):
        oh = np.zeros((self.n_sample, self.n_class))
        oh[np.arange(self.n_sample), y.T] = 1 #must set = 1 to get one hot!!!!!!!!
        return oh

    def fit(self, X, y_true, n_class, n_iterator=200, learning_rate=0.008):
        self.n_sample, n_features = X.shape
        self.n_class = n_class
        self.theta = np.random.rand(self.n_class, n_features)
        self.bias = np.zeros((1, self.n_class))

        losses=[]

        for i in range(n_iterator):
            scores = self.compute_scores(X)
            hx = self.softmax(scores)
            y_predict = np.argmax(hx, axis=1)[:, np.newaxis]
            y_one_hot = self.one_hot(y_true)

            #loss 是对应元素相乘，所以用*
            loss = -(1 / self.n_sample) * np.sum(y_one_hot * np.log(hx))
            losses.append(loss)

            #dtheta = -1/m * Sigma(xi * (hxi - yi)) = -1/m * (X (Hx - y)) 矩阵乘法
            dTheta = (1 / self.n_sample) * np.dot(X.T, hx - y_one_hot) #注意，只要np.dot, 不用np.sum(np.dot)
            dBias = (1 / self.n_class) * np.sum(hx - y_one_hot, axis=0)

            self.theta = self.theta - learning_rate * dTheta.T
            self.bias = self.bias - learning_rate * dBias

            if i % 100 == 0:
                print(f'Iteration number: {i}, Loss is: {np.round(loss, 4)}')

        return self.theta, self.bias, losses

    def predict(self, X):
        hx = self.softmax(np.dot(X, self.theta.T) + self.bias)
        y_predict = np.argmax(hx, axis=1)[:, np.newaxis]
        return y_predict

    def compute_scores(self, X):
        """
        计算X中样本的类别分数
        
        Args:
            X: numpy array of shape (n_samples, n_features)
        Returns:
            scores: numpy array of shape (n_samples, n_classes)
        """
        return np.dot(X, self.theta.T) + self.bias
