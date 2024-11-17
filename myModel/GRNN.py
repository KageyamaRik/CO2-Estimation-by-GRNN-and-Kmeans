import numpy as np


class GeneralRegNN:
    def __init__(self):
        self.t = None  # training samples (INPUT)
        self.y = None  # training samples (OUTPUT)
        self.sigma = None  # hyper-parameter
        self.t_mean = None  # mean of each feature (INPUT)
        self.t_std = None  # std of each feature (INPUT)
        self.y_mean = None  # mean of each feature (OUTPUT)
        self.y_std = None  # std of each feature (OUTPUT)

    def fit(self, t_samples, y_samples, sigma):
        self.t = t_samples
        self.y = y_samples
        self.sigma = sigma
        self.t_mean = self.t.mean(axis=0)
        self.t_std = self.t.std(axis=0)
        self.y_mean = self.y.mean(axis=0)
        self.y_std = self.y.std(axis=0)

        # Normalization
        self.t = (self.t - self.t_mean) / self.t_std
        self.y = (self.y - self.y_mean) / self.y_std

    def predict(self, x):
        m, n = self.t.shape
        k = self.y.shape[1]
        x = (x - self.t_mean) / self.t_std

        # Pattern Layer
        gauss = np.zeros(m)

        for i in range(m):
            t = self.t[i]
            diff = x - t
            gauss[i] = np.exp(- (diff @ diff) / (2 * self.sigma ** 2))

        # Summation Layer
        s = np.zeros(k)

        for j in range(k):
            s[j] = self.y[:, j] @ gauss

        # Output Layer
        out = s / gauss.sum()
        out = out * self.y_std + self.y_mean

        return out

    def rmse(self, x_test, y_test):
        res = []
        k = y_test.shape[1]
        for i in range(len(x_test)):
            y_pred = self.predict(x_test[i])
            r = (((y_test[i] - y_pred) ** 2).sum() / k) ** 0.5
            res.append(r)

        return res

    def total_rmse(self, x_test, y_test):
        pred = []
        for i in range(len(x_test)):
            y_pred = self.predict(x_test[i])
            pred.append(y_pred)

        pred = np.array(pred).squeeze()
        n = len(pred)

        return (((y_test.squeeze() - pred) ** 2).sum() / n) ** 0.5
