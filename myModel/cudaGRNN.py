import torch
from torch import nn


class GaussLayer(nn.Module):
    def __init__(self, training_inputs, sigma):
        super(GaussLayer, self).__init__()
        self.training_inputs = training_inputs
        self.sigma = sigma  # smoothing parameter

    def forward(self, x):
        out = x - self.training_inputs
        out = (out ** 2).sum(axis=1)
        out = - out / (2 * self.sigma ** 2)
        out = torch.exp(out)
        return out


class SumAndOutputLayer(nn.Module):
    def __init__(self, training_outputs):
        super(SumAndOutputLayer, self).__init__()
        self.training_outputs = training_outputs

    def forward(self, x):
        trans = self.training_outputs.T
        s0 = x.sum()
        out = (x * trans).sum(axis=1)  # Summation Layer
        out = out / s0  # Output Layer
        return out


class CudaGeneralRegNN:
    def __init__(self):
        self.t = None  # training samples (INPUT)
        self.y = None  # training samples (OUTPUT)
        self.sigma = None  # smoothing parameter
        self.t_mean = None  # mean of each feature (INPUT)
        self.t_std = None  # std of each feature (INPUT)
        self.y_mean = None  # mean of each feature (OUTPUT)
        self.y_std = None  # std of each feature (OUTPUT)
        self.net = None  # General Regression Neural Network
        self.device = None  # cpu/cuda

    def fit(self, t_samples, y_samples, sigma):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

        self.t = self.t.to(device=self.device)
        self.y = self.y.to(device=self.device)
        self.sigma = self.sigma.to(device=self.device)

        self.net = nn.Sequential(
            GaussLayer(self.t, self.sigma),
            SumAndOutputLayer(self.y)
        )
        self.net = self.net.to(device=self.device)

    def predict(self, x):
        x = (x - self.t_mean) / self.t_std
        x = x.to(device=self.device)

        out = self.net(x)

        out = out.to(device=torch.device('cpu'))
        out = out * self.y_std + self.y_mean

        return out

    def batch_predict(self, x):
        pred = []
        for i in range(len(x)):
            y_pred = self.predict(x[i])
            pred.append(y_pred)

        return torch.tensor(pred)
