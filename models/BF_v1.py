from torch.nn import functional as F
import torch.nn as nn

class Encode(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        mu, logvar = self.fc21(h1), self.fc22(h1)
        return mu, logvar

class Decode(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(20, 400)
        self.fc2 = nn.Linear(400, 784)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)#0-1随机数
        return mu + eps*std
    def forward(self, mu, logvar):
        z = self.reparameterize(mu, logvar)
        h3 = F.relu(self.fc1(z))
        x_f = torch.sigmoid(self.fc2(h3))
        return x_f