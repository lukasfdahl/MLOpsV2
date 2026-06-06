import torch.nn as nn
from torch import Tensor

# Just a basic Linear model to use
class MnistMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(784, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, 128)
        self.l5 = nn.Linear(128, 128)
        self.l6 = nn.Linear(128, 10)

        self.relu = nn.ReLU()

    def forward(self, x : Tensor):
        # This flattens the image from [batch, 1, 28, 28] to (batch, 784)
        x = x.view(x.size(0), -1)
        x = self.relu(self.l1(x))
        x = self.relu(self.l2(x))
        x = self.relu(self.l3(x))
        x = self.relu(self.l4(x))
        x = self.relu(self.l5(x))
        x = self.l6(x)

        return x
