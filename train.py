import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import MnistMLP


def train_model(model : MnistMLP, dataloader : DataLoader, epochs : int = 3, initial_learning_rate : float = 0.001) -> MnistMLP:
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    for epoch in range(epochs):
        model.train() # enable training mode
        for images, labels in dataloader:
            x = model(images)
            loss = loss_func(x, labels)
            optimizer.zero_grad() # reset gradients
            loss.backward()
            optimizer.step()

    return model
