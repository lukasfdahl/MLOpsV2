import torch.optim as optim
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from model import MnistMLP


def evaluate_model(model : MnistMLP, dataloader : DataLoader) -> MnistMLP:
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)
