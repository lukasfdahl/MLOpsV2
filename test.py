import torch.optim as optim
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from model import MnistMLP
from device import DEVICE


def evaluate_model(model : MnistMLP, dataloader : DataLoader) -> tuple[float, float]:
    model.eval()
    correct = 0
    total = 0

    loss_func = nn.CrossEntropyLoss()
    running_loss = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE) # transfer data to the correct device
            x = model(images)
            guesses = torch.argmax(x, dim=1) # pick out the most likely option for each tensor in the batch
            total += labels.size(0) # add the number of images in the batch to the total count
            matches : Tensor = (guesses == labels)
            correct += matches.sum().item()
            running_loss += loss_func(x, labels).item() * labels.size(0)

    accuracy = correct / total
    average_loss = running_loss / total

    return accuracy, average_loss
