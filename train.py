import torch.optim as optim
import torch.nn as nn
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from model import MnistMLP
from test import evaluate_model
import os
import copy
from device import DEVICE
from typing import Callable


def train_model(model : MnistMLP, train_dataloader : DataLoader, val_dataloader : DataLoader, epochs : int = 3, initial_learning_rate : float = 0.001, save_path : str = "runs/test", penalty_func : None | Callable[[nn.Module], torch.Tensor] = None) -> MnistMLP:
    os.makedirs(save_path, exist_ok=True)

    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    # to keep track of best model
    best_acc = 0.0
    best_weights = None

    for epoch in range(epochs):
        model.train() # enable training mode
        for images, labels in train_dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE) # transfer data to the correct device
            x = model(images)

            # if we have a penatly function (like ewc) then add it to loss
            if penalty_func is not None:
                loss = loss_func(x, labels) + penalty_func(model)
            else:
                loss = loss_func(x, labels)

            optimizer.zero_grad() # reset gradients
            loss.backward()
            optimizer.step()

        # save most recent model
        torch.save(model.state_dict(), os.path.join(save_path, "recent.pth"))

        # evaluate after each epoch
        accuracy, average_loss = evaluate_model(model, val_dataloader)
        print(f"Epoch {epoch}: acc {accuracy} | loss {average_loss}")

        # check if new model i sbest model
        if accuracy > best_acc:
            best_acc = accuracy
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, os.path.join(save_path, "best.pth"))

    # return the best model
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model


#############
#--- EWC ---#
#############

def ewc_calculate_importance(model: nn.Module, dataloader: DataLoader) -> tuple[dict, dict]:
    model.eval()
    importance = {}
    loss_func = nn.CrossEntropyLoss()

    # Create a dictionary with a value for all the layers and their weights set to 0 (will be populated with importance score later)
    for name, param in model.named_parameters():
        importance[name] = torch.zeros_like(param)

    for images, labels in dataloader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        model.zero_grad()
        x = model(images)
        loss = loss_func(x, labels)
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                # calculates fisher importance
                importance[name] += param.grad ** 2
                pass

    for name in importance:
        # adjusts learning scores based on dataloader batch size,
        importance[name] /= len(dataloader)

    return importance, copy.deepcopy(model.state_dict())

# ewc_lambda is a score controlling how much to value past data
def ewc_create_panalty(anchor_weights: dict, importance : dict, ewc_lamda = 1000.0) -> Callable[[nn.Module], torch.Tensor]:

    def penalty(current_model: nn.Module) -> Tensor:
        total_penalty = torch.tensor(0.0, device=DEVICE)
        for name, current_weight in current_model.named_parameters():
            anchor = anchor_weights[name]
            stiffness = importance[name]
            # formular for penalty
            total_penalty += (((current_weight - anchor) ** 2) * stiffness).sum()

        return total_penalty * ewc_lamda

    return penalty
