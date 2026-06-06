from model import MnistMLP
from dataloader import get_mnist_loader, concat_dataloaders
from train import train_model, ewc_calculate_importance, ewc_create_panalty
from test import evaluate_model
from device import DEVICE



model = MnistMLP()
model = model.to(DEVICE)
train_loader_04 = get_mnist_loader([0, 1, 2, 3, 4])
val_loader_04 = get_mnist_loader([0, 1, 2, 3, 4], train=False)
train_loader_59 = concat_dataloaders(get_mnist_loader([5, 6, 7, 8, 9]), train_loader_04, 200)
val_loader_59 = concat_dataloaders(get_mnist_loader([5, 6, 7, 8, 9], train=False), val_loader_04, 200)

model = train_model(model, train_loader_04, val_loader_04, epochs=5, save_path="runs/task1-3/04")

accuracy, average_loss = evaluate_model(model, val_loader_04)
print("===================================")
print(f"0-4 Accuracy: acc {accuracy} | loss {average_loss}")
print("===================================")

ewc_importance, anchor_weights = ewc_calculate_importance(model, val_loader_04)
penatly_func = ewc_create_panalty(anchor_weights, ewc_importance, ewc_lamda=1000)
model = train_model(model, train_loader_59, val_loader_59, epochs=5, save_path="runs/task1-3/59", penalty_func=penatly_func)

accuracy, average_loss = evaluate_model(model, val_loader_04)
print("===================================")
print(f"Final 0-4 Accuracy: acc {accuracy} | loss {average_loss}")

accuracy, average_loss = evaluate_model(model, val_loader_59)
print(f"Final 5-9 Accuracy: acc {accuracy} | loss {average_loss}")
print("===================================")
