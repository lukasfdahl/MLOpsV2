from model import MnistMLP
from dataloader import get_mnist_loader
from train import train_model
from test import evaluate_model
from device import DEVICE



model = MnistMLP()
model = model.to(DEVICE)
train_loader = get_mnist_loader([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
val_loader_all = get_mnist_loader([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], train=False)

val_loaders : list[]

model = train_model(model, train_loader, val_loader_all, epochs=5, save_path="runs/task2-1")

accuracy, average_loss = evaluate_model(model, val_loader_all)
print("===================================")
print(f"Final Accuracy: acc {accuracy} | loss {average_loss}")
print("===================================")
