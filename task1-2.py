from model import MnistMLP
from dataloader import get_mnist_loader
from train import train_model
from test import evaluate_model
from device import DEVICE



model = MnistMLP()
model = model.to(DEVICE)
train_loader_04 = get_mnist_loader([0, 1, 2, 3, 4])
val_loader_04 = get_mnist_loader([0, 1, 2, 3, 4], train=False)
train_loader_59 = get_mnist_loader([5, 6, 7, 8, 9])
val_loader_59 = get_mnist_loader([5, 6, 7, 8, 9], train=False)

model = train_model(model, train_loader_04, val_loader_04, epochs=5, save_path="runs/task1-2/04")

accuracy, average_loss = evaluate_model(model, val_loader_04)
print("===================================")
print(f"0-4 Accuracy: acc {accuracy} | loss {average_loss}")
print("===================================")

model = train_model(model, train_loader_59, val_loader_59, epochs=5, save_path="runs/task1-2/59")

accuracy, average_loss = evaluate_model(model, val_loader_04)
print("===================================")
print(f"Final 0-4 Accuracy: acc {accuracy} | loss {average_loss}")

accuracy, average_loss = evaluate_model(model, val_loader_59)
print(f"Final 5-9 Accuracy: acc {accuracy} | loss {average_loss}")
print("===================================")
