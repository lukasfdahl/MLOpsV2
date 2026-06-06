from model import MnistMLP
from dataloader import get_mnist_loader
from train import train_model
from test import evaluate_model



model = MnistMLP()
train_loader = get_mnist_loader([0, 1, 2, 3, 4])
val_loader = get_mnist_loader([0, 1, 2, 3, 4], train=False)

model = train_model(model, train_loader, val_loader, epochs=5)

accuracy, average_loss = evaluate_model(model, val_loader)
print("===================================")
print(f"Final Accuracy: acc {accuracy} | loss {average_loss}")
print("===================================")
