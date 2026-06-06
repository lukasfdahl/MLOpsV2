from model import MnistMLP
from dataloader import get_mnist_loader, get_confused_dataloader, concat_dataloaders
from train import train_model, ewc_calculate_importance, ewc_create_panalty
from test import evaluate_model
from device import DEVICE
from torch.utils.data import DataLoader

model = MnistMLP()
model = model.to(DEVICE)
train_loader = get_mnist_loader([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
train_loader_7 = get_mnist_loader([7])
val_loader_all = get_mnist_loader([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], train=False)

val_loaders : list[DataLoader] = []
for i in range(0, 10):
    val_loaders.append(get_mnist_loader([i], train=False))

model = train_model(model, train_loader, val_loader_all, epochs=5, save_path="runs/task2-2&3")

# unlearn 7
ewc_importance, anchor_weights = ewc_calculate_importance(model, val_loader_all)
penatly_func = ewc_create_panalty(anchor_weights, ewc_importance, ewc_lamda=1000)

train_loader_7 = get_confused_dataloader(train_loader_7, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [7])
unlearn_7_train_loader = concat_dataloaders(train_loader_7, train_loader, 5000)

model = train_model(model, train_loader_7, val_loader_all, epochs=1, penalty_func=penatly_func, save_path="runs/task2-2&3")

accuracy, average_loss = evaluate_model(model, val_loader_all)
print("===================================")
print(f"Final Accuracy: acc {accuracy} | loss {average_loss}")

for i in range(0, 10):
    accuracy, average_loss = evaluate_model(model, val_loaders[i])
    print(f"{i} acc: {accuracy} | loss : {average_loss}")

print("===================================")
