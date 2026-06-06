from torch.utils.data import DataLoader, Subset, ConcatDataset
from torchvision import datasets, transforms
import torch
from typing import cast
import random

def get_mnist_loader(allowed_digits: list[int], train : bool = True, shuffle : bool = True, batch_size : int = 64) -> DataLoader:
    # transform just specifies some actions to run on the data before it is made avilable in the code, here it is to turn the image files into tensorts and normalize the pixel values to have a center of 0 to make the model train faster
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    # Mask for digits
    mask = torch.isin(train_dataset.targets, torch.tensor(allowed_digits))
    indices = torch.where(mask)[0].tolist()
    subset_train = Subset(train_dataset, indices) # subnet ensures all images are not loaded into ram at once, it just essentially just remebrs where to find them when it needs them

    return DataLoader(dataset=subset_train, batch_size=batch_size, shuffle=shuffle)


def concat_dataloaders(main_dataloader : DataLoader, side_dataloader : DataLoader, side_dataloader_samples = 200, shuffle : bool = True, batch_size : int = 64) -> DataLoader:
    main_dataset : Subset = cast(Subset, main_dataloader.dataset)
    side_dataset : Subset = cast(Subset, side_dataloader.dataset)

    total_side_samples = len(side_dataset)
    indices = random.sample(range(total_side_samples), side_dataloader_samples)
    side_subset = Subset(side_dataset, indices)

    mixed_dataset = ConcatDataset([main_dataset, side_subset])

    return DataLoader(dataset=mixed_dataset, batch_size=batch_size, shuffle=shuffle)
