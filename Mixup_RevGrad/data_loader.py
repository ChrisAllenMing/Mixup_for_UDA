from torchvision import datasets, transforms
import torch
import numpy as np
import os

mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.5, 0.5, 0.5])

def load_training(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([28, 28]),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir, 'trainset'), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs)

    return train_loader

def load_testing(root_path, dir, batch_size, kwargs):
    transform = transforms.Compose(
        [transforms.Resize([28, 28]),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    data = datasets.ImageFolder(root=os.path.join(root_path, dir, 'testset'), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=False, **kwargs)

    return test_loader