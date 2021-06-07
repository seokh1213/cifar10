import torchvision
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
from PIL import Image
import random


def get_transform(test=False):
    transforms_list = []
    if not test:
        transforms_list.append(transforms.RandomHorizontalFlip())
    transforms_list.append(transforms.ToTensor())
    transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return transforms.Compose(transforms_list)


class CustomDataset(Dataset):
    def __init__(self, images, targets, valid=False):
        self.images = images
        self.targets = targets
        self.transform = get_transform(valid)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, target = self.images[idx], self.targets[idx]
        image_tensor = self.transform(Image.fromarray(image))
        target_tensor = torch.LongTensor([target])
        return image_tensor, target_tensor


def get_dataloader(batch_size, valid_ratio=0.1, seed=None):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train, x_valid, y_train, y_valid = train_test_split(dataset.data, dataset.targets,
                                                          test_size=valid_ratio,
                                                          random_state=seed,
                                                          stratify=dataset.targets)

    trainset = CustomDataset(x_train, y_train, valid=False)
    validset = CustomDataset(x_valid, y_valid, valid=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                           transform=get_transform(test=True))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, validloader, testloader


def get_rotation_dataloader(batch_size):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    x_data = []
    y_data = []
    for image in dataset.data:
        images = []
        labels = []
        rot = list(range(4))
        random.shuffle(rot)
        for i in rot:
            images.append(np.rot90(image, i))
            labels.append(i)
        x_data.extend(images)
        y_data.extend(labels)
    trainset = CustomDataset(x_data, y_data, valid=True)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    return trainloader
