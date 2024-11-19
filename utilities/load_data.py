import torch
from torchvision import datasets, transforms


class Configuration:
    def __init__(self):
        self.shuffle = False
        self.batch_size = 1
        return


class ConfigDataLoad:
    def __init__(self, type):
        self.type = type
        self.train = Configuration()
        self.test = Configuration()


def load_cifar10_dataset(cfg):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=cfg.train.batch_size, shuffle=cfg.train.shuffle)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.test.batch_size, shuffle=cfg.test.shuffle)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    label_map = []
    return train_loader, test_loader, classes, label_map


def load_data(cfg):
    if cfg.type == 'cifar10':
        train_loader, test_loader, classes, label_map = load_cifar10_dataset(cfg)
    else:
        print("Invalid dataset type")
    return train_loader, test_loader, classes, label_map
