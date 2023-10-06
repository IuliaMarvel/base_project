import torch
import torchvision
from torchvision import transforms
from torchvision import datasets

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)


def get_loader(data_dir, batch_size, train=True, shuffle=True):
    dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=train, download=True, transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2
    )
    return dataloader
