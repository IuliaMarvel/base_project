import torch.nn as nn
from torch import optim


def get_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion


def get_SGD_optimizer(model, lr=0.06, momentum=0.9):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return optimizer
