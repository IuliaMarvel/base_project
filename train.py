from ds_project.train_val_utils import train_model, get_device
from ds_project.model_utils import get_network
from ds_project.optim_utils import get_criterion, get_SGD_optimizer
from ds_project.data_utils import get_loader


def main():
    device = get_device()
    model = get_network()
    train_loader = get_loader('cifar10', batch_size=128, train=True, shuffle=True)
    optimizer = get_SGD_optimizer(model)
    criterion = get_criterion()
    train_model(model, train_loader, optimizer, criterion, device, n_epochs=30)


if __name__ == '__main__':
    main()