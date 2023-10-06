from ds_project.train_val_utils import validate_model, get_device
from ds_project.model_utils import get_network
from ds_project.data_utils import get_loader


def main():
    device = get_device()
    model = get_network()
    val_loader = get_loader('cifar10', batch_size=128, train=False, shuffle=False)
    criterion = get_criterion()
    validate_model(val_loader, device)

if __name__ == '__main__':
    main()