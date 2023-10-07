from ds_project.data_utils import get_loader
from ds_project.train_val_utils import get_device, validate_model


def main():
    device = get_device()
    val_loader = get_loader("cifar10", batch_size=128, train=False, shuffle=False)
    validate_model(val_loader, device)


if __name__ == "__main__":
    main()
