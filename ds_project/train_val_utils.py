import torch
from tqdm import tqdm
from model_utils import get_network


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def train_model(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    n_epochs,
    path="./cifar_model.pth",
):
    print("Training started.")
    model = model.to(device)
    model.train()
    for i in tqdm(range(n_epochs)):
        for data in train_loader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print(f"Finished Training. Model saved to {path} .")
    torch.save(net.state_dict(), path)
    return model


def validate_model(val_loader, device, model_path="./cifar_model.pth"):
    print("Validation started")
    model = get_network()
    model = model.load_state_dict(torch.load(model_path)).to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(val_tloader):
            images, labels = data[0].to(device), data[1].to(device)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on test images: {100 * correct // total} %")
