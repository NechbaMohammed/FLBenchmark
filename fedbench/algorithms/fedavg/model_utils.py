import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def train(net, trainloader, epochs: int,device: torch.device, learning_rate: float,
    momentum: float,
    weight_decay: float,  verbose=False):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay
    )
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = net(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += target.size(0)
            correct += (torch.max(outputs.data, 1)[1] == target).sum().item()
        epoch_loss /= len(trainloader.dataset)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")

def compute_accuracy(model, dataloader, device):
    """Compute accuracy."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    loss = loss / total
    acc = correct / total

    return  loss, acc

def test(net, test_dataloader, device):
    """Test function."""
    net.to(device)
    loss, test_acc = compute_accuracy(net, test_dataloader, device=device)
    #print(">> Test accuracy: %f" % test_acc)
    net.to("cpu")
    return loss, test_acc