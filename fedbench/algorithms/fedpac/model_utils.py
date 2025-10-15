import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from collections import defaultdict


def train_fedpac(
    model: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    lr_f: float,
    lr_g: float,
    lambda_reg: float,
    global_centroids: dict = None,
    current_round: int = 0,
    start_reg_round: int = 5,
):
    """
    FedPAC alternating optimization training function.
    Phase 1: Update classifier with frozen feature extractor.
    Phase 2: Update feature extractor with frozen classifier and optional feature alignment.
    """
    model.to(device)
    model.train()
    feature_extractor = model.feature_extractor
    classifier = model.classifier

    criterion = nn.CrossEntropyLoss()
    optimizer_g = torch.optim.SGD(classifier.parameters(), lr=lr_g)
    optimizer_f = torch.optim.SGD(feature_extractor.parameters(), lr=lr_f)

    # Phase 1: Fix feature extractor, train classifier
    feature_extractor.eval()
    classifier.train()
    for _ in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            with torch.no_grad():
                features = feature_extractor(inputs)
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer_g.zero_grad()
            loss.backward()
            optimizer_g.step()

    # Phase 2: Fix classifier, train feature extractor
    feature_extractor.train()
    classifier.eval()
    for _ in range(epochs):
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = feature_extractor(inputs)
            outputs = classifier(features)
            loss_cls = criterion(outputs, labels)

            loss_reg = 0.0
            if global_centroids is not None and current_round >= start_reg_round:
                for i in range(len(labels)):
                    cls = labels[i].item()
                    if cls in global_centroids:
                        centroid = global_centroids[cls].to(device)
                        loss_reg += torch.norm(features[i] - centroid) ** 2
                loss_reg /= len(labels)
                loss_cls += lambda_reg * loss_reg

            optimizer_f.zero_grad()
            loss_cls.backward()
            optimizer_f.step()

    model.to("cpu")
    centroids, sample_counts = compute_local_centroids(model, trainloader, device)
    return model.state_dict(), {
        "centroids": centroids,
        "sample_counts": sample_counts
    }


def evaluate_fedpac(model, dataloader: DataLoader, device: torch.device):
    """Evaluate model on given data using proper architecture."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    model.to(device)
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            features = model.feature_extractor(data)
            output = model.classifier(features)
            loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    model.to("cpu")
    return loss / total, correct / total


def compute_local_centroids(model, dataloader: DataLoader, device: torch.device):
    """
    Compute local feature centroids per class.
    
    Returns:
        centroids (dict): {class_label: centroid_tensor}
        sample_counts (dict): {class_label: count}
    """
    model.eval()
    model.to(device)
    feature_extractor = model.feature_extractor

    class_sums = defaultdict(lambda: 0)
    class_counts = defaultdict(lambda: 0)

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            features = feature_extractor(inputs)
            for i in range(len(labels)):
                label = labels[i].item()
                class_sums[label] += features[i]
                class_counts[label] += 1

    centroids = {cls: class_sums[cls] / class_counts[cls] for cls in class_sums}
    model.to("cpu")
    return centroids, dict(class_counts)
