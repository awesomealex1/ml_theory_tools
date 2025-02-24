from ml_theory_tools.hessian import eigen
import torch
import pytest
from torchvision.models import resnet18
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import os
import tempfile
import numpy as np
from torchvision import transforms
import torch.nn.functional as F


# Define the fully connected neural network
class SimpleFCNet(nn.Module):
    def __init__(self):
        super(SimpleFCNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 10)  # 28*28 is the input image size
        self.fc2 = nn.Linear(10, 10)  # 10 hidden units
        self.fc3 = nn.Linear(10, 10)  # Output layer for 10 classes (digits 0-9)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = F.relu(self.fc1(x))  # Apply ReLU activation after the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU activation after the second layer
        x = self.fc3(x)  # Output layer
        return x


def test_eigen(eigen_fixture):
    model, criterion, train_loader, n = eigen_fixture
    eigenvals, eigenvecs = eigen(n, model, train_loader, criterion)

    assert len(eigenvals) == n
    assert len(eigenvecs) == n


@pytest.fixture
def eigen_fixture():
    # Initialize model
    model = SimpleFCNet()  # MNIST has 10 classes

    # Initialize criterion
    criterion = nn.CrossEntropyLoss()

    # Initialize MNIST dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),  # MNIST normalization
        ]
    )

    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)

    # Limit datasets to only the first 32 samples
    train_subset = Subset(
        train_dataset, range(32)
    )  # First 32 samples from the training set

    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

    n = 2

    return model, criterion, train_loader, n
