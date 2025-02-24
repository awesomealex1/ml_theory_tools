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


def test_eigen_with_state_dict(eigen_fixture, create_initial_weights):
    model, criterion, train_loader, n = eigen_fixture
    params_path = create_initial_weights

    eigenvals, eigenvecs = eigen(n, model, train_loader, criterion, params_path)

    assert len(eigenvals) == n
    assert len(eigenvecs) == n


@pytest.fixture(scope="session")
def create_initial_weights(tmp_path_factory):
    """
    Creates two sets of initial weights and saves them to disk.
    Uses session scope to create files once per test session.
    """
    # Create a temporary directory that persists for the session
    temp_dir = tmp_path_factory.mktemp("model_weights")

    # Initialize model
    model = SimpleFCNet()

    # Save initial weights as params_A
    params_path = os.path.join(temp_dir, "params.pth")
    torch.save(model.state_dict(), params_path)

    return params_path

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
