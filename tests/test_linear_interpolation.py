from ml_theory_tools.linear_interpolation import one_dimensional_linear_interpolation
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

def test_one_dimensional_linear_interpolation_parameters(one_dimensional_linear_interpolation_fixture):
    (
        model,
        criterion,
        grid_size,
        train_loader,
        test_loader,
        params_A_path,
        params_B_path
    ) = one_dimensional_linear_interpolation_fixture
    
    with pytest.raises(ValueError):
        one_dimensional_linear_interpolation(
            params_A_path, params_B_path, criterion, model, None, None, grid_size
        )
    
    with pytest.raises(FileNotFoundError):
        one_dimensional_linear_interpolation(
            "", params_B_path, criterion, model, train_loader, test_loader, grid_size
        )

def test_one_dimensional_linear_interpolation_execution(one_dimensional_linear_interpolation_fixture):
    (
        model,
        criterion,
        grid_size,
        train_loader,
        test_loader,
        params_A_path,
        params_B_path
    ) = one_dimensional_linear_interpolation_fixture
    
    # Test successful execution
    train_losses, test_losses = one_dimensional_linear_interpolation(
        params_A_path,
        params_B_path,
        model,
        criterion,
        train_loader,
        test_loader,
        grid_size
    )
    
    # Check if result contains expected keys
    assert isinstance(train_losses, np.ndarray)
    assert isinstance(test_losses, np.ndarray)
    
    # Check if arrays have correct shapes
    assert len(train_losses) == grid_size
    assert len(test_losses) == grid_size

@pytest.fixture(scope="session")
def create_initial_weights(tmp_path_factory):
    """
    Creates two sets of initial weights and saves them to disk.
    Uses session scope to create files once per test session.
    """
    # Create a temporary directory that persists for the session
    temp_dir = tmp_path_factory.mktemp("model_weights")
    
    # Initialize model
    model = resnet18(num_classes=10)
    
    # Save initial weights as params_A
    params_A_path = os.path.join(temp_dir, 'params_A.pth')
    torch.save(model.state_dict(), params_A_path)
    
    # Create different weights for params_B
    # Train for one epoch on random data to get different weights
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Create some random training data
    random_inputs = torch.randn(100, 3, 224, 224)  # 100 random images
    random_targets = torch.randint(0, 10, (100,))   # 100 random labels
    
    # Train for one batch to get different weights
    model.train()
    optimizer.zero_grad()
    outputs = model(random_inputs)
    loss = criterion(outputs, random_targets)
    loss.backward()
    optimizer.step()
    
    # Save modified weights as params_B
    params_B_path = os.path.join(temp_dir, 'params_B.pth')
    torch.save(model.state_dict(), params_B_path)
    
    return params_A_path, params_B_path

@pytest.fixture
def one_dimensional_linear_interpolation_fixture(create_initial_weights):
    params_A_path, params_B_path = create_initial_weights

    # Initialize model
    model = resnet18(num_classes=10)  # MNIST has 10 classes
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss()

    grid_size = 3
    
    # Initialize MNIST dataset
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet expects 224x224 images
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
    ])
    
    train_dataset = MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    # Limit datasets to only the first 32 samples
    train_subset = Subset(train_dataset, range(32))  # First 32 samples from the training set
    test_subset = Subset(test_dataset, range(32))  # First 32 samples from the test set
    
    # Create data loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=32,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=32,
        shuffle=False
    )
    
    return model, criterion, grid_size, train_loader, test_loader, params_A_path, params_B_path