from ml_theory_tools.linear_interpolation import one_dimensional_linear_interpolation
import torch
import pytest


def test_one_dimensional_linear_interpolation_parameters():
    params_A_path = None
    params_B_path = None
    criterion = None
    model = None
    train_loader = None
    test_loader = None
    with pytest.raises(ValueError):
        one_dimensional_linear_interpolation(
            params_A_path, params_B_path, criterion, model, train_loader, test_loader
        )

    train_loader = 1
    test_loader = 1
    with pytest.raises(AttributeError):
        one_dimensional_linear_interpolation(
            params_A_path, params_B_path, criterion, model, train_loader, test_loader
        )
