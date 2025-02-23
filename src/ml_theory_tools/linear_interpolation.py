import numpy as np
import torch.nn.functional as F
from torch import nn
import torch
from pickle import UnpicklingError


def one_dimensional_linear_interpolation(
    params_A_path: str,
    params_B_path: str,
    model: torch.nn.Module,
    loss: torch.nn.modules.loss._Loss,
    train_loader: torch.utils.data.DataLoader = None,
    test_loader: torch.utils.data.DataLoader = None,
    grid_size: int = 100,
    save_file_prefix: str = None,
) -> tuple[np.ndarray, np.ndarray]:
    def theta(alpha, param1, param2):
        return (1 - alpha) * param1 + alpha * param2

    if not (train_loader or test_loader):
        raise ValueError("Need to supply at least one dataloader. None were supplied.")

    try:
        params_A = torch.load(params_A_path, weights_only=True)
        params_B = torch.load(params_B_path, weights_only=True)
    except UnpicklingError:
        params_A = torch.load(
            params_A_path, weights_only=True, map_location=torch.device("cpu")
        )
        params_B = torch.load(
            params_B_path, weights_only=True, map_location=torch.device("cpu")
        )

    alpha_range = np.linspace(-1, 2, grid_size)
    train_losses, test_losses = np.zeros(grid_size), np.zeros(grid_size)

    for i, alpha in enumerate(alpha_range):

        params_interpolated = {}
        for key, value in params_A.items():
            params_interpolated[key] = value * alpha + (1 - alpha) * params_B[key]
        model.load_state_dict(params_interpolated)

        if train_loader:
            loss_sum = 0
            for X, Y in iter(train_loader):
                predictions = model(X)
                loss_sum += loss(X, Y).item()
            train_losses.append(loss_sum)

        if test_loader:
            loss_sum = 0
            for X, Y in iter(test_loader):
                predictions = model(X)
                loss_sum += loss(X, Y).item()
            test_losses.append(loss_sum)

    if save_file_name:
        np.save(f"save_file_prefix_{train_losses}", train_losses)
        np.save(f"save_file_prefix_{test_losses}", test_losses)

    return train_losses, test_losses
