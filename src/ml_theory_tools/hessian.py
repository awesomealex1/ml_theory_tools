from hessian_eigenthings import compute_hessian_eigenthings
import numpy as np
import torch


def eigen(
    n: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss: torch.nn.modules.loss._Loss,
) -> tuple[np.ndarray, np.ndarray]:
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model, dataloader, loss, n, use_gpu=use_gpu
    )
    return eigenvals, eigenvecs
