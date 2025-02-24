from hessian_eigenthings import compute_hessian_eigenthings
import numpy as np
import torch
from typing import Optional


def eigen(
    n: int,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss: torch.nn.modules.loss._Loss,
    params_path: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False
    if params_path:
        model.load_state_dict(torch.load(params_path, weights_only=True))
    eigenvals, eigenvecs = compute_hessian_eigenthings(
        model, dataloader, loss, n, use_gpu=use_gpu
    )
    return eigenvals, eigenvecs
