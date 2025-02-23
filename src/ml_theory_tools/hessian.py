from hessian_eigenthings import compute_hessian_eigenthings
import numpy as np


def eigen(n: int, model, dataloader, loss) -> tuple[np.ndarray, np.ndarray]:
    eigenvals, eigenvecs = compute_hessian_eigenthings(model, dataloader, loss, n)
    return eigenvals, eigenvecs
