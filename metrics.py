import numpy as np


def mean_squared_error(preds: np.ndarray, targets: np.ndarray) -> float:
    return np.mean((targets - preds) ** 2)
