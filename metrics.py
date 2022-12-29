import numpy as np


class Metrics:
    def __init__(self, metcis: str):
        self.metrics = metcis

    def __call__(self, preds: np.ndarray, targets: np.ndarray):
        if self.metrics == "mse":
            return self.mean_squared_error(preds, targets)
        elif self.metrics == "acc":
            return self.accuracy(preds, targets)

    @staticmethod
    def mean_squared_error(preds: np.ndarray, targets: np.ndarray) -> float:
        return np.mean((targets - preds) ** 2)

    @staticmethod
    def accuracy(preds: np.ndarray, targets: np.ndarray) -> float:
        return (preds == targets).sum() / len(targets)
