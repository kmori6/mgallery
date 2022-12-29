import numpy as np
from numpy import linalg

from dataset import WineDataset
from metrics import Metrics
from utils import get_logger

logger = get_logger(__file__)


class LogisticRegressionModel:
    def __init__(self, fdim: int, eps: float = 1e-8):
        self.w = np.random.uniform(0, 1, fdim)
        self.eps = eps

    def fit_irls(self, x: np.ndarray, y: np.ndarray, iters: int = 10):
        for _ in range(iters):
            l = x @ self.w
            p = np.clip(self.sigmoid(l), a_min=0, a_max=1 - self.eps)
            d = np.diag(p * (1 - p))
            self.w = linalg.pinv(x.T @ d @ x) @ x.T @ d @ (l - linalg.pinv(d) @ (p - y))

    def binary_cross_entropy(self, p: np.ndarray, y: np.ndarray, eps: float = 1e-8):
        return np.mean(-y * np.log(p + eps) - (1 - y) * np.log(1 - p + eps))

    def predict(self, x: np.ndarray) -> np.ndarray:
        l = x @ self.w
        return self.sigmoid(l)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, a_min=-256, a_max=256)
        return 1 / (1 + np.exp(-x))


def main():
    dataset = WineDataset(binary=True)
    (x_train, y_train), _, (x_test, y_test) = dataset.get_dataset()

    logistic = LogisticRegressionModel(fdim=x_train.shape[-1])
    logistic.fit_irls(x_train, y_train)
    preds = np.where(logistic.predict(x_test) >= 0.5, 1, 0)
    metrics = Metrics("acc")
    logger.info(f"acc: {metrics(preds, y_test):.3f}")


if __name__ == "__main__":
    main()
