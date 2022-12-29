import numpy as np
from numpy import linalg

from dataset import WineDataset
from metrics import Metrics
from utils import get_logger, get_parser

logger = get_logger(__file__)


class LogisticRegressionModel:
    def __init__(self, fdim: int, classes: int, eps: float = 1e-8):
        self.classes = classes
        self.w = np.random.normal(0, 0.1, size=(fdim, classes) if classes > 2 else fdim)
        self.eps = eps

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        iters: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
    ):
        if self.classes > 2:
            y = np.identity(self.classes)[y]
        for _ in range(iters):
            if self.classes > 2:
                self._fit_sgd(x, y, lr, weight_decay)
            else:
                self._fit_binary_irls(x, y)

    def _fit_binary_irls(self, x: np.ndarray, y: np.ndarray):
        l = x @ self.w
        p = self.sigmoid(l)
        d = np.diag(p * (1 - p))
        hessian = x.T @ d @ x
        self.w = linalg.pinv(hessian) @ x.T @ d @ (l - linalg.pinv(d) @ (p - y))

    def _fit_sgd(self, x: np.ndarray, y: np.ndarray, lr: float, weight_decay: float):
        l = x @ self.w
        p = self.softmax(l)
        grad = x.T @ (p - y) + 2 * weight_decay * self.w
        self.w -= lr * grad

    def binary_cross_entropy(self, p: np.ndarray, y: np.ndarray):
        return np.mean(-y * np.log(p + self.eps) - (1 - y) * np.log(1 - p + self.eps))

    def predict(self, x: np.ndarray) -> np.ndarray:
        l = x @ self.w
        if self.classes > 2:
            p = self.softmax(l)
            return p.argmax(-1)
        else:
            p = self.sigmoid(l)
            return np.where(p >= 0.5, 1, 0)

    def sigmoid(
        self, logits: np.ndarray, exp_min: float = -256.0, exp_max: float = 256.0
    ) -> np.ndarray:
        p = 1 / (1 + np.exp(-np.clip(logits, a_min=exp_min, a_max=exp_max)))
        p[logits <= exp_min], p[logits >= exp_max] = 0.0, 1.0
        return p

    def softmax(self, logits: np.ndarray) -> np.ndarray:
        return np.exp(logits) / np.sum(np.exp(logits), axis=-1)[:, None]


def main():
    parser = get_parser()
    parser.add_argument("--binary", action="store_true")
    args = parser.parse_args()

    dataset = WineDataset(binary=args.binary)
    (x_train, y_train), _, (x_test, y_test) = dataset.get_dataset()

    logistic = LogisticRegressionModel(
        fdim=x_train.shape[-1], classes=len(set(y_train))
    )
    logistic.fit(x_train, y_train)
    preds = logistic.predict(x_test)
    metrics = Metrics("acc")
    logger.info(f"acc: {metrics(preds, y_test):.3f}")


if __name__ == "__main__":
    main()
