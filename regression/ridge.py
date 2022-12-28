import numpy as np
from numpy import linalg

from dataset import CaliforniaDataset
from metrics import mean_squared_error
from utils import get_logger, get_parser

logger = get_logger(__name__)


class RidgeModel:
    def __init__(self, fdim: int):
        self.fdim = fdim

    def fit(self, x: np.ndarray, y: np.ndarray, weight_decay: float):
        self.w = linalg.inv(weight_decay * np.identity(self.fdim) + x.T @ x) @ x.T @ y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.sum(self.w * x, axis=-1)


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset = CaliforniaDataset()
    (x_train, y_train), _, (x_test, y_test) = dataset.get_dataset()

    ridge = RidgeModel(x_train.shape[-1])
    ridge.fit(x_train, y_train, weight_decay=args.weight_decay)
    y_hat = ridge.predict(x_test)
    mse = mean_squared_error(y_hat, y_test)
    logger.info(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
