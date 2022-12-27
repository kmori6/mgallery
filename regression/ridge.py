import argparse

import numpy as np
from numpy import linalg
from sklearn.datasets import fetch_california_housing

from utils import split_dataset, get_logger
from metrics import mean_squared_error

logger = get_logger(__name__)


class RidgeModel:
    def __init__(self, fdim: int):
        self.fdim = fdim

    def fit(self, x: np.ndarray, y: np.ndarray, lamb: float):
        self.w = linalg.inv(lamb * np.identity(self.fdim) + x.T @ x) @ x.T @ y

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.sum(self.w * x, axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lamb", default=1e-3, type=float)
    args = parser.parse_args()

    x, y = fetch_california_housing(return_X_y=True)
    (x_train, y_train), _, (x_test, y_test) = split_dataset(x, y)

    ridge = RidgeModel(x.shape[1])
    ridge.fit(x_train, y_train, lamb=args.lamb)
    y_hat = ridge.predict(x_test)
    mse = mean_squared_error(y_hat, y_test)
    logger.info(f"MSE: {mse:.4f}")


if __name__ == "__main__":
    main()
