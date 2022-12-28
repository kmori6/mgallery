from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import linalg

from dataset import SinusoidalDataset
from utils import get_parser


class BayesModel:
    def __init__(self, fdim: int, alpha: float = None, beta: float = None):
        self.alpha = alpha
        self.beta = beta
        self.fdim = fdim

    def fit(self, x: np.ndarray, y: np.ndarray, iters: int = 100):
        x = self.basis_function(x)
        if self.alpha and self.beta:
            self.sigma = linalg.inv(
                self.alpha * np.identity(self.fdim) + self.beta * x.T @ x
            )
            self.mu = self.beta * self.sigma @ x.T @ y
        else:
            self.alpha = np.random.uniform()
            self.beta = np.random.uniform()
            w = self.beta * linalg.eig(x.T @ x)[0]
            for i in range(iters):
                self.sigma = linalg.inv(
                    self.alpha * np.identity(self.fdim) + self.beta * x.T @ x
                )
                self.mu = self.beta * self.sigma @ x.T @ y
                gamma = np.sum(w / (self.alpha + w))
                alpha = gamma / (self.mu.T @ self.mu)
                beta = (x.shape[0] - gamma) / np.sum((y - x @ self.mu) ** 2)
                if np.allclose([self.alpha, self.beta], [alpha, beta]):
                    break
                else:
                    self.alpha = alpha
                    self.beta = beta

    def basis_function(self, x: np.ndarray) -> np.ndarray:
        return x

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray]:
        x = self.basis_function(x)
        mean = x @ self.mu
        var = 1 / self.beta + (x[:, None, :] @ self.sigma @ x[:, :, None]).squeeze()
        return mean, var


def main():
    parser = get_parser()
    parser.add_argument("--num_samples", default=32, type=int)
    parser.add_argument("--polynomial_dim", default=3, type=int)
    args = parser.parse_args()

    dataset = SinusoidalDataset(args.num_samples)
    t, _, y = dataset.get_dataset()
    x = dataset.polynomial_t(fdim=args.polynomial_dim)

    bayes = BayesModel(fdim=args.polynomial_dim)
    bayes.fit(x, y)
    mean, var = bayes.predict(x)
    std = np.sqrt(var)

    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(t, y)
    ax.plot(t, mean)
    upper = mean + std
    lower = mean - std
    ax.fill_between(t, upper, lower, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
