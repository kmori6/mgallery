from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import linalg

from dataset import SinusoidalDataset
from utils import get_parser, get_logger, polynomial_basis_fn

logger = get_logger(__file__)


class BayesModel:
    def __init__(self, fdim: int, alpha: float = None, beta: float = None):
        self.alpha = alpha
        self.beta = beta
        self.fdim = fdim

    def fit(self, x: np.ndarray, y: np.ndarray, iters: int = 100):
        x = polynomial_basis_fn(x, degree=self.fdim)
        if self.alpha and self.beta:
            self.sigma = linalg.inv(
                self.alpha * np.identity(self.fdim) + self.beta * x.T @ x
            )
            self.mu = self.beta * self.sigma @ x.T @ y
        else:
            self.alpha = np.random.normal(0, 0.1)
            self.beta = np.random.normal(0, 0.1)
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

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray]:
        x = polynomial_basis_fn(x, degree=self.fdim)
        mean = x @ self.mu
        var = 1 / self.beta + (x[:, None, :] @ self.sigma @ x[:, :, None]).squeeze()
        return mean, var

    def model_evidence(self, x: np.ndarray, y: np.ndarray):
        x = polynomial_basis_fn(x, degree=self.fdim)
        first_term = self.fdim / 2 * np.log(self.alpha)
        second_term = x.shape[0] / 2 * np.log(self.beta)
        integral_term = (
            self.beta / 2 * linalg.norm(y - x @ self.mu, ord=2)
            + self.alpha / 2 * self.mu.T @ self.mu
            + 1 / 2 * np.log(linalg.det(self.sigma))
            + x.shape[0] / 2 * np.log(2 * np.pi)
        )
        return first_term + second_term - integral_term


def main():
    parser = get_parser()
    parser.add_argument("--num_samples", default=32, type=int)
    parser.add_argument("--polynomial_degree", default=3, type=int)
    args = parser.parse_args()

    dataset = SinusoidalDataset(args.num_samples)
    x, _, y = dataset.get_dataset()

    bayes = BayesModel(fdim=args.polynomial_degree)
    bayes.fit(x, y)
    mean, var = bayes.predict(x)
    std = np.sqrt(var)
    logger.info(f"model evidence: {bayes.model_evidence(x, y):.3f}")

    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(x, y)
    ax.plot(x, mean)
    upper = mean + std
    lower = mean - std
    ax.fill_between(x, upper, lower, alpha=0.5)
    plt.show()


if __name__ == "__main__":
    main()
