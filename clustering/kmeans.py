import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import linalg

from dataset import NormalDataset


class Kmeans:
    def __init__(self, clusters: int, fdim: int):
        self.clusters = clusters
        self.fdim = fdim
        self.vec = np.random.normal(0, 0.1, (self.clusters, fdim))

    def fit(self, x: np.ndarray, iters: int = 100):
        for i in range(iters):
            clusters = self.estep(x)
            self.mstep(x, clusters)

    def estep(self, x: np.ndarray) -> np.ndarray:
        return self.clustering(x)

    def mstep(self, x: np.ndarray, clusters: np.ndarray):
        for c in range(self.clusters):
            if c in clusters:
                self.vec[c, :] = sum(x[clusters == c]) / sum(clusters == c)

    def clustering(self, x: np.ndarray) -> np.ndarray:
        l2_norms = linalg.norm((x[:, None, :] - self.vec), ord=2, axis=-1)  # (N, C)
        clusters = l2_norms.argmin(-1)
        return clusters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clusters", default=3, type=int)
    parser.add_argument("--fdim", default=2, type=int)
    parser.add_argument("--samples_per_class", default=100, type=int)
    args = parser.parse_args()

    dataset = NormalDataset(args.clusters, args.fdim, args.samples_per_class)
    x = dataset.x

    kmeans = Kmeans(args.clusters, args.fdim)
    kmeans.fit(x)
    clusters = kmeans.clustering(x)

    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for c in range(args.clusters):
        ax.scatter(x[clusters == c][:, 0], x[clusters == c][:, 1], alpha=0.5)
        ax.scatter(kmeans.vec[c, 0], kmeans.vec[c, 1], marker="x")
    plt.show()


if __name__ == "__main__":
    main()
