import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from dataset import SinusoidalDataset
from models.ridge import RidgeModel
from utils import get_parser


def main():
    parser = get_parser()
    parser.add_argument("--fdim", default=5, type=int)
    parser.add_argument("--num_samples", default=16, type=int)
    parser.add_argument("--num_datasets", default=100, type=int)
    args = parser.parse_args()

    sns.set()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    stats = {"lamb": np.logspace(-4, 0, 50), "bias": [], "variance": [], "sum": []}
    for lamb in stats["lamb"]:
        y_hat = []
        for _ in range(args.num_datasets):
            dataset = SinusoidalDataset(args.num_samples)
            _, sin, y = dataset.get_dataset()
            t = dataset.polynomial_t(fdim=args.fdim)
            ridge = RidgeModel(fdim=args.fdim)
            ridge.fit(t, y, weight_decay=lamb)
            y_hat.append(ridge.predict(t))
        y_hat = np.stack(y_hat)
        bias = np.mean((np.mean(y_hat, axis=0) - sin) ** 2)
        variance = np.mean(np.mean((y_hat - np.mean(y_hat, axis=0)) ** 2, axis=0))
        stats["bias"].append(bias)
        stats["variance"].append(variance)
        stats["sum"].append(bias + variance)

    ax.plot(stats["lamb"], stats["bias"], label="bias^2")
    ax.plot(stats["lamb"], stats["variance"], label="variance")
    ax.plot(stats["lamb"], stats["sum"], label="bias^2 + variance")
    ax.set_xscale("log")
    ax.set_xlabel("lambda")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
