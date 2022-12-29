from dataclasses import dataclass

import numpy as np
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split


def split_dataset(
    x: np.ndarray,
    y: np.ndarray,
    dev_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 0,
):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=dev_size + test_size, random_state=random_state
    )
    x_dev, x_test, y_dev, y_test = train_test_split(
        x_test,
        y_test,
        test_size=test_size / (dev_size + test_size),
        random_state=random_state,
    )
    return (x_train, y_train), (x_dev, y_dev), (x_test, y_test)


class CaliforniaDataset:
    def __init__(self, dev_size: float = 0.1, test_size: float = 0.1):
        x, y = fetch_california_housing(return_X_y=True)
        self.train_dataset, self.dev_dataset, self.test_dataset = split_dataset(
            x, y, dev_size, test_size
        )

    def get_dataset(self):
        return self.train_dataset, self.dev_dataset, self.test_dataset


class WineDataset:
    def __init__(
        self,
        dev_size: float = 0.1,
        test_size: float = 0.1,
        binary: bool = False,
        standardize: bool = True,
    ):
        x, y = load_wine(return_X_y=True)
        if binary:
            x, y = x[y <= 1], y[y <= 1]
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=dev_size + test_size, random_state=0, stratify=y
        )
        x_dev, x_test, y_dev, y_test = train_test_split(
            x_test,
            y_test,
            test_size=test_size / (dev_size + test_size),
            random_state=0,
            stratify=y_test,
        )

        if standardize:
            self.mean = np.mean(x_train, axis=0)
            self.std = np.std(x_train, axis=0)
            x_train = self.standardize(x_train)
            x_dev = self.standardize(x_dev)
            x_test = self.standardize(x_test)

        self.train_dataset = (x_train, y_train)
        self.dev_dataset = (x_dev, y_dev)
        self.test_dataset = (x_test, y_test)

    def get_dataset(self):
        return self.train_dataset, self.dev_dataset, self.test_dataset

    def standardize(self, x: np.ndarray):
        return (x - self.mean) / self.std


class NormalDataset:
    def __init__(
        self,
        classes: int,
        fdim: int,
        samples_per_class: int,
        max_mean: float = 5.0,
        min_mean: float = -5.0,
        std: float = 0.5,
    ):
        x, y = [], []
        for c in range(classes):
            feats = []
            for f in range(fdim):
                mean = np.random.uniform(min_mean, max_mean)
                feats.append(np.random.normal(mean, std, samples_per_class))
            feats = np.stack(feats, axis=-1)  # (N, F)
            x.append(feats)
            y.append(np.full(samples_per_class, c))
        self.x = np.concatenate(x, axis=0)
        self.y = np.concatenate(y, axis=0)


class SinusoidalDataset:
    def __init__(
        self,
        samples: int,
        mean: float = 0.0,
        std: float = 0.1,
        tmin: float = 0.0,
        tmax: float = 1.0,
    ):
        self.t = np.sort(np.random.uniform(tmin, tmax, samples))
        self.sin = np.sin(2 * np.pi * self.t)
        self.y = self.sin + np.random.normal(mean, std, samples)

    def get_dataset(self):
        return self.t, self.sin, self.y

    def polynomial_t(self, fdim: int):
        return (
            np.concatenate([self.t**i for i in range(1, fdim + 1)])
            .reshape(fdim, -1)
            .T
        )
