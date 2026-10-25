"""Logistic regression written with NumPy only. This is the "learned" expert.

I wrote it by hand so each step is visible: standardise the features, then minimise
cross-entropy with L2 regularisation using full-batch gradient descent. The fitted state is
plain numbers that convert to JSON, so there is no pickle and nothing executable to load.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Logistic function. The clip avoids overflow in exp() for extreme inputs."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticModel:
    def __init__(self, l2: float = 1.0, lr: float = 0.1, epochs: int = 300) -> None:
        self.l2, self.lr, self.epochs = l2, lr, epochs
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.bias_: float = 0.0

    @property
    def fitted(self) -> bool:
        return self.coef_ is not None

    def fit(self, x: np.ndarray, y: np.ndarray) -> LogisticModel:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if x.ndim != 2 or len(x) != len(y) or len(x) < 10:
            raise ValueError("Need a 2-D feature matrix and at least 10 matching labels.")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("Training data contains NaN or infinite values.")
        # Standardise so one large-scale feature (e.g. volume ratio) cannot dominate the
        # gradient. A constant column has zero spread; use 1.0 to avoid dividing by zero.
        self.mean_ = x.mean(axis=0)
        std = x.std(axis=0)
        self.std_ = np.where(std < 1e-12, 1.0, std)
        z = (x - self.mean_) / self.std_
        n, d = z.shape
        w = np.zeros(d)
        # Initialise the bias at log-odds of the base rate so the untrained model already
        # predicts "the usual outcome" instead of starting from 50/50.
        b = float(np.log((y.mean() + 1e-6) / (1 - y.mean() + 1e-6)))
        for _ in range(self.epochs):
            p = sigmoid(z @ w + b)
            err = p - y  # gradient of cross-entropy w.r.t. the logit
            w -= self.lr * (z.T @ err / n + self.l2 * w / n)  # L2 penalty on weights, not bias
            b -= self.lr * float(err.mean())
        self.coef_, self.bias_ = w, b
        logger.debug("LogisticModel fitted: n=%d features=%d epochs=%d", n, d, self.epochs)
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Model is not fitted.")
        z = (np.asarray(x, dtype=float) - self.mean_) / self.std_
        return sigmoid(z @ self.coef_ + self.bias_)

    def to_dict(self) -> dict[str, Any]:
        if not self.fitted:
            raise RuntimeError("Model is not fitted.")
        return {
            "mean": self.mean_.tolist(),
            "std": self.std_.tolist(),
            "coef": self.coef_.tolist(),
            "bias": self.bias_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> LogisticModel:
        model = cls()
        model.mean_ = np.asarray(data["mean"], dtype=float)
        model.std_ = np.asarray(data["std"], dtype=float)
        model.coef_ = np.asarray(data["coef"], dtype=float)
        model.bias_ = float(data["bias"])
        return model
