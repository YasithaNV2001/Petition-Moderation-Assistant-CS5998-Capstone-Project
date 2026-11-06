"""Calibration, similarity features and the fusion model.

Used by notebooks/08_fusion.ipynb, 09 and later src/pipeline.py, so validation, test and new petitions
are all treated in exactly the same way.
"""
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SIMILARITY_COLUMNS = ["sim_top1", "sim_top5_mean", "sim_gap", "pool_size"]


def fit_temperature(logits, y):
    """The temperature T with the lowest log-loss on these logits (T > 1 makes the model less confident)."""
    def log_loss(t):
        p = softmax(logits / t, axis=1)
        return -np.log(p[np.arange(len(y)), y]).mean()
    return float(minimize_scalar(log_loss, bounds=(0.05, 20), method="bounded").x)


def apply_temperature(logits, t):
    """Probabilities from logits at temperature t. The class order and the top class do not change."""
    return softmax(logits / t, axis=1)


class PlattScaler:
    """Platt scaling from raw SVC scores: one sigmoid per class, then each row is scaled to add up to 1."""

    def fit(self, scores, y):
        self.sigmoids = [LogisticRegression(C=1e6).fit(scores[:, [i]], y == i) for i in range(scores.shape[1])]
        return self

    def transform(self, scores):
        p = np.column_stack([m.predict_proba(scores[:, [i]])[:, 1] for i, m in enumerate(self.sigmoids)])
        return p / p.sum(axis=1, keepdims=True)


def similarity_features(query_vector, pool_vectors):
    """[top-1 cosine, mean of the top 5 cosines, top-1 minus top-2, pool size] for one query.

    Small pools: the mean uses the members there are, the gap is 0 with a single member, and an empty
    pool gives all zeros (only #313513 has one).
    """
    n = len(pool_vectors)
    if n == 0:
        return [0.0, 0.0, 0.0, 0]
    cosines = pool_vectors @ query_vector
    top = np.sort(np.partition(cosines, max(n - 5, 0))[max(n - 5, 0):])[::-1]
    gap = top[0] - top[1] if n > 1 else 0.0
    return [float(top[0]), float(top.mean()), float(gap), n]


def fusion_model(C, class_weight=None):
    """Multinomial logistic regression on standardised inputs (pool size and probabilities have very different scales)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, class_weight=class_weight, max_iter=5000))


def expected_calibration_error(probabilities, y, bins=15):
    """Average gap between confidence and accuracy, over equal-width confidence bins weighted by their size."""
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == y
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        in_bin = (confidence > low) & (confidence <= high)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)
