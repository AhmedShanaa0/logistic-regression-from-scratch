"""Run with:  python -m pytest -q"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data import fit_scaler, load_data, stratified_split, transform
from src.model import cost, fit, gradient, sigmoid


def test_sigmoid_is_stable_and_bounded():
    z = np.array([-1e6, -1.0, 0.0, 1.0, 1e6])
    s = sigmoid(z)
    assert np.all(np.isfinite(s)) and np.all((s >= 0) & (s <= 1))
    assert sigmoid(np.array([0.0]))[0] == 0.5


def test_gradient_matches_numerical_gradient():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 4))
    y = (rng.random(40) > 0.5).astype(float)
    w, b, lam = rng.normal(size=4), 0.3, 2.0
    dw, db = gradient(x, y, w, b, lam)
    eps = 1e-6
    for j in range(4):
        e = np.zeros(4); e[j] = eps
        num = (cost(x, y, w + e, b, lam) - cost(x, y, w - e, b, lam)) / (2 * eps)
        assert abs(num - dw[j]) < 1e-6
    num_b = (cost(x, y, w, b + eps, lam) - cost(x, y, w, b - eps, lam)) / (2 * eps)
    assert abs(num_b - db) < 1e-6


def test_stratified_split_keeps_class_ratio():
    _, y, _ = load_data()
    for idx in stratified_split(y, [0.6, 0.2, 0.2]):
        assert abs(y[idx].mean() - y.mean()) < 0.01


def test_matches_sklearn_predictions():
    X, y, _ = load_data()
    tr, _, te = stratified_split(y, [0.6, 0.2, 0.2])
    mu, sd = fit_scaler(X[tr])
    Xtr, Xte = transform(X[tr], mu, sd), transform(X[te], mu, sd)
    w, b, _ = fit(Xtr, y[tr], lambda_=1.0, verbose=False)
    clf = LogisticRegression(C=1.0, max_iter=10000).fit(Xtr, y[tr])
    mine = (Xte @ w + b >= 0).astype(float)
    assert np.mean(mine == clf.predict(Xte)) > 0.98
    assert np.allclose(w, clf.coef_[0], atol=2e-2)
