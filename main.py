"""Train and evaluate logistic regression from scratch on Breast Cancer Wisconsin.

Usage:
    python main.py                      # defaults, saves figures to ./figures
    python main.py --l2 10 --alpha 0.1  # different regularization / learning rate
    python main.py --no-plots           # skip figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression  # only used to verify the model

from src.data import fit_scaler, load_data, stratified_kfold, stratified_split, transform
from src.metrics import fbeta, metrics
from src.model import fit, predict_proba
from src.plots import plot_confusion, plot_cost, plot_thresholds, plot_weights


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--alpha", type=float, default=0.5, help="learning rate (default 0.5)")
    p.add_argument("--l2", dest="lambda_", type=float, default=1.0, help="L2 strength (default 1.0)")
    p.add_argument("--max-iter", type=int, default=20000, help="max iterations (default 20000)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--folds", type=int, default=5, help="cross-validation folds (default 5)")
    p.add_argument("--no-plots", action="store_true", help="do not save figures")
    p.add_argument("--fig-dir", type=Path, default=Path("figures"), help="figure folder")
    return p.parse_args()


def section(title: str) -> None:
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def print_metrics(label: str, m: dict) -> None:
    print(f"{label}")
    print(f"  accuracy={m['accuracy']:.3f}  precision={m['precision']:.3f}  "
          f"recall={m['recall']:.3f}  f1={m['f1']:.3f}")
    print(f"  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}")


def main() -> None:
    args = parse_args()

    # ---- data -------------------------------------------------------------
    section("1. Data")
    X, y, feature_names = load_data()
    print(f"samples={X.shape[0]}  features={X.shape[1]}  malignant ratio={y.mean():.3f}")
    print(f"baseline accuracy (always benign): {(y == 0).mean():.3f}")

    train_idx, val_idx, test_idx = stratified_split(y, [0.6, 0.2, 0.2], seed=args.seed)
    for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"  {name:<5}: {len(idx):3d} rows | malignant ratio = {y[idx].mean():.3f}")

    mu, sigma = fit_scaler(X[train_idx])
    X_train, y_train = transform(X[train_idx], mu, sigma), y[train_idx]
    X_val, y_val = transform(X[val_idx], mu, sigma), y[val_idx]
    X_test, y_test = transform(X[test_idx], mu, sigma), y[test_idx]

    # ---- training ---------------------------------------------------------
    section("2. Training")
    w, b, history = fit(X_train, y_train, alpha=args.alpha, max_iter=args.max_iter,
                        lambda_=args.lambda_)
    assert not np.isnan(history).any(), "cost became NaN"

    # ---- threshold selection on validation -------------------------------
    section("3. Threshold selection (validation set, maximize F2)")
    p_val = predict_proba(X_val, w, b)
    print_metrics("Validation @ 0.50", metrics(y_val, (p_val >= 0.5).astype(float)))

    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        r = metrics(y_val, (p_val >= t).astype(float))
        rows.append((t, r["precision"], r["recall"], r["f1"], fbeta(r["precision"], r["recall"])))
    rows = np.array(rows)
    best_t = float(rows[np.argmax(rows[:, 4]), 0])
    print(f"\nselected threshold = {best_t:.2f}")

    # ---- final test evaluation -------------------------------------------
    section("4. Test set (used once)")
    p_test = predict_proba(X_test, w, b)
    for label, t in [("default 0.50", 0.5), (f"selected {best_t:.2f}", best_t)]:
        print_metrics(f"Test @ threshold {label}", metrics(y_test, (p_test >= t).astype(float)))
    print(f"\nbaseline accuracy (always benign): {(y_test == 0).mean():.3f}")
    print("note: ~115 test rows -> one patient moves recall by ~2 points; prefer the CV numbers.")

    # ---- cross-validation -------------------------------------------------
    section(f"5. Stratified {args.folds}-fold cross-validation @ 0.50")
    dev_idx = np.concatenate([train_idx, val_idx])
    X_dev, y_dev = X[dev_idx], y[dev_idx]
    scores: dict[str, list[float]] = {k: [] for k in ["accuracy", "precision", "recall", "f1"]}
    for tr, te in stratified_kfold(y_dev, k=args.folds, seed=args.seed):
        mu_f, sd_f = fit_scaler(X_dev[tr])
        w_f, b_f, _ = fit(transform(X_dev[tr], mu_f, sd_f), y_dev[tr], alpha=args.alpha,
                          max_iter=args.max_iter, lambda_=args.lambda_, verbose=False)
        pred = (predict_proba(transform(X_dev[te], mu_f, sd_f), w_f, b_f) >= 0.5).astype(float)
        r = metrics(y_dev[te], pred)
        for k in scores:
            scores[k].append(r[k])
    for k, v in scores.items():
        print(f"  {k:<9}: {np.mean(v):.3f} +/- {np.std(v):.3f}")

    # ---- verification against scikit-learn --------------------------------
    section("6. Verification against scikit-learn")
    clf = LogisticRegression(C=1.0 / args.lambda_, max_iter=10000).fit(X_train, y_train)
    print(f"sklearn test accuracy : {clf.score(X_test, y_test):.4f}")
    print(f"scratch test accuracy : {np.mean((p_test >= 0.5) == y_test):.4f}")
    print(f"max |w_scratch - w_sklearn| = {np.max(np.abs(w - clf.coef_[0])):.2e}")

    # ---- figures ----------------------------------------------------------
    if not args.no_plots:
        args.fig_dir.mkdir(parents=True, exist_ok=True)
        m = metrics(y_test, (p_test >= best_t).astype(float))
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        plot_cost(history, args.fig_dir / "cost_curve.png")
        plot_thresholds(rows, best_t, args.fig_dir / "threshold_tradeoff.png")
        plot_confusion(cm, best_t, args.fig_dir / "confusion_matrix.png")
        plot_weights(w, feature_names, args.fig_dir / "feature_weights.png")
        print(f"\nfigures saved to {args.fig_dir}/")


if __name__ == "__main__":
    main()
