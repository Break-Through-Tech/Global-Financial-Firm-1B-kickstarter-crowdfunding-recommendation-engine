"""
A single evaluation harness for every model in Milestone 3.

Why this exists: if three people each write their own metric code, Task 8's
comparison table becomes an argument about whose numbers are right. Every
model goes through `evaluate()` and comes back in the same shape, so results
concatenate into one table without reconciliation.

Design note: these functions RETURN data rather than printing it. Printed
metrics cannot be collected into a comparison table, and Task 8 needs exactly
that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# The five metrics the challenge brief asks for, in reporting order.
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def evaluate(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    label: str,
    feature_set: str = "",
    split: str = "",
) -> pd.DataFrame:
    """Score one fitted model. Returns a single-row DataFrame.

    `label`, `feature_set` and `split` are carried through so rows from
    different notebooks can be concatenated and still be identifiable.
    """
    pred = model.predict(X_test)
    # Not every classifier exposes predict_proba; fall back to decision
    # function, and to NaN AUC if neither exists.
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        proba = model.decision_function(X_test)
    else:
        proba = None

    row = {
        "model": label,
        "feature_set": feature_set,
        "split": split,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if proba is not None else np.nan,
    }
    return pd.DataFrame([row])


def majority_class_baseline(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """The floor every model must beat.

    Predicting the training majority class for every test row. On this
    dataset that is ~54.7% accuracy, so a model at 60% is worth far less
    than it sounds.
    """
    majority = int(y_train.mode().iloc[0])
    pred = np.full(len(y_test), majority)
    return pd.DataFrame([{
        "model": "Majority class",
        "feature_set": "—",
        "split": "—",
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": 0.5,
    }])


def confusion_frame(model, X_test, y_test) -> pd.DataFrame:
    """Confusion matrix with readable labels instead of a bare array."""
    cm = confusion_matrix(y_test, model.predict(X_test))
    return pd.DataFrame(
        cm,
        index=["actual: failed", "actual: successful"],
        columns=["predicted: failed", "predicted: successful"],
    )


def roc_points(model, X_test, y_test) -> tuple[np.ndarray, np.ndarray, float]:
    """(fpr, tpr, auc) for plotting ROC curves across models."""
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    return fpr, tpr, roc_auc_score(y_test, proba)


def comparison_table(*frames: pd.DataFrame, sort_by: str = "f1") -> pd.DataFrame:
    """Stack evaluation rows into one sorted comparison table."""
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(sort_by, ascending=False).reset_index(drop=True)


def describe_errors(model, X_test, y_test, context: pd.DataFrame) -> pd.DataFrame:
    """Where the model goes wrong, broken out by outcome type.

    Useful for Milestone 4: a false "you will succeed" sends someone into a
    campaign that fails, while a false "you will fail" discourages someone
    who would have made it. These are different harms and the team should
    decide which to prioritise avoiding.
    """
    pred = model.predict(X_test)
    df = context.loc[X_test.index].copy()
    df["actual"] = y_test.values
    df["predicted"] = pred
    df["error_type"] = np.select(
        [
            (df.actual == 1) & (df.predicted == 1),
            (df.actual == 0) & (df.predicted == 0),
            (df.actual == 0) & (df.predicted == 1),
            (df.actual == 1) & (df.predicted == 0),
        ],
        ["true positive", "true negative",
         "false positive (told to launch, would fail)",
         "false negative (discouraged, would succeed)"],
        default="unclassified",
    )
    summary = df.groupby("error_type", observed=True).agg(
        n=("actual", "size"),
        median_goal=("goal", "median"),
        median_duration=("duration_days", "median"),
    )
    summary["pct"] = (100 * summary["n"] / len(df)).round(1)
    return summary.sort_values("n", ascending=False)
