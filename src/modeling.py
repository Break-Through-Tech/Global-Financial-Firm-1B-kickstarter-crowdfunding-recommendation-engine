"""
Models and the one evaluation harness (Milestone 3, Tasks 7-10).

Every model in the project is built here and scored by ``evaluate``, so the
results from three people's notebook sections stack into one table without
reconciliation. The evaluation functions RETURN data rather than printing it,
because printed metrics cannot be collected into a comparison table.

The final model is ``make_final_model()``: a tuned Random Forest with isotonic
calibration, trained on the ``with_updates`` feature set
(``src.features.FINAL_FEATURE_SET``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data import RANDOM_STATE

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_models(random_state: int = RANDOM_STATE) -> dict:
    """Build the three classifiers the brief asks for, unfitted.

    Logistic Regression and KNN are wrapped in a scaler pipeline because both
    are sensitive to feature magnitude; Random Forest is not. The pipeline also
    keeps the scaler fitted on training folds only during cross-validation.

    Args:
        random_state: Seed for the models that use one.

    Returns:
        ``{"Logistic Regression": ..., "Random Forest": ..., "KNN": ...}``
        with the team's default settings (C=1.0; 400 trees with
        min_samples_leaf=5; k=25).
    """
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, random_state=random_state, n_jobs=-1,
        ),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
        ]),
    }


FINAL_MODEL_NAME = "Random Forest (tuned + calibrated)"

# Settings chosen by the randomized search in notebooks/02_modeling.ipynb
# (Task 9) on the `with_updates` feature set, and written to
# reports/m3_tuned_config.json. Kept here so every notebook builds the
# identical model; Task 10 checks the two agree.
FINAL_RF_PARAMS = dict(n_estimators=400, min_samples_leaf=3, max_features=0.3,
                       max_depth=12)


def make_final_model(random_state: int = RANDOM_STATE):
    """Build the final model: the tuned Random Forest with isotonic calibration.

    A Random Forest averages tree votes, so its raw probabilities do not mean
    what they say. The prototype shows the probability to a person, so they
    have to. ``CalibratedClassifierCV`` fits the forest on two-thirds of the
    training data and a calibration map on the rest, three times over, and
    averages. Isotonic (a free-form increasing step function) replaced the
    earlier sigmoid when `updates` entered the model: the probabilities became
    bumpy in a way one S-curve cannot follow, and sigmoid calibration made the
    worst decile worse, not better (Task 10, section 10.2). Ranking (ROC-AUC)
    is essentially unchanged.

    Args:
        random_state: Seed for the forest.

    Returns:
        An unfitted ``CalibratedClassifierCV``. Fit it on a matrix built from
        ``FINAL_FEATURE_SET`` (see ``src.features.model_matrices``).
    """
    forest = RandomForestClassifier(random_state=random_state, n_jobs=-1, **FINAL_RF_PARAMS)
    return CalibratedClassifierCV(forest, method="isotonic", cv=3)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

# The five metrics the challenge brief asks for, in reporting order.
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series, *, label: str,
             feature_set: str = "", split: str = "") -> pd.DataFrame:
    """Score one fitted model on held-out data.

    Args:
        model: A fitted classifier.
        X_test: Test matrix.
        y_test: True labels.
        label: Model name for the row.
        feature_set: Feature-set name for the row.
        split: Split name for the row.

    Returns:
        One row with the labels and the five ``METRICS``. ROC-AUC uses
        ``predict_proba`` if available, else ``decision_function``, else NaN.
    """
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        proba = model.decision_function(X_test)
    else:
        proba = None
    return pd.DataFrame([{
        "model": label,
        "feature_set": feature_set,
        "split": split,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if proba is not None else np.nan,
    }])


def majority_class_baseline(y_train: pd.Series, y_test: pd.Series) -> pd.DataFrame:
    """Score the floor every model must beat: always predict the majority class.

    Args:
        y_train: Training labels (the majority class is taken from these).
        y_test: Test labels.

    Returns:
        One row in the same shape as ``evaluate``, with ROC-AUC 0.5.
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
    """Confusion matrix with readable labels.

    Args:
        model: A fitted classifier.
        X_test: Test matrix.
        y_test: True labels.

    Returns:
        A 2x2 DataFrame indexed by actual outcome, with predicted outcomes as
        columns.
    """
    cm = confusion_matrix(y_test, model.predict(X_test))
    return pd.DataFrame(cm, index=["actual: failed", "actual: successful"],
                        columns=["predicted: failed", "predicted: successful"])


def roc_points(model, X_test, y_test) -> tuple[np.ndarray, np.ndarray, float]:
    """Points for a ROC curve.

    Args:
        model: A fitted classifier with ``predict_proba``.
        X_test: Test matrix.
        y_test: True labels.

    Returns:
        ``(false_positive_rates, true_positive_rates, roc_auc)``.
    """
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    return fpr, tpr, roc_auc_score(y_test, proba)


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    """Compare predicted and observed success in quantile bins.

    Args:
        p: Predicted probabilities.
        y: True labels (0/1).
        bins: Number of quantile bins; duplicate edges are merged.

    Returns:
        One row per bin with mean ``predicted``, ``observed`` rate, ``n`` and
        ``gap_pp`` (observed minus predicted, in percentage points).
    """
    q = pd.qcut(p, bins, duplicates="drop")
    c = (pd.DataFrame({"p": np.asarray(p), "y": np.asarray(y)}).groupby(q, observed=True)
         .agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size")))
    c["gap_pp"] = (c["observed"] - c["predicted"]) * 100
    return c


def describe_errors(model, X_test, y_test, context: pd.DataFrame) -> pd.DataFrame:
    """Break the test set down by outcome type (true/false positive/negative).

    A false "you will succeed" sends someone into a campaign that fails; a
    false "you will fail" discourages someone who would have made it.

    Args:
        model: A fitted classifier.
        X_test: Test matrix.
        y_test: True labels.
        context: The test rows with at least ``goal`` and ``duration_days``,
            indexed like ``X_test``.

    Returns:
        One row per outcome type with its count, median goal, median
        duration and share of the test set, largest first.
    """
    pred = model.predict(X_test)
    df = context.loc[X_test.index].copy()
    df["actual"] = y_test.values
    df["predicted"] = pred
    df["error_type"] = np.select(
        [(df.actual == 1) & (df.predicted == 1), (df.actual == 0) & (df.predicted == 0),
         (df.actual == 0) & (df.predicted == 1), (df.actual == 1) & (df.predicted == 0)],
        ["true positive", "true negative",
         "false positive (told to launch, would fail)",
         "false negative (discouraged, would succeed)"],
        default="unclassified",
    )
    summary = df.groupby("error_type", observed=True).agg(
        n=("actual", "size"), median_goal=("goal", "median"),
        median_duration=("duration_days", "median"))
    summary["pct"] = (100 * summary["n"] / len(df)).round(1)
    return summary.sort_values("n", ascending=False)
