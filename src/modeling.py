"""
Splits and model constructors for Milestone 3.

Two splits are produced, not one:

  random        stratified 80/20 -- the standard comparison
  chronological train on earlier campaigns, test on later ones

The chronological split exists because the deployment scenario is
"predict a campaign that has not launched yet" using a model trained on
campaigns that already finished. A random split lets the model learn from
2012 campaigns to predict 2010 ones, which it could never do in practice.
If accuracy drops sharply under the chronological split, that gap is the
honest estimate of real-world performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def random_split(df: pd.DataFrame, target: str = "success"):
    """Stratified 80/20 split. Returns (train, test)."""
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[target],
    )
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def chronological_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Train on the earliest (1 - test_size) of campaigns by launch date.

    Returns (train, test). The cut date is reported by `split_summary` so it
    can be stated in the write-up.
    """
    ordered = df.sort_values("launch_date")
    cut = int(len(ordered) * (1 - test_size))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def split_summary(train: pd.DataFrame, test: pd.DataFrame, name: str) -> pd.DataFrame:
    """One row describing a split, for the record."""
    return pd.DataFrame([{
        "split": name,
        "train_n": len(train),
        "test_n": len(test),
        "train_success_rate": round(train["success"].mean(), 4),
        "test_success_rate": round(test["success"].mean(), 4),
        "train_end": train["launch_date"].max().date(),
        "test_start": test["launch_date"].min().date(),
    }])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_models(random_state: int = RANDOM_STATE) -> dict:
    """The three classifiers the brief asks for, with sane starting settings.

    Logistic Regression and KNN are wrapped in a scaler pipeline because both
    are distance- or magnitude-sensitive; Random Forest is not, so scaling it
    would only cost time. Wrapping in a Pipeline also means the scaler is
    fitted on training folds only during cross-validation, which is the
    correct behaviour and easy to get wrong by hand.
    """
    return {
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=random_state)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            random_state=random_state,
            n_jobs=-1,
        ),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=25, n_jobs=-1)),
        ]),
    }


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Force the test matrix to match the training columns exactly.

    One-hot encoding the two halves separately can produce different columns
    when a category appears in only one of them -- a silent source of wrong
    results. Missing columns are filled with 0 (the category was absent),
    and columns unseen in training are dropped.
    """
    return X_test.reindex(columns=X_train.columns, fill_value=0.0)
