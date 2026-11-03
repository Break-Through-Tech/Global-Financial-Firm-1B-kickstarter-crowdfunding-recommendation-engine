"""
One definition of how the Milestone 3/4 train and test splits are produced.

The parquet splits are gitignored, so a fresh clone does not have them.
`load_splits()` reads them if they exist and rebuilds them from the raw data
if they do not (about 40 seconds). Every notebook calls this instead of
carrying its own copy of the rebuild logic, so there is exactly one
definition and the copies cannot drift apart.

The rebuild is identical to notebook 05: clean, build features, split, then
fit the two stateful features (city frequency, goal vs subcategory median)
on the training half only and apply them to the test half.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.cleaning import clean
from src.features import add_location_frequency, add_relative_goal, build_features
from src.loading import load_raw
from src.modeling import chronological_split, random_split

RAW_ZIP = "data/DSI_kickstarterscrape_dataset_csv.zip"

SPLIT_FILES = {
    "random": ("m3_random_train.parquet", "m3_random_test.parquet"),
    "chronological": ("m3_chrono_train.parquet", "m3_chrono_test.parquet"),
}


def _fit_stateful(train: pd.DataFrame, test: pd.DataFrame):
    """Fit city/goal references on train only, apply to both halves."""
    train, city_ref = add_location_frequency(train)
    train, goal_ref = add_relative_goal(train)
    test, _ = add_location_frequency(test, reference=city_ref)
    test, _ = add_relative_goal(test, reference=goal_ref)
    return train, test


def build_splits(root: str | Path = ".."):
    """Rebuild both splits from the raw data. Returns a dict of (train, test)."""
    root = Path(root)
    raw = load_raw(root / RAW_ZIP)
    featured = build_features(clean(raw, verbose=False)[0])
    return {
        "random": _fit_stateful(*random_split(featured)),
        "chronological": _fit_stateful(*chronological_split(featured)),
    }


def load_splits(root: str | Path = "..", *, rebuild: bool = False):
    """Load the splits, building and caching them first if they are missing.

    `root` is the repository root relative to the calling notebook (notebooks
    live one level down, hence the default of "..").
    """
    root = Path(root)
    data_dir = root / "data"
    needed = [data_dir / f for pair in SPLIT_FILES.values() for f in pair]

    if rebuild or not all(p.exists() for p in needed):
        print("Splits not found - rebuilding from the raw data (about 40 seconds) ...")
        splits = build_splits(root)
        for name, (train, test) in splits.items():
            tr_name, te_name = SPLIT_FILES[name]
            train.to_parquet(data_dir / tr_name, index=False)
            test.to_parquet(data_dir / te_name, index=False)
        print("Splits written to data/.")
        return splits

    return {
        name: (pd.read_parquet(data_dir / tr), pd.read_parquet(data_dir / te))
        for name, (tr, te) in SPLIT_FILES.items()
    }
