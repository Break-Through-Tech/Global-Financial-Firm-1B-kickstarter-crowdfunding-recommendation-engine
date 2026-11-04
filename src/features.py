"""
Feature preparation (Milestone 1, Task 3; feature sets locked in Task 6).

Every feature built here is tagged by when it becomes known:

  LAUNCH   -- knowable the moment the campaign goes live
  CAMPAIGN -- creator behaviour while the campaign runs (`updates`)
  LEAKY    -- an outcome, or a direct function of one; never used

Two feature sets are exported:

  launch_only   LAUNCH features only. The October baseline.
  with_updates  LAUNCH features plus `updates`. The final feature set
                (FINAL_FEATURE_SET), used by the final model and the
                recommendation prototype.

`updates` is the number of posts the creator made during the campaign, as
counted when it ended. Before launch, the prototype takes the creator's
*planned* number of updates in its place.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Leakage register -- columns that must never enter a model
# ---------------------------------------------------------------------------

LEAKY_COLUMNS = {
    "pledged": "Dollars raised. This IS the outcome.",
    "funded_percentage": "pledged / goal. Mechanically determines success.",
    "backers": "Count of people who pledged. Pure outcome.",
    "comments": "Audience-generated volume; scales with backers.",
    "status": "The raw target.",
    "success": "The encoded target.",
    "funded_date": "Campaign END date; unknown at launch.",
    "end_date": "Parsed campaign end; unknown at launch.",
    "url": "Identifier, not a feature.",
    "project_id": "Identifier, not a feature.",
    "location": "Superseded by city / state_raw.",
    "reward_levels": "Raw string; superseded by parsed tier features.",
    "name": "Raw string; superseded by parsed title features.",
}

# ---------------------------------------------------------------------------
# Reward tier parsing
# ---------------------------------------------------------------------------

# The reward_levels field looks like:
#   "$25,$50,$100,$250,$500,$1,000,$2,500"
# It is comma-separated AND the amounts contain thousands separators, so a
# naive .split(",") turns "$1,000" into "$1" and "000". This pattern only
# consumes a comma when it is followed by exactly three digits, which
# resolves the ambiguity correctly in both directions:
#   "$1,$5"    -> 1, 5        (comma is a separator)
#   "$1,000"   -> 1000        (comma is a thousands mark)
REWARD_PATTERN = re.compile(r"\$\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?|\d+(?:\.\d{1,2})?)")


def parse_reward_levels(raw: str) -> list[float]:
    """Extract the numeric reward tiers from one reward_levels string.

    Args:
        raw: A string like "$25,$50,$1,000". Non-strings are treated as empty.

    Returns:
        The tier amounts as floats, in the order they appear.
    """
    if not isinstance(raw, str):
        return []
    return [float(m.replace(",", "")) for m in REWARD_PATTERN.findall(raw)]


def add_reward_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn the reward_levels string into tier statistics. All LAUNCH.

    Args:
        df: Table with ``reward_levels``, ``goal`` and ``levels``.

    Returns:
        A copy with tier count, min/max/median tier, spread, entry price as a
        share of the goal, logged min/max tier and ``tier_count_mismatch``
        (parsed count differs from the scraper's ``levels``).
    """
    df = df.copy()
    parsed = df["reward_levels"].map(parse_reward_levels)

    df["n_tiers_parsed"] = parsed.map(len)
    df["min_tier"] = parsed.map(lambda t: min(t) if t else np.nan)
    df["max_tier"] = parsed.map(lambda t: max(t) if t else np.nan)
    df["median_tier"] = parsed.map(lambda t: float(np.median(t)) if t else np.nan)
    df["tier_spread"] = df["max_tier"] - df["min_tier"]

    # Entry price relative to the ask: a $1 tier on a $50,000 goal is a very
    # different strategy from a $1 tier on a $500 goal.
    df["min_tier_pct_of_goal"] = df["min_tier"] / df["goal"]
    df["log_min_tier"] = np.log1p(df["min_tier"])
    df["log_max_tier"] = np.log1p(df["max_tier"])

    # Cross-check against the scraper's own `levels` count. Disagreement
    # flags rows where the reward string is truncated or malformed -- worth
    # inspecting in the EDA notebook rather than trusting blindly.
    df["tier_count_mismatch"] = df["n_tiers_parsed"] != df["levels"]
    return df


# ---------------------------------------------------------------------------
# Goal, duration, timing, title
# ---------------------------------------------------------------------------

def add_goal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Goal transforms. All LAUNCH.

    Goal is heavily right-skewed (median $4,000, max in the millions), so
    log1p is the version any linear model should see.

    Args:
        df: Table with ``goal`` and ``levels``.

    Returns:
        A copy with ``log_goal``, ``goal_per_tier``, ``log_goal_per_tier``
        and ``goal_is_round_1k``.
    """
    df = df.copy()
    df["log_goal"] = np.log1p(df["goal"])
    df["goal_per_tier"] = df["goal"] / df["levels"].replace(0, np.nan)
    df["log_goal_per_tier"] = np.log1p(df["goal_per_tier"])
    # Round-number asks ($5,000 vs $4,750) may signal deliberation.
    df["goal_is_round_1k"] = (df["goal"] % 1000 == 0).astype("int8")
    return df


def add_duration_features(df: pd.DataFrame) -> pd.DataFrame:
    """Duration transforms. All LAUNCH.

    Args:
        df: Table with ``duration`` and ``goal``.

    Returns:
        A copy with ``duration_days``, ``is_30_day`` (the platform default),
        ``duration_bucket``, ``goal_per_day`` and ``log_goal_per_day``.
    """
    df = df.copy()
    df["duration_days"] = df["duration"]
    # 30 days is Kickstarter's default and by far the modal choice; being on
    # the default is itself informative.
    df["is_30_day"] = df["duration"].between(29.5, 30.5).astype("int8")
    df["duration_bucket"] = pd.cut(
        df["duration"],
        bins=[0, 20, 30, 45, 60, 100],
        labels=["<=20d", "21-30d", "31-45d", "46-60d", ">60d"],
    )
    df["goal_per_day"] = df["goal"] / df["duration"].replace(0, np.nan)
    df["log_goal_per_day"] = np.log1p(df["goal_per_day"])
    return df


def add_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Launch-timing features. All LAUNCH (they use launch_date, not end_date).

    Args:
        df: Table with a datetime ``launch_date``.

    Returns:
        A copy with launch year, month, weekday, quarter and a weekend flag.
    """
    df = df.copy()
    ld = df["launch_date"]
    df["launch_year"] = ld.dt.year
    df["launch_month"] = ld.dt.month
    df["launch_dow"] = ld.dt.dayofweek
    df["launch_quarter"] = ld.dt.quarter
    df["launch_is_weekend"] = ld.dt.dayofweek.isin([5, 6]).astype("int8")
    return df


def add_title_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cheap text features from the project title. All LAUNCH.

    Args:
        df: Table with ``name``. Missing titles count as empty strings.

    Returns:
        A copy with title length, word count, "!" and digit flags, and the
        share of letters that are upper case.
    """
    df = df.copy()
    name = df["name"].astype("string").fillna("")
    df["name_length"] = name.str.len()
    df["name_word_count"] = name.str.split().map(len)
    df["name_has_exclaim"] = name.str.contains("!", regex=False).astype("int8")
    df["name_has_digit"] = name.str.contains(r"\d", regex=True).astype("int8")
    # Shouty titles: share of alphabetic characters that are uppercase.
    alpha = name.str.count(r"[A-Za-z]").replace(0, np.nan)
    df["name_upper_ratio"] = (name.str.count(r"[A-Z]") / alpha).fillna(0)
    return df


# ---------------------------------------------------------------------------
# Categorical handling
# ---------------------------------------------------------------------------

def add_location_frequency(df: pd.DataFrame, reference: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Frequency-encode city, which has far too many levels to one-hot.

    Fit the frequency table on the training split and apply it to the test
    split. Fitting it on the full dataset would let test rows influence a
    feature value.

    Args:
        df: Table with ``city`` and ``state_raw``.
        reference: Campaign counts per "city, state" from a previous call.
            If None, the counts are computed from ``df`` (the fit step).

    Returns:
        ``(df, reference)``: a copy with ``city_project_count`` (0 for
        cities not in the reference) and ``log_city_project_count``, plus the
        reference that was used.
    """
    df = df.copy()
    city_key = df["city"].astype("string") + ", " + df["state_raw"].astype("string")
    if reference is None:
        reference = city_key.value_counts()
    df["city_project_count"] = city_key.map(reference).fillna(0)
    df["log_city_project_count"] = np.log1p(df["city_project_count"])
    return df, reference


def add_relative_goal(df: pd.DataFrame, reference: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Goal relative to the median goal in the same subcategory.

    A $20,000 ask is ambitious for a Short Film and modest for a Video Game.
    Same fit/apply discipline as ``add_location_frequency``: the median must
    come from training data only.

    Args:
        df: Table with ``goal`` and ``subcategory``.
        reference: Median goal per subcategory from a previous call. If
            None, the medians are computed from ``df`` (the fit step).

    Returns:
        ``(df, reference)``: a copy with ``goal_vs_subcat_median`` and
        ``log_goal_vs_subcat_median``, plus the reference that was used.
    """
    df = df.copy()
    if reference is None:
        reference = df.groupby("subcategory", observed=True)["goal"].median()
    subcat_median = df["subcategory"].map(reference)
    df["goal_vs_subcat_median"] = df["goal"] / subcat_median
    df["log_goal_vs_subcat_median"] = np.log1p(df["goal_vs_subcat_median"])
    return df, reference


# ---------------------------------------------------------------------------
# Master builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply every stateless feature transform in order.

    The two stateful transforms (``add_location_frequency``,
    ``add_relative_goal``) are deliberately NOT called here: they need a
    train/test split to be done honestly (see ``src.data.fit_stateful``).

    Args:
        df: The cleaned table from ``src.data.clean``.

    Returns:
        A copy with reward, goal, duration, timing and title features added.
    """
    df = add_reward_features(df)
    df = add_goal_features(df)
    df = add_duration_features(df)
    df = add_timing_features(df)
    df = add_title_features(df)
    return df


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

LAUNCH_NUMERIC = [
    "log_goal",
    "goal_is_round_1k",
    "duration_days",
    "is_30_day",
    "log_goal_per_day",
    "levels",
    "n_tiers_parsed",
    "log_min_tier",
    "log_max_tier",
    "median_tier",
    "tier_spread",
    "min_tier_pct_of_goal",
    "log_goal_per_tier",
    "launch_year",
    "launch_month",
    "launch_dow",
    "launch_is_weekend",
    "name_length",
    "name_word_count",
    "name_has_exclaim",
    "name_has_digit",
    "name_upper_ratio",
]

LAUNCH_CATEGORICAL = ["category", "subcategory", "state_raw"]

# Creator behaviour during the campaign. `updates` is the single strongest
# predictor in the dataset (success rate climbs from ~20% at 0 updates to
# ~92% at 13+). It is counted at the END of the campaign, so causation runs
# partly backwards: a campaign that is funding well gets a creator who keeps
# posting. The final model uses it; see notebooks/02_modeling.ipynb, Task 6.
CAMPAIGN_NUMERIC = ["updates"]

FEATURE_SETS = ("launch_only", "with_updates")

# The feature set the final model and the recommendation prototype use.
FINAL_FEATURE_SET = "with_updates"

# Fitted (stateful) numeric features added to every model matrix. They are
# computed by src.data.fit_stateful, not by build_features.
EXTRA_NUMERIC = ["log_city_project_count", "log_goal_vs_subcat_median"]


def feature_columns(feature_set: str = FINAL_FEATURE_SET) -> tuple[list[str], list[str]]:
    """Name the columns in a feature set.

    Args:
        feature_set: "launch_only" or "with_updates".

    Returns:
        ``(numeric_columns, categorical_columns)``, before one-hot encoding
        and without ``EXTRA_NUMERIC``.

    Raises:
        ValueError: If the feature set name is unknown.
    """
    if feature_set == "launch_only":
        return list(LAUNCH_NUMERIC), list(LAUNCH_CATEGORICAL)
    if feature_set == "with_updates":
        return list(LAUNCH_NUMERIC) + list(CAMPAIGN_NUMERIC), list(LAUNCH_CATEGORICAL)
    raise ValueError(f"Unknown feature_set {feature_set!r}; use one of {FEATURE_SETS}.")


def build_model_matrix(
    df: pd.DataFrame,
    feature_set: str = FINAL_FEATURE_SET,
    *,
    extra_numeric: list[str] | None = None,
    drop_first: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """One-hot encode the categoricals and return the model matrix.

    Cardinality is modest after cleaning (13 categories, 49 subcategories,
    51 states), so one-hot encoding is fine and keeps Logistic Regression
    coefficients readable. City is frequency-encoded instead; pass
    ``EXTRA_NUMERIC`` via ``extra_numeric`` to include it.

    Args:
        df: Featured table (needs ``success``).
        feature_set: "launch_only" or "with_updates".
        extra_numeric: More numeric columns to include. Names absent from
            ``df`` are skipped silently.
        drop_first: Drop the first level of each one-hot group.

    Returns:
        ``(X, y)``: an all-float64 matrix with missing values filled by the
        column median of ``df`` itself, and the ``success`` column.

    Raises:
        KeyError: If a feature-set column is missing from ``df``.
        ValueError: If a column in ``LEAKY_COLUMNS`` is requested.
    """
    numeric, categorical = feature_columns(feature_set)
    if extra_numeric:
        numeric = numeric + [c for c in extra_numeric if c in df.columns]

    missing = [c for c in numeric + categorical if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}. Did you run build_features()?")

    leaked = set(numeric + categorical) & set(LEAKY_COLUMNS)
    if leaked:
        raise ValueError(f"Leaky columns requested: {sorted(leaked)}")

    X = pd.get_dummies(
        df[numeric + categorical],
        columns=categorical,
        drop_first=drop_first,
        dtype="float64",
    )
    X = X.fillna(X.median(numeric_only=True))
    # Cast everything to plain float64. The date-derived features arrive as
    # pandas nullable Int dtypes, which scikit-learn cannot consume directly
    # (numpy raises "Cannot interpret 'Int64Dtype()' as a data type").
    X = X.astype("float64")
    y = df["success"]
    return X, y


def align_columns(X_train: pd.DataFrame, X_test: pd.DataFrame) -> pd.DataFrame:
    """Force a matrix to have exactly the training matrix's columns.

    One-hot encoding two halves separately produces different columns when a
    category appears in only one of them, a silent source of wrong results.

    Args:
        X_train: The matrix whose columns are the reference.
        X_test: The matrix to align.

    Returns:
        ``X_test`` reindexed to ``X_train``'s columns: missing columns are
        filled with 0 (the category was absent) and extra columns dropped.
    """
    return X_test.reindex(columns=X_train.columns, fill_value=0.0)


def model_matrices(train: pd.DataFrame, test: pd.DataFrame,
                   feature_set: str = FINAL_FEATURE_SET):
    """Build aligned train and test matrices for one feature set.

    The one call every notebook uses, so all models train on identical
    columns. ``EXTRA_NUMERIC`` is always included.

    Args:
        train: Training half from ``src.data.load_splits``.
        test: Test half.
        feature_set: "launch_only" or "with_updates".

    Returns:
        ``(X_train, y_train, X_test, y_test)`` with ``X_test`` aligned to
        ``X_train``'s columns.
    """
    X_train, y_train = build_model_matrix(train, feature_set, extra_numeric=EXTRA_NUMERIC)
    X_test, y_test = build_model_matrix(test, feature_set, extra_numeric=EXTRA_NUMERIC)
    return X_train, y_train, align_columns(X_train, X_test), y_test
