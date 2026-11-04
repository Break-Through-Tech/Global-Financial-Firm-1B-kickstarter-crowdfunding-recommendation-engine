"""
Loading, cleaning and splitting the Kickstarter scrape.

Milestone 1 (Tasks 1-2) and the split half of Milestone 3 (Task 6).

Three stages, in the order every notebook uses them:

1. ``load_raw``     -- read the scrape as distributed (Task 1)
2. ``clean``        -- filter, clean and validate it, logging every step (Task 2)
3. ``load_splits``  -- build features and the two train/test splits (Task 6)

Loading gotchas
    The raw file is NOT UTF-8. It contains byte 0x92 (a Windows smart quote)
    and byte 0x8f, which is undefined in cp1252. ``latin-1`` decodes every
    byte without raising, so it is the only safe choice. The affected bytes
    sit inside project names, which are only used for cheap title features.
    Column names contain spaces ("project id", "funded percentage"); they are
    normalised to snake_case once, at load time.

Cleaning
    Every transformation appends a row to a step log, so the notebook can
    print an auditable trail of "rows in -> rows out, and why". Deliberate
    deviations from the original project's recipe are marked DEVIATION.

Splits
    Two splits are produced, not one. ``random`` is a stratified 80/20 split
    (the standard comparison). ``chronological`` trains on earlier campaigns
    and tests on the latest 20%, which is the deployment scenario: a random
    split lets the model learn from 2012 campaigns to predict 2010 ones. The
    two stateful features (city frequency, goal vs subcategory median) are
    fitted on the training half only. Rebuilding everything from the raw zip
    takes a few seconds, so nothing is cached to disk.
"""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.features import add_location_frequency, add_relative_goal, build_features

# ===========================================================================
# Loading (Task 1)
# ===========================================================================

RAW_ENCODING = "latin-1"

# Canonical snake_case names, in the raw file's column order.
COLUMN_RENAMES = {
    "project id": "project_id",
    "name": "name",
    "url": "url",
    "category": "category",
    "subcategory": "subcategory",
    "location": "location",
    "status": "status",
    "goal": "goal",
    "pledged": "pledged",
    "funded percentage": "funded_percentage",
    "backers": "backers",
    "funded date": "funded_date",
    "levels": "levels",
    "reward levels": "reward_levels",
    "updates": "updates",
    "comments": "comments",
    "duration": "duration",
}

# Default location of the dataset relative to the repository root.
DEFAULT_PATH = Path("data/DSI_kickstarterscrape_dataset_csv.zip")


def load_raw(path: str | Path = DEFAULT_PATH) -> pd.DataFrame:
    """Read the raw scrape exactly as distributed, with no filtering.

    Args:
        path: The .csv, or the .zip that contains it. pandas reads
            single-member zip archives natively, which is why the 4.4 MB zip
            is committed instead of the 12.9 MB csv.

    Returns:
        The raw table with snake_case column names.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If any expected column is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Download the dataset from "
            "https://www.kaggle.com/parienza/kickstarter and place the csv or "
            "zip in the data/ directory."
        )

    df = pd.read_csv(path, encoding=RAW_ENCODING)
    df = df.rename(columns=COLUMN_RENAMES)

    # Fail loudly if the file is not the one we documented against, rather
    # than letting a differently-shaped file flow silently downstream.
    expected = set(COLUMN_RENAMES.values())
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected schema -- missing columns: {sorted(missing)}")

    return df


def raw_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Summarise each column of a table.

    Args:
        df: Any DataFrame, usually the output of ``load_raw``.

    Returns:
        One row per column with its dtype, null count, null percentage,
        number of distinct values and one example value.
    """
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "nulls": int(s.isna().sum()),
                "null_pct": round(100 * s.isna().mean(), 2),
                "n_unique": int(non_null.nunique()),
                "example": (str(non_null.iloc[0])[:48] if len(non_null) else ""),
            }
        )
    return pd.DataFrame(rows)


# ===========================================================================
# Cleaning (Task 2)
# ===========================================================================

# 50 states + DC + the five inhabited territories. An explicit whitelist
# beats "last token is 2 characters long" because it also rejects genuine
# non-state tokens, but it MUST be applied case-insensitively: Montana
# appears in the raw data as both "MT" (89 rows) and "Mt" (36 rows).
US_STATES = {
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
    "DC", "PR", "VI", "GU", "AS", "MP",
}

# Statuses that represent a resolved binary outcome. live/canceled/suspended
# are dropped: "live" has no outcome yet, and canceled/suspended campaigns
# ended for reasons unrelated to whether they would have funded.
BINARY_STATUSES = ("successful", "failed")

# 2**31 - 1 in cents. Appears once as a goal value and is an integer
# overflow artefact from the scraper, not a real fundraising target.
INT32_SENTINEL = 21474836.47

# Kickstarter capped campaign length at 60 days in June 2011. Longer
# durations are legitimate but indicate a pre-2011 campaign, which is a
# different regime -- we flag rather than drop.
KICKSTARTER_MAX_DURATION_DAYS = 60


class StepLog:
    """Records row counts before and after each cleaning step."""

    def __init__(self) -> None:
        """Start an empty log."""
        self.steps: list[dict] = []

    def record(self, name: str, before: int, after: int, note: str = "") -> None:
        """Append one step.

        Args:
            name: Short step name.
            before: Rows going in.
            after: Rows coming out.
            note: Why the step exists.
        """
        self.steps.append(
            {
                "step": name,
                "rows_before": before,
                "rows_after": after,
                "removed": before - after,
                "note": note,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the log as a DataFrame, one row per step."""
        return pd.DataFrame(self.steps)


# ---------------------------------------------------------------------------
# Individual cleaning operations
# ---------------------------------------------------------------------------

def unescape_html_columns(df: pd.DataFrame, columns=("category", "subcategory", "name")) -> pd.DataFrame:
    """Decode HTML entities that the scraper never unescaped.

    This is the highest-impact single fix in the whole pipeline. The raw file
    contains BOTH "Film &amp; Video" (10,925 rows) and "Film & Video" (411
    rows) as separate category strings -- the same category, split in two,
    with the smaller group looking like a rare category to any model.

    Also affects subcategories "Board &amp; Card Games" and
    "Country &amp; Folk". Fixing it takes categories 14 -> 13 and
    subcategories 51 -> 49.

    Args:
        df: Table to fix.
        columns: Columns to unescape; any that are absent are skipped.

    Returns:
        A copy with the listed columns as unescaped strings.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].astype("string").map(
                lambda v: html.unescape(v) if pd.notna(v) else v
            )
    return df


def split_location(df: pd.DataFrame) -> pd.DataFrame:
    """Split `location` into `city` and `state_raw`.

    The field is mostly "City, ST" but not always. Of 44,635 non-null
    values, 8 do not have exactly one comma:

      - "St Croix, Virgin Islands, U.S."      (US territory, two commas)
      - "Nakagyo Ward, Kyoto, Japan"          (international, two commas)
      - "10, Middleburg, MD"                  (malformed, leading number)
      - "Maar"                                (no comma at all)

    Splitting on the LAST comma handles the malformed-leading-number case
    correctly and leaves the territory/international rows with a long
    `state_raw` that the US filter will reject.

    Args:
        df: Table with a ``location`` column.

    Returns:
        A copy with ``city`` and ``state_raw`` added. ``city`` is missing
        where the location has no comma.
    """
    df = df.copy()
    parts = df["location"].astype("string").str.rsplit(",", n=1)
    df["city"] = parts.str[0].str.strip()
    df["state_raw"] = parts.str[-1].str.strip()
    # A comma-less value like "Maar" yields city == state_raw; blank the city.
    no_comma = df["location"].notna() & ~df["location"].astype("string").str.contains(",", na=False)
    df.loc[no_comma, "city"] = pd.NA
    return df


def parse_funded_date(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `funded_date` and recover the launch date.

    `funded_date` is the campaign's END date (RFC-2822 format, e.g.
    "Fri, 19 Aug 2011 19:28:17 -0000"), NOT the launch date. Since
    `duration` is the campaign length in days, launch = end - duration.

    Recovering launch matters twice over: it gives launch-timing features
    (year, month, weekday) that are genuinely known at prediction time, and
    it lets you do a chronological train/test split instead of a random one.

    Args:
        df: Table with ``funded_date`` and ``duration``.

    Returns:
        A copy with ``end_date`` and ``launch_date`` (UTC timestamps) added.
        Unparseable dates become NaT.
    """
    df = df.copy()
    end = pd.to_datetime(df["funded_date"], format="%a, %d %b %Y %H:%M:%S %z", errors="coerce", utc=True)
    df["end_date"] = end
    df["launch_date"] = end - pd.to_timedelta(df["duration"], unit="D")
    return df


def flag_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Add boolean quality flags without dropping anything.

    Flagging rather than dropping keeps the decision visible and reversible.
    Only the goal sentinel is dropped later, in ``clean``.

    Args:
        df: Table with ``goal``, ``duration`` and ``end_date``.

    Returns:
        A copy with five ``flag_*`` columns (sentinel goal, goal under $100,
        goal over $1,000,000, duration over 60 days, unparseable date) and
        ``n_quality_flags``, their sum.
    """
    df = df.copy()
    df["flag_goal_sentinel"] = df["goal"].eq(INT32_SENTINEL)
    df["flag_goal_tiny"] = df["goal"].lt(100)
    df["flag_goal_huge"] = df["goal"].gt(1_000_000)
    df["flag_duration_over_cap"] = df["duration"].gt(KICKSTARTER_MAX_DURATION_DAYS)
    df["flag_bad_date"] = df["end_date"].isna()
    quality_cols = [c for c in df.columns if c.startswith("flag_")]
    df["n_quality_flags"] = df[quality_cols].sum(axis=1)
    return df


# ---------------------------------------------------------------------------
# The pipeline
# ---------------------------------------------------------------------------

def clean(
    df: pd.DataFrame,
    *,
    us_only: bool = True,
    dedupe: bool = True,
    drop_goal_sentinel: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full cleaning pipeline.

    Passing ``dedupe=False`` and ``drop_goal_sentinel=False`` reproduces the
    original project's table to within one row (38,492 against their 38,491),
    which is how the recipe was validated.

    Args:
        df: The raw table from ``load_raw``.
        us_only: Keep only rows whose state code is in ``US_STATES``.
        dedupe: Drop repeated ``project_id`` values, keeping the first.
        drop_goal_sentinel: Drop the $21,474,836.47 overflow goal.
        verbose: Print the step log and class balance.

    Returns:
        ``(clean_df, step_log)``: the cleaned table with an integer
        ``success`` target, and one row per cleaning step.
    """
    log = StepLog()
    n0 = len(df)
    log.record("raw", n0, n0, "as distributed")

    # --- Step 1: resolved binary outcomes only -----------------------------
    before = len(df)
    df = df[df["status"].isin(BINARY_STATUSES)].copy()
    log.record(
        "filter_status",
        before,
        len(df),
        "dropped live (no outcome yet), canceled, suspended",
    )

    # --- Step 2: unescape HTML before any grouping on category -------------
    df = unescape_html_columns(df)
    log.record("unescape_html", len(df), len(df), "Film &amp; Video -> Film & Video")

    # --- Step 3: split location, then filter to the US ---------------------
    df = split_location(df)
    if us_only:
        before = len(df)
        # DEVIATION 1: case-insensitive on purpose. Montana is spelled "MT" in most rows
        # and "Mt" in 36 others (Kalispell, Whitefish, Gallatin Gateway --
        # unambiguously Montana). A case-sensitive check silently deletes
        # those campaigns; a "len == 2" check keeps them but as a SEPARATE
        # state level, which is the same bug as `Film &amp; Video`: one real
        # entity split into two encoded categories.
        state_code = df["state_raw"].str.upper()
        df = df[state_code.isin(US_STATES)].copy()
        # Normalise so "Mt" and "MT" group as one state downstream.
        df["state_raw"] = df["state_raw"].str.upper()
        log.record(
            "filter_us",
            before,
            len(df),
            "state_raw in US whitelist, case-insensitive; codes upper-cased",
        )

    # --- Step 4: dates ------------------------------------------------------
    df = parse_funded_date(df)
    log.record("parse_dates", len(df), len(df), "recovered launch_date = end - duration")

    # --- Step 5: missing values --------------------------------------------
    # Only two columns still carry nulls at this point: `pledged` (a leaky
    # outcome column we will not model on) and `reward_levels`. Dropping on
    # ALL columns, as the original did, silently deletes rows for the sake of
    # a column that never enters the model.
    before = len(df)
    required = ["goal", "duration", "levels", "category", "subcategory", "city", "state_raw", "status"]
    df = df.dropna(subset=required).copy()
    log.record(
        "drop_missing_required",
        before,
        len(df),
        f"dropna on modelling columns only: {', '.join(required)}",
    )

    before = len(df)
    df = df[df["reward_levels"].notna()].copy()
    log.record("drop_missing_reward_levels", before, len(df), "needed for tier features")

    # --- Step 6: duplicates -------------------------------------------------
    if dedupe:
        before = len(df)
        df = df.drop_duplicates(subset=["project_id"], keep="first").copy()
        # DEVIATION 2: the original project did not dedupe. 113 duplicated
        # project_ids survive its filters, so our table is intentionally
        # ~113 rows smaller than their published 38,491.
        log.record(
            "dedupe_project_id",
            before,
            len(df),
            "DEVIATION: original project kept duplicates",
        )

    # --- Step 7: quality flags, then the one drop we can justify -----------
    df = flag_data_quality(df)
    if drop_goal_sentinel:
        before = len(df)
        df = df[~df["flag_goal_sentinel"]].copy()
        # DEVIATION 3: $21,474,836.47 is 2**31-1 cents -- a scraper overflow,
        # not a fundraising target. Keeping it distorts every goal statistic.
        log.record("drop_goal_sentinel", before, len(df), "int32 overflow artefact")

    # --- Step 8: the target -------------------------------------------------
    df["success"] = (df["status"] == "successful").astype("int8")
    log.record("add_target", len(df), len(df), "success = 1 if status == 'successful'")

    df = df.reset_index(drop=True)
    step_log = log.to_frame()

    if verbose:
        print(step_log.to_string(index=False))
        print(f"\nFinal: {len(df):,} rows x {df.shape[1]} columns")
        rate = df["success"].mean()
        print(f"Class balance: {rate:.1%} successful / {1 - rate:.1%} failed")
        print(f"Majority-class accuracy baseline: {max(rate, 1 - rate):.1%}")

    return df, step_log


# ===========================================================================
# Splits (Task 6)
# ===========================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20


def random_split(df: pd.DataFrame, target: str = "success"):
    """Split rows 80/20, stratified on the target.

    Args:
        df: The featured table.
        target: Column to stratify on.

    Returns:
        A ``(train, test)`` tuple of copies.
    """
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df[target],
    )
    return df.loc[train_idx].copy(), df.loc[test_idx].copy()


def chronological_split(df: pd.DataFrame, test_size: float = TEST_SIZE):
    """Train on the earliest campaigns by launch date, test on the latest.

    Args:
        df: The featured table (needs ``launch_date``).
        test_size: Share of the latest campaigns held out.

    Returns:
        A ``(train, test)`` tuple of copies.
    """
    ordered = df.sort_values("launch_date")
    cut = int(len(ordered) * (1 - test_size))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def split_summary(train: pd.DataFrame, test: pd.DataFrame, name: str) -> pd.DataFrame:
    """Describe one split in a single row.

    Args:
        train: Training half.
        test: Test half.
        name: Label for the split.

    Returns:
        One row: row counts, success rates, last training launch date and
        first test launch date.
    """
    return pd.DataFrame([{
        "split": name,
        "train_n": len(train),
        "test_n": len(test),
        "train_success_rate": round(train["success"].mean(), 4),
        "test_success_rate": round(test["success"].mean(), 4),
        "train_end": train["launch_date"].max().date(),
        "test_start": test["launch_date"].min().date(),
    }])


def fit_stateful(train: pd.DataFrame, test: pd.DataFrame):
    """Fit city frequency and subcategory goal medians on train, apply to both.

    Args:
        train: Training half (featured).
        test: Test half (featured).

    Returns:
        A ``(train, test)`` tuple with ``city_project_count``,
        ``log_city_project_count``, ``goal_vs_subcat_median`` and
        ``log_goal_vs_subcat_median`` added.
    """
    train, city_ref = add_location_frequency(train)
    train, goal_ref = add_relative_goal(train)
    test, _ = add_location_frequency(test, reference=city_ref)
    test, _ = add_relative_goal(test, reference=goal_ref)
    return train, test


def build_featured(root: str | Path = "..") -> pd.DataFrame:
    """Load, clean and feature the raw data.

    Args:
        root: Repository root relative to the caller. Notebooks live one level
            down, hence the default of "..".

    Returns:
        The cleaned table with every stateless feature added.
    """
    raw = load_raw(Path(root) / DEFAULT_PATH)
    return build_features(clean(raw, verbose=False)[0])


_SPLIT_CACHE: dict = {}


def load_splits(root: str | Path = ".."):
    """Build both train/test splits from the raw data.

    Results are cached for the rest of the Python session, so calling this
    from several places in one notebook only builds them once.

    Args:
        root: Repository root relative to the caller.

    Returns:
        ``{"random": (train, test), "chronological": (train, test)}``, with
        the stateful features fitted on each training half.
    """
    key = str(Path(root).resolve())
    if key not in _SPLIT_CACHE:
        featured = build_featured(root)
        _SPLIT_CACHE[key] = {
            "random": fit_stateful(*random_split(featured)),
            "chronological": fit_stateful(*chronological_split(featured)),
        }
    return {k: (tr.copy(), te.copy()) for k, (tr, te) in _SPLIT_CACHE[key].items()}
