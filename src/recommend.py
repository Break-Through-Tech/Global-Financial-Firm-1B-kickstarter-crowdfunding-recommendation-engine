"""
Milestone 4 -- turning predictions into advice (Tasks 11-15).

TWO MODELS, TWO JOBS
    * The SCORER is the final model: a tuned, calibrated Random Forest on the
      `with_updates` feature set (chosen in Milestone 3, Task 10). It answers
      "what is my chance?" and puts the campaign in a risk band.
    * The ADVISOR is the Logistic Regression from Milestone 3. It answers
      "how much would this change help?". A what-if engine needs a model whose
      response to a change is smooth and always points the same way; a forest
      is a sum of step functions, and when we tested it, cutting the goal
      LOWERED its prediction for 11-19% of held-out campaigns -- advice that
      contradicts itself. The advisor's estimated effect, in percentage
      points, is added to the scorer's probability:

          new chance = RF chance + (LR chance after - LR chance before)

      so every number a creator sees starts from the final model and moves in
      a consistent direction. A points shift (rather than a log-odds shift)
      is deliberate: where the advisor sees a campaign as hopeless (a
      $5,000,000 goal), its points change is tiny, so a forest that cannot
      tell "huge" from "very huge" is never pushed into promising a big gain.

Three further design decisions worth knowing about:

1. FEATURES ARE REBUILT FROM RAW INPUTS, never patched. Changing the goal
   changes log_goal, goal_per_day, goal_per_tier, min_tier_pct_of_goal and
   goal_vs_subcat_median together. Re-running the whole feature pipeline for
   every what-if keeps them consistent; nudging one column by hand does not.
   A test (`self_check`) proves the rebuild reproduces the training matrix.

2. OUTCOMES CAN NEVER ENTER. `pledged`, `backers`, `comments`,
   `funded_percentage` and `status` have no input slot, and passing one
   raises an error. The prototype is incapable of using them, which is
   stronger than promising not to.

3. `updates` IS AN INPUT, NOT A LEVER. The final model was trained on the
   number of updates each campaign posted by the time it ended. Before launch
   that number does not exist yet, so the creator supplies the number they
   PLAN to post. It is never recommended as a change: update counts partly
   reflect how well a campaign was already doing (a campaign funding well gets
   a creator who keeps posting), so "post more updates" may do nothing. The
   report shows how much the score leans on the planned figure.

4. ADVICE IS LIMITED TO WHAT A CREATOR CAN CHANGE before launch: goal,
   duration and reward structure. Category, subcategory, state and city inform
   the prediction but are never recommended.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features import (
    EXTRA_NUMERIC,
    FINAL_FEATURE_SET,
    add_duration_features,
    add_goal_features,
    add_location_frequency,
    add_relative_goal,
    add_reward_features,
    add_timing_features,
    add_title_features,
    align_columns,
    build_model_matrix,
    parse_reward_levels,
)
from src.modeling import FINAL_MODEL_NAME, make_final_model, make_models

EXTRA = EXTRA_NUMERIC

# Raw columns a campaign description must supply. Everything else is derived.
RAW_COLUMNS = [
    "goal", "duration", "levels", "reward_levels", "category", "subcategory",
    "state_raw", "city", "launch_date", "name", "updates",
]

TITLE_COLUMNS = ["name_length", "name_word_count", "name_has_exclaim",
                 "name_has_digit", "name_upper_ratio"]

# Outcome fields: they only exist once a campaign has finished, and most are
# the result itself. Passing any of them is an error, not something to ignore
# silently. (`updates` is deliberately NOT here: it is a required input, the
# creator's planned number of updates.)
OUTCOME_FIELDS = {"pledged", "backers", "comments", "funded_percentage",
                  "status", "success"}

# Features describing how a goal is written rather than how big it is. They are
# held at the campaign's own value in every what-if, so that rounding a
# suggested goal (e.g. $5,000 x 0.9 = $4,500) cannot by itself change the
# estimated effect. Found in Task 15: without this, a 10% goal cut lowered the
# advisor's estimate for about 9% of held-out campaigns, almost all of them
# campaigns whose round $1,000 goal stopped being round.
HELD_FIXED_IN_WHAT_IFS = ["goal_is_round_1k"]

# The scorer's chance is shown between these limits. Isotonic calibration
# outputs exactly 0 or 1 at the extremes, and no campaign in this data is
# certain to fail or to succeed (Task 14: a campaign scored 100% failed).
SHOWN_MIN, SHOWN_MAX = 0.01, 0.99

# Kickstarter has capped campaigns at 60 days since June 2011.
MIN_DURATION, MAX_DURATION = 1, 60


# ---------------------------------------------------------------------------
# Reward tier helpers
# ---------------------------------------------------------------------------

def tiers_to_string(tiers) -> str:
    """Format tier amounts the way the raw data does: "$5,$25,$1,000".

    The thousands separator is essential. The tier parser treats a comma
    followed by exactly three digits as a thousands mark, so "$1000" would be
    read as "$100" followed by a stray zero. Formatting with separators keeps
    the round trip lossless (checked in `self_check`).

    Args:
        tiers: Tier amounts, in any order.

    Returns:
        The amounts sorted and joined, e.g. "$5,$25,$1,000".
    """
    return ",".join(f"${t:,.0f}" if float(t).is_integer() else f"${t:,.2f}"
                    for t in sorted(tiers))


def string_to_tiers(s) -> list[float]:
    """Parse a "$5,$25,$1,000" string back into tier amounts.

    Args:
        s: A reward_levels string.

    Returns:
        The amounts as floats.
    """
    return parse_reward_levels(s)



# ---------------------------------------------------------------------------
# Policy constants -- every number the advice engine relies on lives here
# ---------------------------------------------------------------------------

# Smallest predicted gain worth recommending. The final model's held-out
# calibration is within a few points in every decile, so gains below this are
# inside the noise and are reported as "no meaningful change" rather than as
# advice.
MIN_GAIN = 0.02

# Risk bands, by predicted probability. Observed success rates for each band
# are measured on held-out data in `Recommender.fit(..., test=...)`, so the
# label always comes with what actually happened to campaigns scored there.
BANDS = [(0.35, "High risk"), (0.55, "Moderate risk"), (1.01, "Favourable")]

# Candidate changes, ordered mildest to strongest. The engine recommends the
# MILDEST candidate that clears MIN_GAIN, and shows all of them.
GOAL_FACTORS = [0.90, 0.75, 0.60, 0.50]
DURATION_TARGETS = [45, 35, 28]          # days; the platform default (30) is skipped
TIER_TARGETS = [6, 8, 10]                # total number of reward tiers
MIN_TIER_ADDITION = 2                    # advice must add at least this many tiers;
                                         # "add a tenth tier" to someone with nine is noise

# The 30-day platform default is deliberately never a target or a source of
# advice. Campaigns left at exactly 30 days succeed less than neighbouring
# durations even within the same goal size, but that is plausibly a statement
# about who accepts the default, not about the number of days.
DEFAULT_DURATION = (29.5, 30.5)
GOAL_FLOOR_QUANTILE = 0.10               # never suggest a goal below this subcategory quantile

# Amounts a real creator would plausibly offer, used to snap suggested tiers.
NICE_AMOUNTS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200,
                250, 300, 400, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000,
                7500, 10000, 15000, 20000, 25000, 50000, 100000]


def ordinal(n: int) -> str:
    """Format an integer as an English ordinal.

    Args:
        n: The number.

    Returns:
        For example '91st', '12th', '23rd'.
    """
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _snap(x: float, lo: float, hi: float, used: set):
    """Find the believable price nearest ``x`` (in log terms) inside a gap.

    Args:
        x: Target price.
        lo: Lower bound, exclusive.
        hi: Upper bound, exclusive.
        used: Amounts already taken.

    Returns:
        A value from ``NICE_AMOUNTS`` strictly between ``lo`` and ``hi`` and
        not in ``used``, or None if there is none.
    """
    cands = [a for a in NICE_AMOUNTS if lo < a < hi and a not in used]
    if cands:
        return float(min(cands, key=lambda a: abs(math.log(a) - math.log(x))))
    return None


def suggest_tiers(tiers, n: int) -> list[float]:
    """Grow a tier list to `n` entries by filling the widest price gaps.

    Existing tiers are kept untouched. Each new tier goes in the largest
    remaining gap (measured in log-price, since $5 to $10 is as big a jump as
    $500 to $1,000) and is snapped to a believable amount. If the gaps are
    exhausted, higher-priced tiers are added instead.

    Args:
        tiers: Current tier amounts.
        n: Target number of tiers.

    Returns:
        The sorted, de-duplicated tier list, grown towards ``n``. It can be
        shorter than ``n`` if no believable amount is left.

    Raises:
        ValueError: If ``tiers`` has no positive amount.
    """
    t = sorted({float(x) for x in tiers if float(x) > 0})
    if not t:
        raise ValueError("reward_tiers must contain at least one positive amount")
    if len(t) == 1:                               # single tier: add a top anchor
        top = max(t[0] * 20, 250.0)
        top = float(min((a for a in NICE_AMOUNTS if a >= top), default=top))
        t = sorted({t[0], top})
    while len(t) < n:
        gaps = sorted(range(len(t) - 1),
                      key=lambda i: math.log(t[i + 1]) - math.log(t[i]), reverse=True)
        for i in gaps:
            mid = math.sqrt(t[i] * t[i + 1])
            new = _snap(mid, t[i], t[i + 1], set(t))
            if new is not None:
                t = sorted(t + [new])
                break
        else:
            # No believable price point left between the extremes. Extend the
            # top instead of cramming in $12, $13, $15 -- a higher-priced
            # tier is the natural way to add an option.
            above = [a for a in NICE_AMOUNTS if a >= 2 * t[-1] and a not in t]
            if not above:
                break
            t = sorted(t + [float(above[0])])
    return t

# ---------------------------------------------------------------------------
# The recommender
# ---------------------------------------------------------------------------

class Recommender:
    """Final-model wrapper: featurise, predict, explain, advise.

    Build it with ``Recommender.fit`` or ``get_recommender`` rather than
    calling the constructor directly.
    """

    def __init__(self, model, advisor, columns, medians, city_ref, goal_ref,
                 train: pd.DataFrame, X_train: pd.DataFrame):
        """Store the fitted models and the training references.

        Args:
            model: Fitted scorer (the final model).
            advisor: Fitted Logistic Regression used for what-if effects.
            columns: Training matrix columns.
            medians: Training matrix column medians, used for missing inputs.
            city_ref: Campaign counts per "city, state" from training.
            goal_ref: Median goal per subcategory from training.
            train: The featured training rows.
            X_train: The training matrix.
        """
        self.model = model            # scorer: the final Random Forest
        self.advisor = advisor        # what-if effects: Logistic Regression
        self.model_name = FINAL_MODEL_NAME
        self.columns = list(columns)
        self.medians = medians
        self.city_ref = city_ref
        self.goal_ref = goal_ref
        self.train = train
        self.X_train = X_train
        self.year_range = (int(X_train["launch_year"].min()),
                           int(X_train["launch_year"].max()))
        self.default_launch_date = train["launch_date"].max()
        self.taxonomy = (train.groupby("category")["subcategory"]
                         .agg(lambda s: sorted(s.unique())).to_dict())
        self.states = sorted(train["state_raw"].unique())
        self.band_rates: dict = {}
        self.holdout_p = None
        self.holdout_y = None
        self.entry_stats = self._entry_price_stats()

    # -- construction -------------------------------------------------------

    @classmethod
    def fit(cls, train: pd.DataFrame, test: pd.DataFrame | None = None) -> "Recommender":
        """Fit the scorer (final Random Forest) and the advisor (Logistic
        Regression) on the (already featured) training split.

        If `test` is given, the observed success rate of held-out campaigns in
        each risk band is recorded, so every probability shown to a creator can
        be accompanied by what actually happened to campaigns scored like it.

        Args:
            train: Training half from ``src.data.load_splits``.
            test: Optional test half for measuring the risk bands.

        Returns:
            A fitted ``Recommender``.
        """
        X_train, y_train = build_model_matrix(train, FINAL_FEATURE_SET, extra_numeric=EXTRA)
        model = make_final_model()
        model.fit(X_train, y_train)
        advisor = make_models()["Logistic Regression"]
        advisor.fit(X_train, y_train)

        city_key = train["city"].astype("string") + ", " + train["state_raw"].astype("string")
        city_ref = city_key.value_counts()
        goal_ref = train.groupby("subcategory", observed=True)["goal"].median()
        rec = cls(model, advisor, X_train.columns, X_train.median(), city_ref, goal_ref,
                  train, X_train)
        if test is not None:
            rec.band_rates = rec._measure_bands(test)
        return rec

    # -- features -----------------------------------------------------------

    def featurize(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Raw campaign rows -> model matrix, using the same pipeline as training.

        Missing optional inputs fall back to training medians rather than to
        zero, so leaving out a city or a title does not silently penalise the
        campaign. Launch year is clipped to the training years.

        Args:
            raw: Rows with every column in ``RAW_COLUMNS``.

        Returns:
            A float64 matrix with the training columns, in the training order.
        """
        df = raw[RAW_COLUMNS].copy()
        df["launch_date"] = pd.to_datetime(df["launch_date"], utc=True)

        df = add_reward_features(df)
        df = add_goal_features(df)
        df = add_duration_features(df)
        df = add_timing_features(df)
        df = add_title_features(df)
        df, _ = add_location_frequency(df, reference=self.city_ref)
        df, _ = add_relative_goal(df, reference=self.goal_ref)

        X, _ = build_model_matrix(df.assign(success=0), FINAL_FEATURE_SET, extra_numeric=EXTRA)
        X = align_columns(self.X_train, X)

        # Optional inputs: use the training median instead of a made-up value.
        no_city = raw["city"].isna().to_numpy()
        no_name = raw["name"].isna().to_numpy()
        if no_city.any():
            X.loc[no_city, "log_city_project_count"] = self.medians["log_city_project_count"]
        if no_name.any():
            for c in TITLE_COLUMNS:
                X.loc[no_name, c] = self.medians[c]

        # The model has only seen 2009-2012. Extrapolating a linear year term
        # to the present day would be meaningless, so clip to the trained range.
        X["launch_year"] = X["launch_year"].clip(*self.year_range)

        return X.fillna(self.medians).astype("float64")

    def predict_proba(self, raw: pd.DataFrame) -> np.ndarray:
        """Score raw campaign rows with the final model (the scorer).

        Args:
            raw: Rows with every column in ``RAW_COLUMNS``.

        Returns:
            Probability of success for each row, kept between ``SHOWN_MIN``
            and ``SHOWN_MAX`` (1% and 99%).
        """
        p = self.model.predict_proba(self.featurize(raw))[:, 1]
        return np.clip(p, SHOWN_MIN, SHOWN_MAX)

    def advisor_proba(self, raw: pd.DataFrame) -> np.ndarray:
        """Score raw campaign rows with the advisor (Logistic Regression).

        Args:
            raw: Rows with every column in ``RAW_COLUMNS``.

        Returns:
            The advisor's probability of success for each row.
        """
        return self.advisor.predict_proba(self.featurize(raw))[:, 1]

    def advisor_shift(self, before: pd.DataFrame, after: pd.DataFrame) -> np.ndarray:
        """The advisor's change in probability from ``before`` to ``after``.

        Features in ``HELD_FIXED_IN_WHAT_IFS`` keep their ``before`` values.

        Args:
            before: Raw rows as they are.
            after: The same rows with one or more levers changed.

        Returns:
            ``advisor(after) - advisor(before)`` for each row.
        """
        Xb, Xa = self.featurize(before), self.featurize(after)
        Xa[HELD_FIXED_IN_WHAT_IFS] = Xb[HELD_FIXED_IN_WHAT_IFS].to_numpy()
        return self.advisor.predict_proba(Xa)[:, 1] - self.advisor.predict_proba(Xb)[:, 1]

    # Shown chances are kept inside this range after a what-if shift.
    P_FLOOR, P_CEIL = 0.005, 0.995

    def what_if_proba(self, before: pd.DataFrame, after: pd.DataFrame,
                      base: np.ndarray | None = None) -> np.ndarray:
        """Chance after a change, row by row.

        `before` and `after` are raw frames of equal length (row i of `after`
        is row i of `before` with one or more levers changed). The advisor's
        change in probability is added to the scorer's probability for
        `before`.
        Pass `base` (the scorer's probabilities for `before`) to skip
        re-scoring it.

        Args:
            before: Raw rows as they are.
            after: The same rows with one or more levers changed.
            base: Optional scorer probabilities for ``before``.

        Returns:
            ``base + advisor_shift(before, after)``, clipped to
            ``[P_FLOOR, P_CEIL]``.
        """
        if base is None:
            base = self.predict_proba(before)
        shift = self.advisor_shift(before, after)
        return np.clip(np.asarray(base, dtype=float) + shift, self.P_FLOOR, self.P_CEIL)



    # -- batch what-ifs -----------------------------------------------------

    def apply_lever(self, raw: pd.DataFrame, lever: str, value) -> pd.DataFrame:
        """Apply one lever to every row of a raw frame, for population-level checks.

        lever="goal"          value = multiplier (0.75 -> three-quarters of the goal)
        lever="duration"      value = new duration in days
        lever="reward tiers"  value = target number of tiers
        Uses the same rounding and tier logic as the single-campaign engine.

        Args:
            raw: Rows with every column in ``RAW_COLUMNS``.
            lever: "goal", "duration" or "reward tiers".
            value: See above.

        Returns:
            A changed copy of ``raw``.

        Raises:
            ValueError: If the lever name is unknown.
        """
        r = raw.copy()
        if lever == "goal":
            r["goal"] = (r["goal"] * value).map(lambda g: float(f"{g:.2g}"))
        elif lever == "duration":
            r["duration"] = float(value)
        elif lever == "reward tiers":
            new = [suggest_tiers(string_to_tiers(s), value) for s in r["reward_levels"]]
            r["reward_levels"] = [tiers_to_string(x) for x in new]
            r["levels"] = [len(x) for x in new]
        else:
            raise ValueError(f"Unknown lever {lever!r}")
        return r

    # -- bands and context --------------------------------------------------

    @staticmethod
    def band_of(p: float) -> str:
        """Name the risk band for a probability.

        Args:
            p: Predicted probability of success.

        Returns:
            "High risk", "Moderate risk" or "Favourable" (see ``BANDS``).
        """
        for upper, name in BANDS:
            if p < upper:
                return name
        return BANDS[-1][1]

    def _measure_bands(self, test: pd.DataFrame) -> dict:
        """Observed success per risk band on held-out rows; stores the scores.

        Args:
            test: Held-out rows with ``RAW_COLUMNS`` and ``success``.

        Returns:
            ``{band: {"observed_rate", "n", "range"}}`` for each non-empty band.
        """
        p = self.predict_proba(test[RAW_COLUMNS])
        y = test["success"].to_numpy()
        self.holdout_p, self.holdout_y = p, y
        out = {}
        for i, (upper, name) in enumerate(BANDS):
            lower = BANDS[i - 1][0] if i else 0.0
            m = (p >= lower) & (p < upper)
            if m.sum():
                out[name] = {"observed_rate": float(y[m].mean()), "n": int(m.sum()),
                             "range": (lower, min(upper, 1.0))}
        return out

    def local_rate(self, p: float, half_width: float = 0.05, min_n: int = 30):
        """Observed success rate of held-out campaigns scored near `p`.

        More honest than a band average: a campaign scored 88% should be
        compared with campaigns scored near 88%, not with everything above 55%.

        Args:
            p: The probability to look around.
            half_width: Half the width of the window.
            min_n: Fewest held-out campaigns needed to report a rate.

        Returns:
            ``{"low", "high", "rate", "n"}``, or None if the bands were never
            measured or the window holds fewer than ``min_n`` campaigns.
        """
        if self.holdout_p is None:
            return None
        lo, hi = max(0.0, p - half_width), min(1.0, p + half_width)
        m = (self.holdout_p >= lo) & (self.holdout_p <= hi)
        if m.sum() < min_n:
            return None
        return {"low": lo, "high": hi, "rate": float(self.holdout_y[m].mean()),
                "n": int(m.sum())}

    def _entry_price_stats(self) -> dict:
        """Training success rates and tier counts split at a $25 entry price."""
        tiers = self.train["reward_levels"].map(string_to_tiers)
        mn = tiers.map(min)
        hi = mn > 25
        return {"share_over_25": float(hi.mean()),
                "rate_over_25": float(self.train.loc[hi, "success"].mean()),
                "rate_at_or_under_25": float(self.train.loc[~hi, "success"].mean()),
                "median_tiers_over_25": float(self.train.loc[hi, "levels"].median()),
                "median_tiers_other": float(self.train.loc[~hi, "levels"].median())}

    def standing(self, c: dict) -> dict:
        """Place a campaign among comparable training campaigns.

        Uses the subcategory, or the category if the subcategory has fewer
        than 100 training campaigns.

        Args:
            c: A validated campaign dict.

        Returns:
            The comparison group, its size, the goal's percentile, the median
            goal, the success rate of campaigns with half to double the goal,
            what successful campaigns there typically did, and the goal floor
            used by the goal lever.
        """
        sub = self.train[self.train["subcategory"] == c["subcategory"]]
        scope = c["subcategory"]
        if len(sub) < 100:
            sub = self.train[self.train["category"] == c["category"]]
            scope = c["category"]
        similar = sub[(sub["goal"] >= c["goal"] * 0.5) & (sub["goal"] <= c["goal"] * 2)]
        won = sub[sub["success"] == 1]
        return {
            "scope": scope, "n": int(len(sub)),
            "goal_percentile": float((sub["goal"] < c["goal"]).mean()),
            "median_goal": float(sub["goal"].median()),
            "similar_n": int(len(similar)),
            "similar_success_rate": float(similar["success"].mean()) if len(similar) else float("nan"),
            "winners_median_duration": float(won["duration_days"].median()),
            "winners_median_tiers": float(won["levels"].median()),
            "winners_median_updates": float(won["updates"].median()),
            "goal_floor": float(sub["goal"].quantile(GOAL_FLOOR_QUANTILE)),
        }

    # -- input handling -----------------------------------------------------

    def validate(self, campaign: dict) -> dict:
        """Check a campaign description and normalise it.

        Args:
            campaign: Required keys: goal, duration (1-60), reward_tiers (list
                of positive amounts), category, subcategory, state (two-letter
                code) and updates (planned number, 0 or more). Optional: city,
                name, launch_date.

        Returns:
            A normalised copy: numbers as floats, tiers sorted, state upper-case.

        Raises:
            ValueError: If an outcome field is present, a required field is
                missing, or any value is out of range or unknown.
        """
        c = dict(campaign)
        leaked = sorted(OUTCOME_FIELDS & set(c))
        if leaked:
            raise ValueError(
                f"{leaked} cannot be used: they are outcomes, only known after a "
                f"campaign ends. Describe the campaign as planned instead.")
        missing = [k for k in ("goal", "duration", "reward_tiers", "category",
                               "subcategory", "state", "updates") if c.get(k) in (None, [])]
        if missing:
            raise ValueError(f"Missing required fields: {missing}")
        c["goal"] = float(c["goal"])
        c["duration"] = float(c["duration"])
        c["updates"] = float(c["updates"])
        if not c["goal"] > 0:
            raise ValueError("goal must be positive")
        if c["updates"] < 0:
            raise ValueError("updates (the number you plan to post) cannot be negative")
        if not MIN_DURATION <= c["duration"] <= MAX_DURATION:
            raise ValueError(f"duration must be between {MIN_DURATION} and "
                             f"{MAX_DURATION} days (Kickstarter's current maximum)")
        tiers = [float(x) for x in c["reward_tiers"]]
        if not tiers or min(tiers) <= 0:
            raise ValueError("reward_tiers must be a non-empty list of positive amounts")
        c["reward_tiers"] = sorted(tiers)
        c["state"] = str(c["state"]).upper()
        if c["state"] not in self.states:
            raise ValueError(f"Unknown state {c['state']!r}. Use a two-letter code.")
        if c["category"] not in self.taxonomy:
            raise ValueError(f"Unknown category {c['category']!r}. "
                             f"Choose from: {sorted(self.taxonomy)}")
        if c["subcategory"] not in self.taxonomy[c["category"]]:
            raise ValueError(f"{c['subcategory']!r} is not a subcategory of "
                             f"{c['category']!r}. Choose from: "
                             f"{self.taxonomy[c['category']]}")
        return c

    def to_raw(self, cases: list[dict]) -> pd.DataFrame:
        """Turn validated campaign dicts into raw rows for the feature pipeline.

        Args:
            cases: Campaign dicts already passed through ``validate``.

        Returns:
            One row per campaign with the columns in ``RAW_COLUMNS``. A
            missing launch date becomes the latest training launch date.
        """
        rows = []
        for c in cases:
            rows.append({
                "goal": c["goal"], "duration": c["duration"],
                "levels": len(c["reward_tiers"]),
                "reward_levels": tiers_to_string(c["reward_tiers"]),
                "category": c["category"], "subcategory": c["subcategory"],
                "state_raw": c["state"], "city": c.get("city"),
                "launch_date": c.get("launch_date") or self.default_launch_date,
                "name": c.get("name"), "updates": c["updates"],
            })
        return pd.DataFrame(rows, columns=RAW_COLUMNS)

    def probability(self, campaign: dict) -> float:
        """Chance of success for one campaign description, from the scorer.

        Args:
            campaign: A campaign dict (see ``validate``).

        Returns:
            The final model's probability of success.
        """
        return float(self.predict_proba(self.to_raw([self.validate(campaign)]))[0])

    # -- the advice engine --------------------------------------------------

    def _candidates(self, c: dict, st: dict) -> list[dict]:
        """List every what-if change to evaluate, mildest first within each lever.

        Args:
            c: A validated campaign dict.
            st: Its ``standing`` (for the goal floor).

        Returns:
            Dicts with ``lever``, ``label`` and the changed ``campaign``.
        """
        out, notes = [], []

        # GOAL -- lower is better, monotonically, for every campaign tested.
        floor = max(st["goal_floor"], 1.0)
        seen = set()
        for f in GOAL_FACTORS:
            g = float(f"{c['goal'] * f:.2g}")            # two significant figures
            if g < c["goal"] and g >= floor and g not in seen:
                seen.add(g)
                out.append({"lever": "goal", "label": f"Lower the goal to ${g:,.0f}",
                            "campaign": {**c, "goal": g}})

        # DURATION -- only campaigns running past the default, never *to* 30.
        d = c["duration"]
        if d > DEFAULT_DURATION[1]:
            for target in sorted(DURATION_TARGETS, reverse=True):
                if target < d - 0.5:
                    out.append({"lever": "duration",
                                "label": f"Shorten the campaign to {target} days",
                                "campaign": {**c, "duration": float(target)}})
        # REWARD TIERS -- more distinct price points.
        n_now = len(c["reward_tiers"])
        for target in TIER_TARGETS:
            if target >= n_now + MIN_TIER_ADDITION:
                new = suggest_tiers(c["reward_tiers"], target)
                if len(new) > n_now:
                    out.append({"lever": "reward tiers",
                                "label": f"Offer {len(new)} reward tiers ({tiers_to_string(new)})",
                                "campaign": {**c, "reward_tiers": new}})
        return out

    def assess(self, campaign: dict, min_gain: float = MIN_GAIN) -> "Assessment":
        """Score a campaign, evaluate every candidate change and build a plan.

        Args:
            campaign: A campaign dict (see ``validate``).
            min_gain: Smallest gain, as a probability, worth recommending.

        Returns:
            An ``Assessment`` holding the chance, band, standing, options,
            combined plan and notes.
        """
        c = self.validate(campaign)
        st = self.standing(c)
        cands = self._candidates(c, st)

        # The scorer gives the starting chance; the advisor gives the effect
        # of every variant (one batch), added in percentage points.
        raw_c = self.to_raw([c])
        base = float(self.predict_proba(raw_c)[0])
        cases = [c] + [x["campaign"] for x in cands]
        X_cases = self.featurize(self.to_raw(cases))
        X_cases[HELD_FIXED_IN_WHAT_IFS] = X_cases[HELD_FIXED_IN_WHAT_IFS].iloc[[0] * len(X_cases)].to_numpy()
        adv = self.advisor.predict_proba(X_cases)[:, 1]
        probs = np.concatenate([[base], np.clip(base + (adv[1:] - adv[0]),
                                                self.P_FLOOR, self.P_CEIL)])

        rows, chosen = [], {}
        for x, p in zip(cands, probs[1:]):
            gain = float(p) - base
            row = {"lever": x["lever"], "change": x["label"],
                   "probability": float(p), "gain_pp": gain * 100,
                   "goal": x["campaign"]["goal"],
                   "funding_floor": float(p) * x["campaign"]["goal"],
                   "recommended": False}
            if x["lever"] not in chosen and gain >= min_gain:
                row["recommended"] = True
                chosen[x["lever"]] = x
            rows.append(row)
        options = pd.DataFrame(rows, columns=["lever", "change", "probability", "gain_pp",
                                              "goal", "funding_floor", "recommended"])

        plan = None
        if chosen:
            combined = dict(c)
            for lever, x in chosen.items():
                key = {"goal": "goal", "duration": "duration",
                       "reward tiers": "reward_tiers"}[lever]
                combined[key] = x["campaign"][key]
            p_plan = float(self.what_if_proba(raw_c, self.to_raw([combined]),
                                              base=np.array([base]))[0])
            plan = {"changes": [x["label"] for x in chosen.values()],
                    "probability": p_plan, "gain_pp": (p_plan - base) * 100,
                    "goal": combined["goal"],
                    "funding_floor": p_plan * combined["goal"],
                    "campaign": combined}

        return Assessment(
            campaign=c, probability=base, band=self.band_of(base),
            band_info=self.band_rates.get(self.band_of(base)),
            local_info=self.local_rate(base),
            standing=st, options=options, plan=plan,
            notes=self._notes(c, st, options, base, float(adv[0])),
            base_floor=base * c["goal"])

    def _notes(self, c: dict, st: dict, options: pd.DataFrame,
               probability: float, advisor_probability: float | None = None) -> list[str]:
        """Write the caveats shown under a report.

        Args:
            c: A validated campaign dict.
            st: Its ``standing``.
            options: The evaluated options table.
            probability: The scorer's chance.
            advisor_probability: The advisor's chance, if known.

        Returns:
            Plain-text notes, in display order.
        """
        notes = []
        if c["updates"] > 0:
            p_none = self.probability({**c, "updates": 0})
            notes.append(
                f"Updates: this chance assumes you post {c['updates']:g} updates during the "
                f"campaign. With none planned the model would say {p_none:.0%}. The model "
                f"learned from the number of updates campaigns had posted by the end, which "
                f"partly reflects how well they were already doing, so this is the figure "
                f"the estimate leans on most and it is not advice to post more.")
        else:
            notes.append(
                "Updates: you entered 0 planned updates. Campaigns that posted none rarely "
                "succeeded, which weighs heavily on this chance; if you do plan to post "
                "updates, enter the number.")
        d = c["duration"]
        if DEFAULT_DURATION[0] <= d <= DEFAULT_DURATION[1]:
            notes.append(
                "Duration: 30 days is the platform default. Campaigns left at exactly 30 "
                "days succeeded less often than 26-29 day campaigns of the same goal size, "
                "but that may reflect who accepts the default rather than the number of "
                "days, so no duration change is recommended.")
        elif d < DEFAULT_DURATION[0]:
            notes.append("Duration: already shorter than the default; no change suggested.")
        if len(c["reward_tiers"]) > max(TIER_TARGETS) - MIN_TIER_ADDITION:
            notes.append(f"Reward tiers: you already offer {len(c['reward_tiers'])}, "
                         f"at or near the highest level we model, so no further tiers "
                         f"are suggested.")
        if min(c["reward_tiers"]) > 25:
            es = self.entry_stats
            cheaper = {**c, "reward_tiers": sorted(set(c["reward_tiers"] + [10.0]))}
            delta = (float(self.what_if_proba(self.to_raw([c]), self.to_raw([cheaper]))[0])
                     - self.probability(c)) * 100
            notes.append(
                f"Entry price: your cheapest reward is above $25. Campaigns like that "
                f"succeeded {es['rate_over_25']:.0%} of the time versus "
                f"{es['rate_at_or_under_25']:.0%} otherwise, but they also offered far fewer "
                f"tiers (median {es['median_tiers_over_25']:.0f} versus "
                f"{es['median_tiers_other']:.0f}). With goal size and the number of tiers "
                f"held fixed, that gap shrinks to a few points, and adding a $10 tier to yours changes "
                f"the predicted chance by {delta:+.1f} points. The number of tiers is the "
                f"lever that matters here.")
        if (options["lever"].eq("reward tiers") & options["recommended"]).any():
            notes.append(
                "Reward tiers: this is the strongest association among the levers and also the "
                "one most likely to be partly selection - creators who design many tiers "
                "tend to have planned the whole campaign more carefully. Padding the list "
                "with options nobody wants is unlikely to deliver the full predicted gain; "
                "add tiers backers would actually choose.")
        if (options["lever"].eq("goal") & options["recommended"]).any():
            notes.append(
                "Goal: lowering it raises the chance of reaching it, but only helps if you "
                "can still deliver the project on the smaller budget.")
        if options.empty or not options["recommended"].any():
            adv_p = probability if advisor_probability is None else advisor_probability
            if min(probability, adv_p) < 0.05:
                notes.append(
                    f"Your chance is very low and none of the modelled changes lifts it by "
                    f"even {MIN_GAIN * 100:.0f} points. Your goal is at the "
                    f"{ordinal(round(st['goal_percentile'] * 100))} percentile for {st['scope']} "
                    f"(median ${st['median_goal']:,.0f}). The Random Forest cannot tell campaigns "
                    f"this far out from merely large ones, so read its {probability:.0%} as an "
                    f"upper bound (the Logistic Regression puts it at {adv_p:.1%}). These levers "
                    f"cannot rescue a goal that far from what comparable campaigns ask for.")
            elif probability >= BANDS[1][0]:
                notes.append("No single change clears the minimum predicted gain. The chance "
                             "is already in the favourable range.")
            else:
                notes.append("No single change clears the minimum predicted gain, so none is "
                             "recommended.")
        return notes

    def report(self, campaign: dict) -> str:
        """Plain-text assessment of one campaign.

        Args:
            campaign: A campaign dict (see ``validate``).

        Returns:
            The formatted report from ``Assessment.report``.
        """
        return self.assess(campaign).report()

    def summary_frame(self, cases: dict) -> pd.DataFrame:
        """Assess several named campaigns side by side.

        Args:
            cases: ``{name: campaign dict}``.

        Returns:
            One row per campaign: chance, risk band, levers advised, plan
            chance and plan gain in points.
        """
        rows = []
        for name, c in cases.items():
            a = self.assess(c)
            rec_levers = a.options.loc[a.options["recommended"], "lever"].tolist() \
                if not a.options.empty else []
            rows.append({
                "campaign": name,
                "chance": a.probability,
                "band": a.band,
                "levers advised": ", ".join(rec_levers) if rec_levers else "none",
                "plan chance": a.plan["probability"] if a.plan else a.probability,
                "plan gain (pts)": a.plan["gain_pp"] if a.plan else 0.0,
            })
        return pd.DataFrame(rows)

    # -- consistency guarantee ---------------------------------------------

    def self_check(self, n: int = 2000, seed: int = 0) -> dict:
        """Prove the raw-input path reproduces the training matrix.

        Re-featurising the raw columns of training rows must give back the
        training matrix exactly. If it does not, the what-if engine would be
        answering a slightly different question from the one the model was
        trained on -- the failure that is hardest to notice from outputs alone.

        Args:
            n: Number of training rows to check.
            seed: Sampling seed.

        Returns:
            ``{"rows_checked", "max_abs_difference", "matrix_reproduced",
            "tier_round_trip_ok"}``.
        """
        sample = self.train.sample(min(n, len(self.train)), random_state=seed)
        rebuilt = self.featurize(sample)
        original = self.X_train.reindex(sample.index)
        original["launch_year"] = original["launch_year"].clip(*self.year_range)
        diff = (rebuilt.values - original.values)
        max_abs = float(np.nanmax(np.abs(diff)))

        # Tier round trip: format -> parse must be lossless.
        rt_ok = all(
            string_to_tiers(tiers_to_string(t)) == sorted(float(x) for x in t)
            for t in ([5, 25, 100], [1, 1000, 5000], [10, 2500, 10000], [5.5, 50])
        )
        return {"rows_checked": len(sample), "max_abs_difference": max_abs,
                "matrix_reproduced": max_abs < 1e-9, "tier_round_trip_ok": rt_ok}


# ---------------------------------------------------------------------------
# The result object
# ---------------------------------------------------------------------------

@dataclass
class Assessment:
    """Everything the engine worked out for one campaign."""

    campaign: dict
    probability: float
    band: str
    band_info: dict | None
    local_info: dict | None
    standing: dict
    options: pd.DataFrame
    plan: dict | None
    notes: list = field(default_factory=list)
    base_floor: float = 0.0

    def report(self) -> str:
        """Format the assessment as the plain-text report a creator reads.

        Returns:
            The report: chance and band, standing, options, plan, notes and
            how to read it.
        """
        c, s = self.campaign, self.standing
        L = []
        tiers = tiers_to_string(c["reward_tiers"])
        L.append(f"CAMPAIGN ASSESSMENT")
        L.append(f"  {c['category']} / {c['subcategory']} - {c['state']} - goal "
                 f"${c['goal']:,.0f} - {c['duration']:g} days - {len(c['reward_tiers'])} "
                 f"reward tiers ({tiers}) - {c['updates']:g} planned updates")
        L.append("")
        L.append(f"Predicted chance of reaching the goal: {self.probability:.0%}   "
                 f"[{self.band}]   (final model: {FINAL_MODEL_NAME})")
        if self.local_info:
            li = self.local_info
            L.append(f"  Held-out campaigns scored {li['low']:.0%}-{li['high']:.0%} actually "
                     f"succeeded {li['rate']:.0%} of the time (n={li['n']:,}).")
        elif self.band_info:
            L.append(f"  Held-out campaigns in this band succeeded "
                     f"{self.band_info['observed_rate']:.0%} of the time "
                     f"(n={self.band_info['n']:,}).")
        L.append("")
        L.append(f"Where you stand ({s['scope']}, {s['n']:,} campaigns)")
        L.append(f"  Goal: ${c['goal']:,.0f} is at the {ordinal(round(s['goal_percentile'] * 100))} "
                 f"percentile (median ${s['median_goal']:,.0f}).")
        if s["similar_n"]:
            L.append(f"  Campaigns with a similar goal (half to double yours): "
                     f"{s['similar_success_rate']:.0%} succeeded (n={s['similar_n']:,}).")
        L.append(f"  Successful campaigns here typically ran {s['winners_median_duration']:.0f} "
                 f"days with {s['winners_median_tiers']:.0f} reward tiers and posted "
                 f"{s['winners_median_updates']:.0f} updates.")
        L.append("")
        L.append("What you could change (estimated effect on your chance)")
        if self.options.empty:
            L.append("  Nothing in the modelled range of goal, duration or reward tiers.")
        for _, r in self.options.iterrows():
            tag = "  <- recommended" if r["recommended"] else ""
            L.append(f"  [{r['lever']:<12}] {r['change']}: {r['probability']:.0%} "
                     f"({r['gain_pp']:+.1f} pts){tag}")
        if self.plan:
            L.append("")
            L.append(f"Combined plan: {self.probability:.0%} -> {self.plan['probability']:.0%} "
                     f"({self.plan['gain_pp']:+.1f} pts)")
            for ch in self.plan["changes"]:
                L.append(f"  - {ch}")
            if self.plan["goal"] < c["goal"]:
                L.append(f"  Trade-off: if funded you would receive at least "
                         f"${self.plan['goal']:,.0f} instead of ${c['goal']:,.0f}, so this only "
                         f"works if the project can be delivered on that budget.")
            L.append(f"  Goal x probability: ${self.base_floor:,.0f} -> "
                     f"${self.plan['funding_floor']:,.0f}  (a floor on expected funds, since "
                     f"Kickstarter is all-or-nothing)")
        if self.notes:
            L.append("")
            L.append("Notes")
            for n in self.notes:
                L.append(f"  * {n}")
        L.append("")
        L.append("How to read this: the model learned from US Kickstarter campaigns from "
                 "2009-2012. These are associations, not guarantees - campaigns that did these "
                 "things succeeded more often, partly because the creators behind them were "
                 "also better prepared in ways the data cannot see. Treat predicted gains as an "
                 "upper estimate. City, title and launch date are used if supplied; otherwise typical "
                 "values are assumed, which can shift the chance by a few points (city most). "
                 "The planned number of updates is used as given and never recommended. "
                 "The chance comes from the Random Forest; the effect of each change is estimated "
                 "by a Logistic Regression, which responds smoothly and consistently to a change.")
        return "\n".join(L)


def get_recommender(root: str = "..") -> Recommender:
    """Build the prototype from the random split.

    Args:
        root: Repository root relative to the caller.

    Returns:
        A ``Recommender`` fitted on the training half, with risk-band rates
        measured on the test half.
    """
    from src.data import load_splits
    train, test = load_splits(root)["random"]
    return Recommender.fit(train, test=test)


def campaign_from_row(row) -> dict:
    """Turn a row of the featured table into a campaign description.

    Used to replay real held-out campaigns through the engine. Outcome fields
    are not copied; the campaign's actual final update count stands in for a
    planned one.

    Args:
        row: One row of a split from ``src.data.load_splits``.

    Returns:
        A campaign dict accepted by ``Recommender.validate``.
    """
    return dict(goal=float(row["goal"]), duration=float(row["duration_days"]),
                reward_tiers=string_to_tiers(row["reward_levels"]),
                category=row["category"], subcategory=row["subcategory"],
                state=row["state_raw"], city=row["city"], name=row["name"],
                launch_date=row["launch_date"], updates=float(row["updates"]))
