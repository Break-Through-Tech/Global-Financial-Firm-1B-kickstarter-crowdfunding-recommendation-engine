# Kickstarter Crowdfunding Recommendation Engine

Break Through Tech AI Studio — Fall 2026 ·
Challenge Advisor: Neha Panchal · AI Studio Coach: Darshan Ugale

> **Status: complete.** All four milestones and the December wrap-up are done.
> Final model: **Random Forest (tuned + isotonic-calibrated) with `updates`,
> 81.6% accuracy**, up from 71.4% for the same model without `updates`
> (`reports/m3_model_card.json`).

![Final model summary](reports/figures/final_00_best_model_summary.png)

---

## Overview

Kickstarter is all-or-nothing: a campaign that raises 99% of its goal returns
every dollar and the creator gets nothing. This project predicts whether a
campaign will reach its goal and turns what the data associates with success
into advice a creator can act on before publishing.

It has two parts:

- **A model.** Three classifiers (Logistic Regression, Random Forest, KNN)
  trained on 38,378 US campaigns from 2009–2012. The final model is a tuned,
  calibrated Random Forest.
- **A recommendation prototype** (`src/recommend.py`). A creator describes a
  campaign, including how many updates they plan to post; it returns a chance
  of success, a risk band, and the changes to goal, duration and reward tiers
  that the data associates with better odds.

### The `updates` decision

`updates` is the number of posts a creator made *during* a campaign, counted
when it ended. It is by far the strongest predictor in the data. The team
first excluded it, because it is not known on launch day, and reached about
72%. The final model **includes it**, because the brief targets about 80%
accuracy and lists `updates` among the predictive variables. The cost is
stated wherever results appear:

- The headline accuracy uses each campaign's **actual final** update count.
  Before launch only a **plan** exists, so the prototype asks for one and
  never recommends changing it.
- Part of the signal is **reverse causation**: campaigns that are already
  succeeding post more updates.

Every model was fitted with and without `updates`, so the gain is measured,
not asserted (see [Results](#results)).

---

## Milestones and Tasks

Each month's tasks map to one notebook, and each task is one section of it
(section numbers match task numbers, so §11.6 is Task 11, section 6).

### September — Milestone 1 (Data Preparation) & Milestone 2 (Exploratory Data Analysis)

| # | Task | Target | Status | Evidence |
|---|------|--------|--------|----------|
| 1 | Download and inspect the dataset | 9/9 | Done | `notebooks/01_data_prep_and_eda.ipynb`, Task 1 |
| 2 | Filter and clean the dataset | 9/13 | Done | `notebooks/01_data_prep_and_eda.ipynb`, Task 2; `reports/m1_cleaning_step_log.csv` |
| 3 | Prepare features for analysis | 9/17 | Done | `notebooks/01_data_prep_and_eda.ipynb`, Task 3 |
| 4 | Perform exploratory data analysis | 9/24 | Done | `notebooks/01_data_prep_and_eda.ipynb`, Tasks 4–5 |
| 5 | Create visualizations and summarize findings | 9/30 | Done | `notebooks/01_data_prep_and_eda.ipynb`, Tasks 4–5; `reports/m2_*.csv` |

### October — Milestone 3 (Machine Learning / Modeling)

| # | Task | Target | Status | Evidence |
|---|------|--------|--------|----------|
| 6 | Prepare data and lock the feature set | 10/7 | Done | `notebooks/02_modeling.ipynb`, Task 6; `reports/m3_leakage_register.csv`, `reports/m3_split_summary.csv` |
| 7 | Build baseline and classification models | 10/14 | Done | `notebooks/02_modeling.ipynb`, Task 7 (7a Logistic Regression — Yero, 7b Random Forest — James, 7c KNN — Menwe); `reports/m3_task7_model_results.csv` |
| 8 | Compare model performance | 10/20 | Done | `notebooks/02_modeling.ipynb`, Task 8; `reports/m3_task8_comparison.csv`, `reports/m3_updates_effect.csv` |
| 9 | Perform feature selection and tune models | 10/27 | Done | `notebooks/02_modeling.ipynb`, Task 9; `reports/m3_tuned_config.json`, `reports/m3_task9_tuned_results.csv` |
| 10 | Analyze interpretability and select the final model | 10/31 | Done | `notebooks/02_modeling.ipynb`, Task 10; `reports/m3_model_card.json`, `reports/m3_final_feature_importance.csv` |

### November — Milestone 4 (Recommendation Prototype)

| # | Task | Target | Status | Evidence |
|---|------|--------|--------|----------|
| 11 | Define the recommendation approach and identify risk factors | 11/6 | Done | `notebooks/03_recommendation_prototype.ipynb`, Task 11; `reports/m4_risk_bands.csv`, `reports/m4_at_risk_profile.csv` |
| 12 | Connect predictions to recommendations (rules and logic) | 11/13 | Done | `notebooks/03_recommendation_prototype.ipynb`, Task 12; `src/recommend.py`, `reports/m4_lever_effects.csv` |
| 13 | Build the initial recommendation prototype | 11/18 | Done | `notebooks/03_recommendation_prototype.ipynb`, Task 13; `src/recommend.py` |
| 14 | Test the prototype on sample campaigns and improve it | 11/24 | Done | `notebooks/03_recommendation_prototype.ipynb`, Task 14; `reports/m4_sample_assessments.csv`, `reports/m4_edge_cases.csv` |
| 15 | Final validation of the model and the recommendations | 11/30 | Done | `notebooks/03_recommendation_prototype.ipynb`, Task 15; `reports/m4_validation_summary.json` |

### December — Wrap-up

| # | Task | Target | Status | Evidence |
|---|------|--------|--------|----------|
| 16 | Summarize results and key insights | 12/1 | Done | This README: [Results](#results), [Limitations](#limitations) |
| 17 | Finalize documentation and repository | 12/1 | Done | `README.md`, `requirements.txt`, docstrings in `src/` |
| 18 | Create final charts and visuals | 12/1 | Done | `notebooks/04_final_charts.ipynb`; `reports/figures/final_00` … `final_08` |

---

## Data

**Source:** [Kickstarter dataset on Kaggle](https://www.kaggle.com/parienza/kickstarter),
committed as `data/DSI_kickstarterscrape_dataset_csv.zip` (no Kaggle login needed).

| | Value | Source |
|---|---|---|
| Raw scrape | 45,957 campaigns × 17 columns | `reports/m1_cleaning_step_log.csv` |
| Cleaned table | 38,378 US campaigns, successful or failed | `reports/m1_cleaning_step_log.csv` |
| Coverage | 13 categories, 49 subcategories, 51 states, 3,153 cities; launched 2009-04-21 to 2012-05-29 | notebook 01, Task 2 (§2.6 output) |
| Success rate | 54.7% (the "always guess success" floor) | `reports/m3_split_summary.csv`, `reports/m3_model_card.json` |
| Splits | Random stratified 80/20: 30,702 train / 7,676 test. Chronological: same sizes, cut at 2012-02-11 | `reports/m3_split_summary.csv` |

![Data funnel](reports/figures/final_01_data_funnel.png)

### Data quality issues found and fixed

| Issue | Resolution |
|---|---|
| File is neither UTF-8 nor cp1252 | Load with `latin-1` |
| `Film &amp; Video` and `Film & Video` are separate categories | `html.unescape()` before grouping: 14 to 13 categories, 51 to 49 subcategories |
| Montana appears as both `MT` and `Mt` | Case-insensitive state whitelist |
| Duplicated `project_id` values | Deduplicate on the platform's own key (113 rows) |
| Largest goal is $21,474,836.47 = 2³¹−1 cents | Integer overflow; row dropped |
| `funded_date` is the campaign **end** date | Recovered `launch_date = funded_date − duration` |
| `reward_levels` has commas inside amounts (`$1,000`) | Comma-aware parser, checked against the scraper's own `levels` column |

**Recipe check.** Running the original project's recipe unchanged gives 38,492
rows against their published 38,491 (notebook 01, §2.2). Our three deliberate
deviations account for the 114-row difference (§2.3), mostly deduplication.

### What separates successful from failed campaigns

![What drives success](reports/figures/final_02_what_drives_success.png)

All on the training split (notebook 01, Tasks 4–5, and notebook 03, §11.5):

- **Updates** — the strongest signal in the file. Successful campaigns posted
  a median of 5 updates; failed ones a median of 0
  (`reports/m2_summary_by_outcome.csv`).
- **Goal** — success falls from 71.2% in the smallest tenth of goals to 22.7%
  in the largest, monotonically (notebook 01, §4.1).
- **Reward tiers** — within the same goal size, more tiers go with more
  success (notebook 03, §11.5).
- **Duration** — exactly-30-day campaigns (the platform default) succeed less
  than those just shorter or longer, and over 45 days is worst
  (`reports/m4_duration_buckets.csv`).
- **Location** — across the 32 states with 150+ campaigns, success runs from
  37% (FL) to 65% (NY) (`reports/m4_sept_claims_rechecked.csv`).

### What we got wrong in September and fixed

Re-tested in November (notebook 03, §11.6; `reports/m4_sept_claims_rechecked.csv`):

| September claim | What the data shows |
|---|---|
| "The 30-day default outperforms" | False: 30-day campaigns succeed 52.9% of the time against 55.5% for the rest |
| "A low entry price helps" | Mostly a tier-count effect: a −23.1-point raw gap becomes −2.4 points with goal size and tier count held fixed |
| "Creators can just add a $5 tier" | Not supported: adding a $10 tier to a campaign with a >$25 entry changes the prediction by −2.5 points on average |
| "Location is a modest signal" | False: a 27-point spread across states; 43% against 62% between the smallest and largest city-size quartiles |

---

## Setup

Tested with Python 3.12; older versions are not verified.

```bash
git clone https://github.com/Break-Through-Tech/Global-Financial-Firm-1B-kickstarter-crowdfunding-recommendation-engine.git
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lists only packages the code imports directly, plus the
Jupyter tools needed to run the notebooks.

---

## Usage

### Run the notebooks

Open them interactively:

```bash
jupyter lab
```

(`jupyter lab` opens a browser, so it was not run in testing; verify it
manually. Every other command in this README was run as written.)

Or run all four from the command line, in order (notebook 03 reads the model
card that notebook 02 writes, and notebook 04 reads both):

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=4000 01_data_prep_and_eda.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=4000 02_modeling.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=4000 03_recommendation_prototype.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=4000 04_final_charts.ipynb
```

Every table is rebuilt from the raw zip in memory; nothing derived is written to
`data/`. Measured run times on a single CPU core: notebook 01 about 1 minute,
02 about 25 minutes (the randomized search in Task 9 is most of it), 03 about
15 minutes, 04 about 10 minutes. A multi-core laptop is faster.

### Try the prototype

From the repository root:

```bash
python - <<'EOF'
from src.recommend import get_recommender

rec = get_recommender(".")
print(rec.report(dict(
    goal=40000, duration=60, reward_tiers=[10, 25, 50],
    category="Film & Video", subcategory="Documentary", state="CA",
    updates=5,          # the number of updates you PLAN to post
)))
EOF
```

`updates` is required. Outcome fields (`backers`, `pledged`, `comments`,
`funded_percentage`) are refused with an explanation. `city`, `name` and
`launch_date` are optional. Notebook 03, Task 13 walks through an editable
example.

### Repository layout

```
src/data.py          Load, clean (with a step log) and split the raw data
src/features.py      Feature engineering, the leakage register, feature sets
src/modeling.py      The three models, make_final_model(), the evaluation harness
src/recommend.py     The recommendation engine (Random Forest scorer + Logistic Regression advisor)

notebooks/01_data_prep_and_eda.ipynb         September: Tasks 1–5
notebooks/02_modeling.ipynb                  October:   Tasks 6–10
notebooks/03_recommendation_prototype.ipynb  November:  Tasks 11–15
notebooks/04_final_charts.ipynb              December:  Task 18

data/                The raw dataset zip
reports/             Result tables written by the notebooks (prefix = milestone)
reports/figures/     Charts, including final_00 … final_08
```

---

## Results

All numbers below come from the current notebooks; the file next to each table
is where to check it.

### What adding `updates` does to each model

Same model, same rows, same settings; the only change is the one column.
Random split, test set of 7,676 campaigns (`reports/m3_updates_effect.csv`;
the final model's row is from `reports/m3_model_card.json`).

| Model | Accuracy without → with `updates` | ROC-AUC without → with | Accuracy on later campaigns, without → with |
|---|---|---|---|
| **Random Forest, tuned + calibrated (final)** | **71.4% → 81.6%** | 0.775 → 0.898 | 71.8% → 80.5% |
| Random Forest, default settings | 71.5% → 81.8% | 0.780 → 0.897 | 72.0% → 80.6% |
| Logistic Regression | 71.6% → 81.6% | 0.777 → 0.894 | 71.8% → 79.9% |
| KNN | 65.3% → 68.6% | 0.703 → 0.741 | 65.7% → 67.9% |
| Always guess "successful" | 54.7% | 0.500 | 53.5% |

![Model comparison](reports/figures/final_03_model_comparison.png)

### The final model

Random Forest (400 trees, maximum depth 12, minimum leaf 3, 30% of features per
split) with isotonic calibration, trained on the `with_updates` feature set.
Built by `src/modeling.make_final_model()`. Source: `reports/m3_model_card.json`.

| | Value |
|---|---|
| Accuracy | **81.6%** (floor 54.7%) |
| Precision / Recall / F1 | 0.821 / 0.847 / 0.834 |
| ROC-AUC | 0.898 |
| Later campaigns (chronological split) | 80.5% accuracy, ROC-AUC 0.887 |
| Calibration | Every decile within 4.1 points on the random split |

**Why the Random Forest** (notebook 02, §10.5): against the tuned Logistic
Regression it is ahead on six of eight criteria — recall, F1, ROC-AUC,
accuracy and ROC-AUC on later campaigns, and calibration. **The margins are
small**: Logistic Regression is ahead on accuracy by 0.03 points and on
precision, and every gap is under one point of accuracy
(`reports/m3_task10_finalists.csv`).

**An honest alternative.** A Logistic Regression with `log(1 + updates)`
instead of the raw count scored 82.3% accuracy, about 0.7 points ahead of the
final model; the same transform does nothing for the Random Forest
(`reports/m3_task9_tuned_results.csv`, notebook 02, §9.2). The team kept the
Random Forest for the reasons above.

**Tuning helped little.** The randomized search raised ROC-AUC from 0.897 to
0.898 but lowered accuracy from 81.8% to 81.6%, because it optimises ROC-AUC.
Class balancing did not raise accuracy either (notebook 02, Task 9).

**Calibration method.** With `updates` in the model, the original sigmoid
calibration made the worst decile *worse* (8.7 points) than the raw forest
(6.4); isotonic calibration brings it to 4.1 with the same accuracy
(notebook 02, §10.2).

### What drives the prediction

![Top features](reports/figures/final_08_top_features.png)

Permutation importance on held-out campaigns: how much ROC-AUC drops when one
input is shuffled (`reports/m3_final_feature_importance.csv`).

| Rank | Input | Drop in ROC-AUC |
|---|---|---|
| 1 | **Number of updates** | 0.255 |
| 2 | Goal per reward tier | 0.023 |
| 3 | Funding goal | 0.010 |
| 4 | Category | 0.007 |
| 5 | Cheapest reward as % of goal | 0.006 |
| 6 | Creator's city size | 0.005 |

`updates` matters about ten times more than anything else. Logistic
Regression agrees: its largest coefficient is `updates`, about three times
the next (notebook 02, §10.3).

### Recommendation prototype

![Worked example](reports/figures/final_06_example_recommendation.png)

- **The Random Forest scores** the campaign; **a Logistic Regression
  advises**, estimating how many points each change is worth. A forest's
  response to a small change is not consistent in direction, so it cannot do
  the second job alone (notebook 03, §11.8).
- **Levers:** goal, duration (only for campaigns running past 30 days) and
  number of reward tiers. Category and location inform the score but are never
  advised. **`updates` is an input, never a lever.**
- **Risk bands** separate cleanly on held-out campaigns: 13.4% of those scored
  High risk succeeded, 45.7% of Moderate risk and 83.8% of Favourable
  (`reports/m4_risk_bands.csv`).

![Risk bands](reports/figures/final_07_risk_bands.png)

**Replayed on 250 held-out at-risk campaigns** (`reports/m4_plan_behaviour.csv`):
97% received advice; the median plan adds 7.3 points to the predicted chance
(mean 8.2) and 18% of plans reach 50% or better. Re-scoring each planned
campaign directly with the Random Forest also raises its chance in 90% of
cases. In **19% of plans goal × probability falls**: the lower goal costs more
than the better odds gain, which is why every report states what a lower goal
gives up.

### Validation — 17 of 18 checks pass

Every check in notebook 03, Task 15 states its pass condition before the result
is shown (`reports/m4_validation_summary.json`).

| Question | Answer |
|---|---|
| Only information a creator can supply? | **Yes** — no outcome column among the 138 features; `updates` is the only campaign-time feature and is required as a plan; all 6 outcome fields are refused, and so is a missing plan |
| Same model as Milestone 3? | **Yes** — accuracy, F1 and ROC-AUC match the model card exactly |
| Holds on later campaigns? | **Yes on accuracy** (81.6% random, 80.5% chronological) |
| Calibrated? | Yes on the random split (worst decile 4.1 points). **FAIL on the chronological split: 5.6 points** against a 5-point limit, with the model *pessimistic* in the middle of the range (scored ~62%, succeeded 68%) |
| Fair across city size? | Calibrated in every group (within 4.0 points), **but** small-city creators are scored High risk about 1.5 times as often as large hubs (41.5% against 27.4%) |
| Levers move the right way? | Yes — 99.9% or more of eligible campaigns for every change, after a fix found here (goal roundness is now held fixed in what-ifs) |
| Evidence stable over time? | Yes — the tier gradient, the 30-day dip and the long-campaign penalty hold in both periods |
| Plans sensible? | No plan lowers a chance; about 2 in 10 lower goal × probability |

---

## Limitations

- **The headline accuracy is not a launch-day accuracy.** 81.6% uses each
  campaign's final update count. Before launch the prototype has only the
  creator's plan, and how well plans match reality is not in this data.
  Without `updates` the same model scores 71.4%.
- **Part of the `updates` effect is reverse causation.** Campaigns that are
  succeeding post more updates. The prototype therefore never advises posting
  more.
- **The two best models are level.** The Random Forest's lead over Logistic
  Regression is under one point on every metric, and a log-updates Logistic
  Regression scored higher on accuracy.
- **Associations, not causes.** No campaign in the data changed one thing and
  was observed again, so no validation check can show that following the
  advice works.
- **2009–2012, US only.** Calibration misses its 5-point limit on the later
  campaigns (5.6 points in the worst decile, too pessimistic in the middle of
  the range), so the model was already drifting within that window.
- **The city feature** is a documented fairness trade-off (notebook 03, §15.4):
  small-city creators are scored High risk about 1.5 times as often as
  those in large hubs (41.5% against 27.4%), although the model is calibrated
  in every city-size group. Dropping the feature costs no accuracy and narrows,
  but does not close, the gap (`reports/m4_fairness_by_city.csv`).
- **Nobody outside the team has used the prototype.**

## Next Steps

- Collect *planned* update counts from creators before launch, to measure how
  far plans and outcomes differ.
- Consider the log-updates Logistic Regression as the final model.
- Add signal the scrape lacks: campaign text, video and image counts, creator
  history and following.
- Test the prototype with real creators.
- Retrain on recent data and re-check calibration.

---

## Team

| Name | GitHub Handle | Contribution |
|------|---------------|--------------|
| James Lu | @jameslu06 | Data exploration, visualization, Random Forest model, overall project coordination, team lead |
| Yero Barry | @Ybarry2 | Python development, Logistic Regression model and tuning, prompt engineering, team communication |
| Menwe Okafor | @menwe2310 | Python development, KNN model, AI model development and data analysis |

## License

Distributed under the MIT License. See `LICENSE` for more information.

## References

- [Kickstarter dataset on Kaggle](https://www.kaggle.com/parienza/kickstarter)
- [scikit-learn documentation](https://scikit-learn.org/stable/)
- `Challenge-Project-Overview.md` — the challenge brief

## Acknowledgements

Thanks to Challenge Advisor Neha Panchal and AI Studio Coach Darshan Ugale for
guidance throughout this project.
