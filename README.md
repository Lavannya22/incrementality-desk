# Incrementality Desk — Criteo Uplift × DuckDB × Streamlit

An interactive decision tool for a performance-marketing agency whose client's CFO wants proof that ad spend is *incremental*
(caused by the ads) rather than credited to sales that would have happened anyway. Built on the Criteo Uplift v2.1 randomized
experiment (13.98M users).

- **Client:** a performance-marketing agency's account team, presenting to a skeptical CFO.
- **Mandate (must-do):** prove incremental lift from the randomized experiment (confidence interval, p-value, effect on the exposed,
  randomization check, the "attribution illusion") and show who to target with an uplift model.
- **Opportunity (can-do):** a *Do-Not-Target* engine: a plain-English cohort rulebook with measured profit per cohort and CSV export,
  plus a business case and scale calculator.

## Project layout

```
app/                  Streamlit dashboard
  streamlit_app.py    all six tabs
  data.py             DuckDB access layer (full database locally, slim database when deployed)
  econ.py             targeting economics and out-of-sample validation
  stats.py            confidence intervals, effect on the exposed, sample-size maths
  ui.py               styling, KPI cards, chart defaults
pipeline/             data build (not needed to run or deploy the app)
  00_download.py      download the dataset (311 MB) from Hugging Face
  01_ingest.py        CSV -> DuckDB (13.98M rows) + 70/30 train/test split
  02_train_score.py   uplift models (T-learner) + scoring of the 30% hold-out + rule tree
  03_aggregates.py    randomization-balance and sample-size tables
  04_build_app_db.py  slim deployable database from the full one
data/app.duckdb       slim database (about 8 MB), committed so the app can deploy
```

## Run it locally

Quick start, using the slim database that is already in the repo:

```bash
pip install -r requirements.txt
INCREMENTALITY_DB=slim streamlit run app/streamlit_app.py     # Windows PowerShell: $env:INCREMENTALITY_DB="slim"; streamlit run app/streamlit_app.py
```

Full rebuild from the raw data (about 15 minutes):

```bash
pip install -r requirements.txt -r pipeline/requirements.txt
python pipeline/00_download.py     # 311 MB from Hugging Face (skip if data/raw already has the file)
python pipeline/01_ingest.py       # CSV.gz -> DuckDB, ~75 s
python pipeline/02_train_score.py  # uplift models + scoring of the 30% hold-out, ~8 min
python pipeline/03_aggregates.py   # balance + convergence tables, seconds
python pipeline/04_build_app_db.py # slim deployable database (data/app.duckdb), ~1 min
streamlit run app/streamlit_app.py
```

Two databases, one app:

| Database | Size | In git | Contents | Used when |
|---|---|---|---|---|
| `data/criteo.duckdb` | ~750 MB | no | `uplift` (13.98M rows), `split`, `scored` (4.19M hold-out users with model scores), `balance`, `convergence`, `meta` | present locally, so the SQL console runs over all rows |
| `data/app.duckdb` | ~8 MB | yes | pre-aggregates DuckDB computed over those 14M rows, plus a 100,000-user hold-out sample | `criteo.duckdb` is absent (deployed), or `INCREMENTALITY_DB=slim` |

Both give identical numbers on every dashboard chart (checked table by table).

## Deploy to Streamlit Community Cloud

The full database is too big for GitHub (100 MB file limit), so the repo ships only the slim one.

1. Create an empty repository on GitHub, then push this folder:
   ```bash
   git remote add origin <your-repo-url>
   git push -u origin main
   ```
2. On [share.streamlit.io](https://share.streamlit.io) choose **Create app**, pick the repository and branch `main`, and set
   **Main file path** to `app/streamlit_app.py`.
3. Deploy. `requirements.txt` holds only the five packages the dashboard needs. The first load can take 30 to 60 seconds.

## App tabs

| Tab | What it answers |
|---|---|
| Executive Brief | One-minute answer: lift, certainty, % truly incremental, validated profit gain, recommendations |
| 1 · Incrementality Ledger | A/B result with CI, ITT vs effect-on-exposed (CACE), naive-vs-true lift, CI vs sample size, covariate balance, sample-size calculator |
| 2 · Who To Target | Qini curve, measured-vs-predicted lift by decile, policy simulator (profit / iROAS / iCPA vs % targeted), out-of-sample check |
| 3 · Do-Not-Target Engine | Four personas with realized lift, cohort rulebook (Target / Test / Suppress), savings, feature profile, CSV download |
| 4 · Business Case | Strategies A/B/C (validated), assumption-sensitivity heatmap, "small lift = millions" scale calculator |
| Methods & SQL | Data card, pipeline, statistics, limitations, live read-only DuckDB console |

Sidebar assumptions (outcome, value per outcome, cost per exposed user, audience size, confidence level) drive every page.

## Key findings (default assumptions: $40 per conversion, $1 per exposed user, 10M-user audience)

- **The lift is real and small:** conversion 0.194% → 0.309% (+0.115 pp, +59% relative, p < 0.0001).
- **Exposure reports overclaim:** only about 59% of the conversions an exposure-based report would credit are incremental. The naive
  exposed-vs-unexposed comparison looks like 45×; the true effect on the exposed is about 2.5×.
- **The model's ranking holds up on unseen users:** top decile +0.81 pp vs near zero for most others; the top 20% of users capture
  about 79% of incremental conversions.
- **Targeting strategies** (cut-off and cohorts chosen on one random half of the hold-out, scored on the other):

  | Strategy | Profit | iROAS | Media spend |
  |---|---|---|---|
  | A · target everyone | $104.4K | 1.29× | $361.1K |
  | B · uplift model, top 10% | $123.5K | 1.62× | $199.6K |
  | C · cost-aware rulebook | $204.7K | 2.77× | $115.6K |

- **No confirmed "sleeping dogs":** the model's negative-uplift group is *not* confirmed by the hold-out (measured lift there is slightly
  positive). The app reports this instead of claiming harm.

## Limitations

- Prices are not in the data: value per outcome and cost per exposure are editable assumptions, and the gains depend on them (see the
  sensitivity heatmap).
- Features f0–f11 are anonymized, so rules cannot be mapped to real customer attributes without the client.
- Cut-offs and suppressed cohorts are validated out-of-sample, but the real proof is a live A/B test of the policy itself.
- One experiment window with no time dimension, so seasonality and ad fatigue are unobserved.
- Per-cohort lift on rare conversions is noisy; intervals are shown wherever a decision depends on a sign.
- The optional Criteo Attribution dataset (campaign-level budget reallocation) is not used.

## Data and licence

Criteo Uplift Prediction Dataset v2.1, from Criteo AI Lab (Hugging Face `criteo/criteo-uplift`), licensed CC BY-NC-SA 4.0:
non-commercial use with attribution, and derivatives must be shared under the same licence. `data/app.duckdb` contains aggregates and a
sample derived from it, so the same terms apply.
