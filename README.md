# NHL Expected Goals (xG) — End-to-End Analytics + Streamlit

Production-style portfolio app: ingest public NHL play-by-play, model shot danger with **interpretable** logistic regression, evaluate on a **time-based** holdout, and explore results in a lightweight **Streamlit** app.

## Why xG matters

Not all shots are equally dangerous. Expected goals summarizes **shot context** (location, angle, rebounds, game state) into a **goal probability**, helping separate **process** (chance creation) from **results** (finishing variance over small samples).

## Data source

- NHL Web API (`api-web.nhle.com`) schedule + per-game play-by-play (`/v1/gamecenter/{game_id}/play-by-play`).
- Shots are **Fenwick-style unblocked attempts**: `shot-on-goal`, `missed-shot`, and `goal` events.

## Train / test setup (mandatory time split)

This app intentionally avoids random train/test splits on the same season.

Configured in [`src/config.py`](src/config.py):

- **Train**
  - Full regular seasons: `20232024`, `20242025`
  - Plus early **2025–26** regular season through **2025-12-31** (`season_id = 20252026`)
- **Test / analysis**
  - **2025–26** regular season games with `game_date >= 2026-01-01` (`season_id = 20252026`)

This mimics deployment: the model never trains on the evaluation window.

> Note: If you only ingest older seasons for a quick smoke test, the test split can be empty—expand ingestion dates/seasons until both windows contain games.

Because schedule discovery walks calendar time, using `--max-games` can stop ingestion **before** the January+ portion of the season. For 2025–26, run additional ingest passes over **late-season date ranges** (or raise/remove `--max-games`) so the held-out test window in `src/config.py` actually contains games.

## Repository layout

```
data/raw/               # optional per-game JSON (ingest default)
data/processed/         # SQLite DB by default (or Postgres via DATABASE_URL)
sql/schema.sql          # core tables: games, events, shots, players
sql/shot_context_features.sql   # CTEs + window functions → shots_with_context
src/config.py           # season/date boundaries + paths
src/utils.py            # HTTP + DB helpers
src/ingest.py           # schedule crawl + PBP → SQL
src/clean.py            # apply schema + rebuild SQL features + optional player names
src/features.py         # rink geometry + modeling frame → parquet
src/train.py            # scikit-learn logistic regression + artifacts
src/evaluate.py         # charts + extra test metrics JSON
app/streamlit_app.py    # interactive explorer
notebooks/              # scratch space
```

## SQL usage (what makes this “real analytics SQL”)

[`sql/shot_context_features.sql`](sql/shot_context_features.sql) builds `shots_with_context` using:

- `LAG` / `LEAD` ordered by `game_id`, `sort_order`
- rebound heuristic: prior event is a same-team shot within **3 seconds**
- rush heuristic: prior event in a small transition class within **4 seconds**
- joins to `games` for team ids + defending side metadata

## Feature engineering (interpretable)

Implemented in [`src/features.py`](src/features.py):

- **Rink normalization** using `homeTeamDefendingSide` (mirror when home defends `"right"` so geometry is consistent).
- **Distance / angle** to the attacked net (`GOAL_LINE_X_FT = 89`).
- Flags: rebound, rush heuristic, PP/SH/EV, empty net (from `situationCode` goalie digits), home/away, period, score differential for shooting team.

## Modeling + evaluation

- **Model**: scikit-learn `Pipeline` with `StandardScaler` on numeric features + `OneHotEncoder` for `shot_type`, then **logistic regression**.
- **Metrics**: ROC-AUC, log loss, Brier score (train + test written to `outputs/metrics.json`).
- **Artifacts**:
  - `models/xg_logistic.joblib`
  - `outputs/coefficients.csv`
  - `outputs/test_shot_predictions.parquet`
  - `outputs/fig_*.png` (calibration, distributions, maps)

## Run locally

### 1) Install

On macOS, the system interpreter is usually **`python3`** (there is often no `python` on `PATH` until a venv is active).

```bash
cd /path/to/Hockey
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

After `source .venv/bin/activate`, the venv provides `python` and `pip` on `PATH`.

### 2) Initialize database + ingest

```bash
export PYTHONPATH=.

# Create tables
python -m src.clean --schema

# Ingest regular-season games (adjust seasons/dates for your machine/time)
python -m src.ingest --seasons 20232024,20242025,20252026 --start 2023-10-01 --end 2026-04-20

# Rebuild SQL window features
python -m src.clean --context

# Optional: populate player names (uses statsapi; takes a while on full data)
python -m src.clean --players
```

### 3) Features → train → evaluate

```bash
python -m src.features --out outputs/modeling_dataset.parquet
python -m src.train --data outputs/modeling_dataset.parquet
python -m src.evaluate --preds outputs/test_shot_predictions.parquet
```

### 4) Streamlit

```bash
PYTHONPATH=. streamlit run app/streamlit_app.py
```

**If you see `command not found: python`:** use `python3` to create the venv (above), then activate it before running `python` / `streamlit`. If `python3` is missing, install Python 3 from [python.org](https://www.python.org/downloads/) or `brew install python`.

## PostgreSQL (optional)

By default the app uses SQLite at `data/processed/nhl_xg.db`.

For Postgres, set:

```bash
export DATABASE_URL=postgresql+psycopg2://USER:PASS@HOST:5432/DBNAME
```

Then apply the DDL in `sql/schema.sql` with minor edits if needed (remove `PRAGMA`).

## Screenshots (placeholders)

Add your own after running the app:

- `docs/screenshots/overview.png`
- `docs/screenshots/player_explorer.png`

## Main findings (fill in after you run the full pipeline)

Template prompts:

- Which teams **out-shot their xG** on the test window?
- Which players have **high xG** but **low goals** (possible cold finishing / small sample)?

Always apply **minimum shot thresholds** in leaderboards.

## Limitations + future work

- Public coordinates depend on NHL’s feed; mirroring assumptions are documented in-code.
- Rush detection is **heuristic**, not optical tracking.
- No shooter handedness / pre-pass context / goalie quality in the baseline model.
- Consider **isotonic calibration** on a validation month within train before deploying.

## License / attribution

This app is for educational/portfolio use. NHL data © NHL. Use responsibly and respect NHL API rate limits.
