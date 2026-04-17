"""
Interactive exploration for the NHL xG portfolio app.

Run from repository root:
    PYTHONPATH=. streamlit run app/streamlit_app.py

Streamlit Cloud does not set PYTHONPATH; the block below adds the repo root so
``import src`` works when the main file is ``app/streamlit_app.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_root_str = str(ROOT)
if _root_str not in sys.path:
    sys.path.insert(0, _root_str)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import TEST_SEASON_ID, nhl_team_display_name

OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

_APP_TAB_LABELS = [
    "About This App",
    "Overview",
    "Player Explorer",
    "Team Explorer",
    "Model Diagnostics",
]
_MAIN_TABS_STATE_KEY = "main_nav_tabs"

# Plot surfaces aligned with app theme (.streamlit/config.toml).
_CHART_PLOT_BG = "#0a181c"
# Slightly lifted teal-gray so low-xG markers (see _XG_COLORSCALE) separate from the ice.
_RINK_PLOT_BG = "#152f38"
_CHART_PAPER_BG = "#061213"
_CHART_GRID = "#1a3d44"
_RINK_GRID = "#2a4f5c"
_CHART_TEXT = "#dcefea"
_CHART_MARKER = "#00c2d4"
_CHART_ACCENT = "#f5841f"
# Continuous xG on rink maps: mid-tone floor (readable on dark) → teal → warm peak.
_XG_COLORSCALE: list[list[float | str]] = [
    [0.0, "#6f949e"],
    [0.28, "#3d9aa3"],
    [0.52, "#1cb5c8"],
    [0.76, "#e6c85c"],
    [1.0, "#ff9f45"],
]


def _style_plotly_fig(fig: go.Figure, *, for_rink: bool = False) -> None:
    plot_bg = _RINK_PLOT_BG if for_rink else _CHART_PLOT_BG
    grid = _RINK_GRID if for_rink else _CHART_GRID
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=_CHART_PAPER_BG,
        plot_bgcolor=plot_bg,
        font=dict(color=_CHART_TEXT),
    )
    fig.update_xaxes(gridcolor=grid, zerolinecolor=grid)
    fig.update_yaxes(gridcolor=grid, zerolinecolor=grid)


ABOUT_PAGE_MARKDOWN = r"""
# 🏒 About This App

## What This Is

This app builds an interpretable **Expected Goals (xG) model** to estimate the probability that a shot becomes a goal using NHL play-by-play data.

The goal of xG is to **separate process from results**:

* *Process* → how often and how well a player/team generates chances
* *Results* → whether those chances are converted into goals

This allows for better evaluation of players, teams, and strategies beyond raw goal totals.

---

## 📊 Data

Data is sourced from the **public NHL API**, using play-by-play event data across multiple seasons.

The pipeline:

* Parses raw game events into structured tables (games, events, shots)
* Filters to shot-based events using a Fenwick-style definition (shots on goal, missed shots, and goals). Shot counts reflect unblocked attempts (shots on goal + missed shots), rather than official NHL shots on goal totals, which only counts shots that hit the net.
* Constructs a **shot-level dataset**
* Enriches each shot with contextual features using **SQL window functions** (e.g., prior events, timing, sequence context)

---

## ⚙️ Methodology

### Model

An **interpretable logistic regression model** is used to estimate the probability that a given shot results in a goal.

### Key Features

The model uses a combination of spatial and contextual variables, including:

* **Shot distance** from goal
* **Shot angle** relative to the net
* **Rebound indicator** (based on prior shot within a short time window)
* **Game state** (even strength vs power play)
* **Time since previous event**
* **Shot type** (e.g., wrist, slap, backhand)

Feature engineering is performed using a combination of **SQL (CTEs, window functions)** and Python.

---

## ⏳ Train / Test Setup (Time-Based)

The model is evaluated using a **time-based split** to reflect real-world deployment:

* **Training:** 2023–2025 seasons
* **Testing / Analysis:** 2025–2026 regular season

This avoids data leakage and ensures the model is evaluated on *future data*, similar to how a team would use it in practice.



---

## 📈 How to Interpret the Results

Key metrics used throughout the app:

* **xG (Expected Goals):**
  Total quality of chances generated

* **Goals − xG:**
  Finishing relative to expectation by subtracting xG from Goals

  * Positive → strong finishing / potential hot streak
  * Negative → underperformance / potential bad luck

* **xG per Shot:**
  Average shot quality (shot selection)

### Example Interpretations:

* High xG, low goals → generating chances but not converting (*buy-low candidate*)
* Low xG, high goals → strong finishing or potentially unsustainable results
* High xG per shot → consistently high-danger opportunities

---

## ⚠️ Limitations

This model uses public data and includes several important limitations:

* No true **player/puck tracking data**, so spatial context is approximated
* **Rebound and rush indicators** are heuristic and inferred from event timing
* Shot coordinates may require **rink orientation assumptions**
* Does not explicitly model **goalie quality or defensive pressure**
* Small sample sizes can introduce noise in player-level results

These limitations are important when interpreting outputs, especially for individual players.

---

## 🚀 Future Improvements

Potential extensions of this app include:

* Incorporating **tracking data** for richer spatial features
* Improving **rush/transition detection**
* Extending to **passing value models** (Expected Threat / xT)
* Adding **goalie-adjusted metrics**
* Modeling **defensive impact and shot suppression**

---

## 🧩 Why This Matters

Separating **chance quality (xG)** from **actual scoring (goals)** helps teams:

* Identify undervalued players
* Evaluate sustainable performance vs variance
* Improve offensive and defensive strategy
* Make better roster and development decisions

This type of analysis is foundational in modern hockey analytics and front office decision-making.
"""


def format_nhl_season_id(season_id: int | str) -> str:
    """
    NHL API season ids are eight digits: startYear + endYear, e.g. 20252026 -> 2025-2026.
    """
    s = str(int(season_id))
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:]}"
    return s


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame:
    path = OUTPUTS / "test_shot_predictions.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    p = OUTPUTS / "metrics.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_coefficients() -> pd.DataFrame:
    p = OUTPUTS / "coefficients.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_player_names() -> pd.DataFrame:
    """Optional local SQLite player dimension (populated via `python -m src.clean --players`)."""
    try:
        from sqlalchemy import create_engine, text

        from src.config import database_url

        engine = create_engine(database_url(), future=True)
        with engine.connect() as conn:
            return pd.read_sql_query(text("SELECT player_id, full_name FROM players"), conn)
    except Exception:
        return pd.DataFrame(columns=["player_id", "full_name"])


def attach_player_names(df: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    if df.empty or names.empty:
        out = df.copy()
        out["player_name"] = out["shooter_id"].fillna(0).astype(int).astype(str) if not out.empty else ""
        return out
    out = df.merge(names, how="left", left_on="shooter_id", right_on="player_id")
    out["player_name"] = out["full_name"].fillna(out["shooter_id"].fillna(0).astype(int).astype(str))
    return out


def _row_still_shows_numeric_id(df: pd.DataFrame) -> pd.Series:
    """True where `player_name` is just the shooter id (no real name loaded yet)."""
    sid = df["shooter_id"].fillna(0).astype(int).astype(str)
    pn = df["player_name"].astype(str).str.strip()
    return pn == sid


@st.cache_data(show_spinner=False, ttl=86_400)
def _fetch_player_names_cached(player_ids: tuple[int, ...]) -> dict[int, str]:
    """Batch-resolve NHL player ids via public statsapi (cached 24h)."""
    if not player_ids:
        return {}
    from src.utils import fetch_player_names

    return fetch_player_names(player_ids)


def enrich_player_names_from_api(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace numeric-id fallbacks with real names using the NHL statsapi.
    Optionally persists new rows into SQLite `players` for faster loads next time.
    """
    if df.empty or "shooter_id" not in df.columns:
        return df
    out = df.copy()
    need = out.loc[_row_still_shows_numeric_id(out), "shooter_id"].dropna().astype(int).unique().tolist()
    if not need:
        return out
    try:
        name_map = _fetch_player_names_cached(tuple(sorted(set(need))))
    except Exception:
        return out
    if not name_map:
        return out
    sid_int = out["shooter_id"].fillna(0).astype(int)
    mapped = sid_int.map(name_map)
    out["player_name"] = mapped.where(mapped.notna(), out["player_name"])

    try:
        from sqlalchemy import create_engine, text

        from src.config import database_url

        eng = create_engine(database_url(), future=True)
        with eng.begin() as conn:
            for pid, nm in name_map.items():
                conn.execute(
                    text(
                        "INSERT OR REPLACE INTO players (player_id, full_name) VALUES (:pid, :nm)"
                    ),
                    {"pid": int(pid), "nm": str(nm)},
                )
    except Exception:
        pass

    return out


INSIGHT_MIN_SHOTS = 30

_PLAYER_AGG_COLS = ["shooter_id", "player_name", "shots", "goals", "xg", "goals_minus_xg", "xg_per_shot"]
_TEAM_AGG_COLS = ["team_abbr", "shots", "goals", "xg", "goals_minus_xg", "xg_per_shot"]


def _player_agg(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per shooter with volume, xG, finishing."""
    if df.empty or "player_name" not in df.columns:
        return pd.DataFrame(columns=_PLAYER_AGG_COLS)
    g = df.groupby(["shooter_id", "player_name"], as_index=False).agg(
        shots=("xg", "count"),
        goals=("is_goal", "sum"),
        xg=("xg", "sum"),
    )
    g["goals_minus_xg"] = g["goals"] - g["xg"]
    g["xg_per_shot"] = g["xg"] / g["shots"].replace(0, pd.NA)
    return g


def _overview_player_ranks(
    df_universe: pd.DataFrame, player_name: str, min_shots: int
) -> tuple[dict[str, int], int] | None:
    """
    Ranks for one player among skaters in df_universe with shots >= min_shots.
    Rank 1 = highest value (same ordering as Player Explorer sort defaults).
    """
    g = _player_agg(df_universe)
    if g.empty or "player_name" not in g.columns:
        return None
    g = g[g["shots"] >= min_shots]
    if g.empty:
        return None
    mask = g["player_name"] == player_name
    if not mask.any():
        return None
    n = int(len(g))
    ranks: dict[str, int] = {}
    for col in ("shots", "goals", "xg", "goals_minus_xg"):
        rk = g[col].rank(method="min", ascending=False)
        ranks[col] = int(rk.loc[mask].iloc[0])
    return ranks, n


def _overview_team_ranks(df_league: pd.DataFrame, team_abbr: str) -> tuple[dict[str, int], int] | None:
    """Ranks for one club among all teams in df_league on volume and finishing (rank 1 = highest)."""
    tg = _team_agg(df_league)
    if tg.empty or "team_abbr" not in tg.columns:
        return None
    tm = str(team_abbr).strip().upper()
    mask = tg["team_abbr"].astype(str).str.upper() == tm
    if not mask.any():
        return None
    n = int(len(tg))
    ranks: dict[str, int] = {}
    for col in ("shots", "goals", "xg", "goals_minus_xg"):
        rk = tg[col].rank(method="min", ascending=False)
        ranks[col] = int(rk.loc[mask].iloc[0])
    return ranks, n


def _team_agg(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "team_abbr" not in df.columns:
        return pd.DataFrame(columns=_TEAM_AGG_COLS)
    d = df.copy()
    d["team_abbr"] = d["team_abbr"].astype(str).str.upper()
    tg = d.groupby("team_abbr", as_index=False).agg(shots=("xg", "count"), goals=("is_goal", "sum"), xg=("xg", "sum"))
    tg["goals_minus_xg"] = tg["goals"] - tg["xg"]
    tg["xg_per_shot"] = tg["xg"] / tg["shots"].replace(0, pd.NA)
    return tg


def _format_player_table(g: pd.DataFrame) -> pd.DataFrame:
    out = g.copy()
    out["xG"] = out["xg"].round(2)
    out["xG per shot"] = out["xg_per_shot"].round(3)
    out["Goals − xG"] = out["goals_minus_xg"].round(2)
    return (
        out[
            ["player_name", "shots", "goals", "xG", "xG per shot", "Goals − xG"]
        ]
        .rename(columns={"player_name": "Player", "shots": "Shots", "goals": "Goals"})
        .reset_index(drop=True)
    )


def _leaderboard_sort_caption(sort_choice: str) -> str:
    """Short analyst-facing line for how the table is ordered."""
    return {
        "Goals − xG": "Sorted by finishing vs expectation (Goals − xG)",
        "xG": "Sorted by total chance generation (xG)",
        "xG per shot": "Sorted by average shot quality (xG per shot)",
    }.get(sort_choice, "")


def _format_team_table(tg: pd.DataFrame) -> pd.DataFrame:
    out = tg.copy()
    out["Team"] = out["team_abbr"].astype(str).map(nhl_team_display_name)
    out["xG"] = out["xg"].round(2)
    out["xG per shot"] = out["xg_per_shot"].round(3)
    out["Goals − xG"] = out["goals_minus_xg"].round(2)
    return (
        out[["Team", "shots", "goals", "xG", "xG per shot", "Goals − xG"]]
        .rename(columns={"shots": "Shots", "goals": "Goals"})
        .reset_index(drop=True)
    )


def _render_insight_box(df_league: pd.DataFrame, season_label: str) -> None:
    """League-wide bullets (season slice, ignores team/player sidebar filters)."""
    if df_league.empty:
        return
    st.subheader(f"Key insights ({season_label})")
    total_g = int(df_league["is_goal"].sum())
    total_xg = float(df_league["xg"].sum())
    delta = total_g - total_xg
    qual = "outperforming" if delta > 0 else "underperforming"
    bullets = [
        f"League shooting is **{qual}** model xG by **{delta:+.1f} goals** "
        f"({total_g:,} goals on {total_xg:,.1f} xG across {len(df_league):,} unblocked attempts).",
        "High **xG per shot** signals chance quality; raw **xG** also reflects volume.",
    ]
    pg = _player_agg(df_league)
    if not pg.empty and "shots" in pg.columns:
        pg = pg[pg["shots"] >= INSIGHT_MIN_SHOTS]
    if not pg.empty:
        top_xg = pg.sort_values("xg", ascending=False).iloc[0]
        worst_finish = pg.sort_values("goals_minus_xg", ascending=True).iloc[0]
        best_finish = pg.sort_values("goals_minus_xg", ascending=False).iloc[0]
        bullets.append(
            f"**Top xG generator** (min {INSIGHT_MIN_SHOTS} shots): **{top_xg['player_name']}** "
            f"({top_xg['xg']:.1f} xG on {int(top_xg['shots'])} shots)."
        )
        bullets.append(
            f"**Biggest underperformer vs xG**: **{worst_finish['player_name']}** "
            f"({worst_finish['goals']:.0f} goals vs {worst_finish['xg']:.1f} xG, Δ={worst_finish['goals_minus_xg']:+.1f})."
        )
        bullets.append(
            f"**Biggest overperformer vs xG**: **{best_finish['player_name']}** "
            f"({best_finish['goals']:.0f} goals vs {best_finish['xg']:.1f} xG, Δ={best_finish['goals_minus_xg']:+.1f})."
        )
    tg_ins = _team_agg(df_league)
    if not tg_ins.empty and len(tg_ins) >= 2:
        worst_team = tg_ins.sort_values("goals_minus_xg", ascending=True).iloc[0]
        best_team = tg_ins.sort_values("goals_minus_xg", ascending=False).iloc[0]
        wnm = nhl_team_display_name(str(worst_team["team_abbr"]))
        bnm = nhl_team_display_name(str(best_team["team_abbr"]))
        bullets.append(
            f"**Biggest underperformer vs xG (teams):** **{wnm}** "
            f"({worst_team['goals']:.0f} goals vs {worst_team['xg']:.1f} xG, Δ={worst_team['goals_minus_xg']:+.1f})."
        )
        bullets.append(
            f"**Biggest overperformer vs xG (teams):** **{bnm}** "
            f"({best_team['goals']:.0f} goals vs {best_team['xg']:.1f} xG, Δ={best_team['goals_minus_xg']:+.1f})."
        )
    bullets.append(
        "On the map, **darker red** marks higher per-shot xG—often the slot and inner slot—"
        "where the model expects finishing to be hardest on goalies."
    )
    for b in bullets:
        st.markdown(f"- {b}")


def _fmt_metric(metrics: dict, key: str, *, nd: int = 2, fallback: str) -> str:
    v = metrics.get(key)
    if v is None:
        return fallback
    return f"{float(v):.{nd}f}"


def _model_performance_and_feature_writeup(metrics: dict) -> str:
    """Narrative for Model Diagnostics tab; uses `metrics.json` test values when present."""
    auc = _fmt_metric(metrics, "test_roc_auc", nd=2, fallback="0.73")
    ll = _fmt_metric(metrics, "test_log_loss", nd=2, fallback="0.24")
    br = _fmt_metric(metrics, "test_brier", nd=2, fallback="0.06")
    return f"""
### Model Performance (2025–2026 Holdout)

The model shows solid predictive performance on the held-out 2025–2026 season:

* **ROC-AUC (~{auc}):** Good separation between goals and non-goals
* **Log Loss (~{ll}):** Reasonable probability calibration
* **Brier Score (~{br}):** Accurate probability estimates overall

Overall, the model captures shot quality well, though like most public xG models, it is limited by the absence of full tracking data.

### Feature Interpretation

Model coefficients show how different factors influence scoring probability:

* **Shot distance (negative):** Shots farther from the net are less likely to score
* **Shot types:** Certain shot types (e.g., slap shots) have lower conversion rates on average
* **Score differential:** Game context impacts shot behavior and probability
* **Rebounds / quick sequences (if included):** Increase scoring likelihood

Positive coefficients increase goal probability, while negative coefficients decrease it.

These results align with hockey intuition, indicating the model is learning meaningful patterns rather than noise.
""".strip()


def _model_interpretation_blurb(metrics: dict) -> None:
    st.markdown(
        """
**How to read this**

- **ROC-AUC** summarizes rank ordering (goal vs non-goal); values in the low 0.70s are typical for a simple geometry + state model on public data.
- **Log loss / Brier** punish confident misses; together they flag overconfidence on rare goals.
- **Calibration** (reliability curve): if points sit *below* the diagonal at high predicted probabilities, the model is **too optimistic** on its best chances; *above* means **too pessimistic**.
        """
    )
    auc = metrics.get("test_roc_auc")
    brier = metrics.get("test_brier")
    if auc is not None and brier is not None:
        st.markdown(
            f"On the held-out test window, this build sits at **ROC-AUC ≈ {float(auc):.3f}** and "
            f"**Brier ≈ {float(brier):.3f}**—use the calibration figure to judge whether probabilities "
            f"are trustworthy in the **~0.1–0.4 xG** range where most shots cluster."
        )
    st.markdown(
        "_Heuristic read_: logistic models often **slightly underfit the extreme tail** "
        "(very high-danger chances) because goals remain rare; the mid-probability bins are usually **best calibrated**."
    )


def main() -> None:
    st.set_page_config(page_title="2025-2026 NHL Expected Goals (xG) Model", layout="wide")
    st.markdown(
        "<style>"
        "[data-testid='stDeployButton']{display:none!important}"
        ".stAppDeployButton{display:none!important}"
        "[data-testid='stMainMenuPopover'] [data-testid='stMainMenuList']+div{display:none!important}"
        "[data-testid='stMainMenuItem-rerun'],"
        "[data-testid='stMainMenuItem-autoRerun'],"
        "[data-testid='stMainMenuItem-clearCache']{display:none!important}"
        "</style>",
        unsafe_allow_html=True,
    )
    st.title("2025-2026 NHL Expected Goals (xG) Model")

    preds = load_predictions()
    metrics = load_metrics()
    coefs = load_coefficients()
    names = load_player_names()
    preds = attach_player_names(preds, names)
    with st.spinner("Resolving player names (first load may take ~30–90s)…"):
        try:
            preds = enrich_player_names_from_api(preds)
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Player name lookup failed; showing IDs where needed. ({exc})")

    # App explores the test / analysis season only (predictions parquet is that window).
    if not preds.empty and "season" in preds.columns:
        sid = int(TEST_SEASON_ID)
        preds = preds[preds["season"].astype(int) == sid].copy()
        if preds.empty:
            st.error(
                f"No rows for test season **{sid}** in predictions. "
                "Re-run `python -m src.train` so `outputs/test_shot_predictions.parquet` "
                "includes the 2025–2026 holdout."
            )
            return

    if preds.empty:
        st.error(
            "No prediction file found. From the repo root, run the pipeline:\n\n"
            "`python -m src.clean --schema --context` → `python -m src.ingest ...` → "
            "`python -m src.features` → `python -m src.train` → `python -m src.evaluate`"
        )
        return

    season = int(TEST_SEASON_ID)
    season_label = format_nhl_season_id(season)

    df_insights = preds.copy()
    df = df_insights.copy()

    teams = sorted(df["team_abbr"].dropna().astype(str).str.upper().unique().tolist()) if "team_abbr" in df.columns else []

    tab_about, tab_overview, tab_players, tab_teams, tab_model = st.tabs(
        _APP_TAB_LABELS,
        key=_MAIN_TABS_STATE_KEY,
        on_change="rerun",
    )

    # Sidebar filters hidden on Model Diagnostics (metrics/coefs do not use them).
    if tab_model.open is not True:
        st.sidebar.header("Filters")
        team = st.sidebar.selectbox(
            "Team (optional)",
            ["All"] + teams,
            format_func=lambda ab: "All teams" if ab == "All" else nhl_team_display_name(ab),
            key="filter_team",
        )
        if "filter_team_prev" not in st.session_state:
            st.session_state["filter_team_prev"] = team
        if "min_shots_slider" not in st.session_state:
            st.session_state["min_shots_slider"] = 100 if team == "All" else 50
        if st.session_state["filter_team_prev"] != team:
            st.session_state["filter_team_prev"] = team
            st.session_state["min_shots_slider"] = 100 if team == "All" else 50

        min_shots = st.sidebar.slider(
            "Minimum shots (leaderboards)",
            min_value=20,
            max_value=200,
            step=5,
            key="min_shots_slider",
            help="Suggested: ~75–100 shots league-wide, ~40–60 for one team. Floor 20 for exploration.",
        )
    else:
        team_opts = ["All"] + teams
        team = str(st.session_state.get("filter_team", "All"))
        if team not in team_opts:
            team = "All"
        min_shots = int(st.session_state.get("min_shots_slider", 100))
        min_shots = max(20, min(200, min_shots))

    # Hide Player filter on Team Explorer; hide all filters above on Model Diagnostics.
    if tab_teams.open is not True and tab_model.open is not True:
        if team != "All" and "team_abbr" in df.columns and "player_name" in df.columns:
            tm = str(team).strip().upper()
            team_mask = df["team_abbr"].astype(str).str.upper() == tm
            players = sorted(df.loc[team_mask, "player_name"].dropna().unique().tolist())
        else:
            players = sorted(df["player_name"].dropna().unique().tolist()) if "player_name" in df.columns else []
        player_options = ["All"] + players
        if "filter_player" in st.session_state and st.session_state["filter_player"] not in player_options:
            st.session_state["filter_player"] = "All"
        player = st.sidebar.selectbox("Player (optional)", player_options, key="filter_player")
    else:
        player = str(st.session_state.get("filter_player", "All"))

    if team != "All" and "team_abbr" in df.columns:
        df = df[df["team_abbr"].astype(str).str.upper() == str(team).strip().upper()]
    if player != "All" and "player_name" in df.columns:
        df = df[df["player_name"] == player]

    # Team Explorer uses team (+ season) only; sidebar Player does not apply there.
    df_teams = df_insights.copy()
    if team != "All" and "team_abbr" in df_teams.columns:
        df_teams = df_teams[df_teams["team_abbr"].astype(str).str.upper() == str(team).strip().upper()]

    with tab_about:
        st.markdown(ABOUT_PAGE_MARKDOWN)

    with tab_overview:
        st.subheader("Summary")
        if df.empty:
            st.warning(
                "No shots match the current filters (for example, a player who does not play for the "
                "selected team). Adjust **Team** and **Player**, or clear filters to see data again."
            )
        total_shots = int(len(df))
        total_goals = int(df["is_goal"].sum()) if not df.empty else 0
        total_xg = float(df["xg"].sum()) if not df.empty else 0.0
        total_gmx = float(total_goals - total_xg)

        df_rank_u = df_insights.copy()
        if team != "All" and "team_abbr" in df_rank_u.columns:
            df_rank_u = df_rank_u[df_rank_u["team_abbr"].astype(str).str.upper() == str(team).strip().upper()]
        rank_pack: tuple[dict[str, int], int] | None = None
        rank_caption: str | None = None
        if player != "All":
            rank_pack = _overview_player_ranks(df_rank_u, player, min_shots)
            if rank_pack is not None:
                rank_caption = (
                    f"Ranks match **Player Explorer**: skaters with at least **{min_shots}** shots in this season "
                    f"slice"
                    + (" for the selected team." if team != "All" else " league-wide.")
                )
            else:
                # Player is below the leaderboard shot cutoff; still show ranks vs a 20-shot floor.
                rank_pack = _overview_player_ranks(df_rank_u, player, 20)
                if rank_pack is not None:
                    rank_caption = (
                        f"Ranks include skaters with at least **20** shots in this season slice"
                        + (" for the selected team" if team != "All" else " league-wide")
                        + f" (this skater is below the **{min_shots}**-shot minimum used on **Player Explorer**)."
                    )
                else:
                    rank_pack = _overview_player_ranks(df_rank_u, player, 1)
                    if rank_pack is not None:
                        rank_caption = (
                            "Ranks include every skater with at least **one** shot in this season slice"
                            + (" for the selected team" if team != "All" else " league-wide")
                            + f" (below **{min_shots}**- and **20**-shot Explorer-style cutoffs)."
                        )

        team_rank_pack: tuple[dict[str, int], int] | None = None
        team_rank_caption: str | None = None
        if team != "All" and player == "All":
            team_rank_pack = _overview_team_ranks(df_insights, team)
            if team_rank_pack is not None:
                _n_teams = team_rank_pack[1]
                team_rank_caption = (
                    f"**Team** ranks compare this club to all **{_n_teams}** teams with shots in this season slice "
                    "(rank **1** = highest Shots, Goals, xG, or Goals − xG)."
                )

        def _rank_suffix(metric_key: str) -> str:
            if rank_pack is not None:
                ranks, n = rank_pack
                return f" (#{ranks[metric_key]} of {n})"
            if team_rank_pack is not None:
                ranks, n = team_rank_pack
                return f" (#{ranks[metric_key]} of {n})"
            return ""

        st.metric("Shots (filtered)", f"{total_shots:,}{_rank_suffix('shots')}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Goals", f"{total_goals:,}{_rank_suffix('goals')}")
        c2.metric("xG", f"{total_xg:,.2f}{_rank_suffix('xg')}")
        c3.metric("Goals − xG", f"{total_gmx:,.2f}{_rank_suffix('goals_minus_xg')}")
        if rank_caption:
            st.caption(rank_caption)
        elif team_rank_caption:
            st.caption(team_rank_caption)

        _render_insight_box(df_insights, season_label)

        st.subheader("Shot map (color = model xG per shot)")
        if df.empty:
            st.info("Shot map is empty until at least one shot matches the filters above.")
        else:
            xg_hi = float(max(0.25, min(0.6, df["xg"].quantile(0.99))))
            fig = px.scatter(
                df,
                x="x_coord",
                y="y_coord",
                color="xg",
                color_continuous_scale=_XG_COLORSCALE,
                range_color=[0.0, xg_hi],
                opacity=0.72,
                height=600,
                title="Shot locations (color = xG per shot)",
                labels={"x_coord": "x (ft)", "y_coord": "y (ft)", "xg": "xG (per shot)"},
            )
            fig.update_traces(marker=dict(size=8, line=dict(width=0)))
            fig.update_layout(coloraxis_colorbar=dict(title="xG", tickfont=dict(color=_CHART_TEXT), title_font=dict(color=_CHART_TEXT)))
            _style_plotly_fig(fig, for_rink=True)
            st.plotly_chart(fig, use_container_width=True)

    with tab_players:
        st.subheader(f"Player finishing vs expected ({season_label})")
        st.caption(
            f"Minimum shots = **{min_shots}** (applied to leaderboard and scatterplot). "
            "Sidebar defaults to **100** for all teams and **50** when a single team is selected—you can still adjust."
        )
        g: pd.DataFrame | None = None
        if "player_name" not in df.columns:
            st.info("Player names unavailable. Run `python -m src.clean --players` after ingestion.")
        else:
            g = _player_agg(df)
            g = g[g["shots"] >= min_shots]
            if g.empty:
                if team != "All" and player != "All":
                    team_nm = nhl_team_display_name(team)
                    st.info(
                        f"**{player}** doesn't meet this shot threshold for **{team_nm}** in the current filters "
                        f"(minimum **{min_shots}** shots). Try lowering the minimum shots filter."
                    )
                elif team != "All":
                    team_nm = nhl_team_display_name(team)
                    st.info(
                        f"No players meet this shot threshold for **{team_nm}** in the current filters "
                        f"(minimum **{min_shots}** shots). Try lowering the minimum shots filter "
                        "(about **40–60** shots is a reasonable range when viewing one team)."
                    )
                elif player != "All":
                    st.info(
                        f"**{player}** doesn't meet this shot threshold for the current filters "
                        f"(minimum **{min_shots}** shots). Try lowering the minimum shots filter."
                    )
                else:
                    st.info(
                        "No players meet this shot threshold for the current filters. "
                        "Try lowering the minimum shots filter (about **75–100** shots is a common league-wide cutoff)."
                    )
                g = None
        if g is not None:
            sort_choice = st.radio(
                "Sort leaderboard by",
                ["Goals − xG", "xG", "xG per shot"],
                horizontal=True,
                index=0,
                key="sort_player_lb",
            )
            st.caption(
                "Sort by Goals − xG to see finishing relative to expectation, xG to see total chance generation, "
                "or xG per shot to evaluate shot quality."
            )
            sort_col = {"Goals − xG": "goals_minus_xg", "xG": "xg", "xG per shot": "xg_per_shot"}[sort_choice]
            g = g.sort_values(sort_col, ascending=False).reset_index(drop=True)
            st.caption(_leaderboard_sort_caption(sort_choice))
            st.dataframe(
                _format_player_table(g),
                use_container_width=True,
                height=480,
                hide_index=True,
            )

            st.subheader("Goals vs xG (same cutoff)")
            st.caption(
                "Each point is one skater in the table above. The dashed diagonal is **Goals = xG** (finishing "
                "right on expectation). **Above** that line means more goals than the model expected from shot "
                "quality; **below** means fewer—often read as cold finishing or bad luck over a short window."
            )
            gg = g.copy()
            fig2 = px.scatter(
                gg,
                x="xg",
                y="goals",
                hover_data=["player_name", "shots", "shooter_id", "xg_per_shot", "goals_minus_xg"],
                labels={
                    "xg": "xG",
                    "goals": "Goals",
                    "player_name": "Player",
                    "shots": "Shots",
                    "xg_per_shot": "xG / shot",
                    "goals_minus_xg": "Goals − xG",
                },
            )
            max_val = float(max(gg["xg"].max(), gg["goals"].max()))
            fig2.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=max_val,
                y1=max_val,
                line=dict(color="#4a6d74", dash="dash"),
            )
            _style_plotly_fig(fig2)
            fig2.update_traces(marker=dict(color=_CHART_MARKER, size=9, line=dict(width=0)))
            st.plotly_chart(fig2, use_container_width=True)

    with tab_teams:
        st.subheader(f"Team explorer ({season_label})")
        if "team_abbr" not in df_teams.columns:
            st.warning("Team abbreviations missing from predictions file.")
        else:
            tg = _team_agg(df_teams)
            st.caption(
                f"The minimum-shots slider (**{min_shots}**) applies on the **Player** tab only. "
                "This tab ignores the **Player** filter—team totals use the **Team** filter (if any) and the "
                "season slice only. Slider defaults to **100** (all teams) or **50** (one team selected)."
            )
            if tg.empty:
                st.info("No team rows for the current filters.")
            else:
                team_sort = st.radio(
                    "Sort team leaderboard by",
                    ["Goals − xG", "xG", "xG per shot"],
                    horizontal=True,
                    index=0,
                    key="sort_team_lb",
                )
                tsort = {"Goals − xG": "goals_minus_xg", "xG": "xg", "xG per shot": "xg_per_shot"}[team_sort]
                tg = tg.sort_values(tsort, ascending=False).reset_index(drop=True)
                st.caption(_leaderboard_sort_caption(team_sort))
                st.dataframe(
                    _format_team_table(tg),
                    use_container_width=True,
                    hide_index=True,
                )

                st.subheader("Team goals vs xG")
                tbar = tg.sort_values("xg", ascending=False).copy()
                tbar["team_name"] = tbar["team_abbr"].astype(str).map(nhl_team_display_name)
                fig_bar = go.Figure(
                    data=[
                        go.Bar(name="Goals", x=tbar["team_name"], y=tbar["goals"], marker_color=_CHART_MARKER),
                        go.Bar(name="xG", x=tbar["team_name"], y=tbar["xg"], marker_color=_CHART_ACCENT),
                    ]
                )
                fig_bar.update_layout(
                    barmode="group",
                    height=520,
                    xaxis_title="Team",
                    yaxis_title="Goals / xG",
                    xaxis_tickangle=-45,
                    legend_title_text="",
                    margin=dict(b=120),
                )
                _style_plotly_fig(fig_bar)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.subheader("Team shot map (color = xG)")
            team_opts = sorted(df_teams["team_abbr"].dropna().unique().tolist())
            if not team_opts:
                st.info("No team rows after filters.")
            else:
                team_pick = st.selectbox(
                    "Team for map",
                    team_opts,
                    format_func=nhl_team_display_name,
                )
                tdf = df_teams[df_teams["team_abbr"].astype(str).str.upper() == str(team_pick).strip().upper()]
                xg_hi_t = float(max(0.25, min(0.6, tdf["xg"].quantile(0.99)))) if len(tdf) else 0.35
                map_title = nhl_team_display_name(team_pick)
                fig3 = px.scatter(
                    tdf,
                    x="x_coord",
                    y="y_coord",
                    color="xg",
                    color_continuous_scale=_XG_COLORSCALE,
                    range_color=[0.0, xg_hi_t],
                    opacity=0.72,
                    height=600,
                    title=f"{map_title} — shot danger (xG)",
                    labels={"x_coord": "x (ft)", "y_coord": "y (ft)", "xg": "xG"},
                )
                fig3.update_traces(marker=dict(size=8, line=dict(width=0)))
                fig3.update_layout(coloraxis_colorbar=dict(title="xG", tickfont=dict(color=_CHART_TEXT), title_font=dict(color=_CHART_TEXT)))
                _style_plotly_fig(fig3, for_rink=True)
                st.plotly_chart(fig3, use_container_width=True)
                st.caption(
                    "Each point is one **unblocked attempt** by this team in the filtered season slice. "
                    "**Color** is the model's xG for that shot (cooler = lower estimated danger, warmer = higher). "
                    "**Position** is rink location—dense clusters show where the team shoots most; bright dots "
                    "highlight individual high-danger looks (e.g. slot / inner slot), not official shot charts."
                )

    with tab_model:
        st.subheader("Held-out metrics (from training run)")
        st.json(metrics)

        st.subheader("Interpretation")
        _model_interpretation_blurb(metrics)

        st.subheader("Top coefficients (standardized numeric + one-hot shot types)")
        st.caption(
            "Each coefficient shows how a feature affects the chance of a shot becoming a goal. "
            "Positive values increase scoring probability, while negative values decrease it. "
            "The odds ratio translates this into a multiplier: values above 1 increase scoring odds, "
            "and values below 1 decrease them."
        )
        if not coefs.empty:
            st.dataframe(coefs.head(25).reset_index(drop=True), use_container_width=True, hide_index=True)

        st.markdown(_model_performance_and_feature_writeup(metrics))

        cal_path = OUTPUTS / "fig_calibration.png"
        if cal_path.exists():
            st.subheader("Calibration (reliability)")
            st.caption(
                "**How to read this:** The dashed line is **perfect calibration**: for each bin of predicted "
                "goal probability, the height should match that probability (e.g. 0.2 on the x-axis lines up with "
                "0.2 on the y-axis). If the **model curve runs below** the diagonal, the model is **too optimistic** "
                "in that range (stated probabilities higher than how often goals actually occur). **Above** the "
                "line means **too pessimistic**. Use this to judge how trustworthy **raw probabilities** are, "
                "especially in the mid-to-high range where most non-goals still live."
            )
            st.image(str(cal_path), use_container_width=True)


if __name__ == "__main__":
    main()
