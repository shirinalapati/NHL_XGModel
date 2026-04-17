"""
Central configuration for the NHL xG pipeline.

Season IDs follow the NHL API convention, e.g. 20232024 == 2023-24 season.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
SQL_DIR = ROOT / "sql"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

# --- NHL API ---
NHL_SCHEDULE_URL = "https://api-web.nhle.com/v1/schedule/{date}"
NHL_PBP_URL = "https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
# Legacy batch endpoint (often blocked or flaky); prefer `NHL_PLAYER_LANDING_URL` in utils.
NHL_PEOPLE_URL = "https://statsapi.web.nhl.com/api/v1/people"
NHL_PLAYER_LANDING_URL = "https://api-web.nhle.com/v1/player/{player_id}/landing"

# NHL club abbreviations → full franchise names (for dashboards / filtering labels).
NHL_TEAM_FULL_NAMES: dict[str, str] = {
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "ATL": "Atlanta Thrashers",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CGY": "Calgary Flames",
    "CAR": "Carolina Hurricanes",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "CBJ": "Columbus Blue Jackets",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montreal Canadiens",
    "NSH": "Nashville Predators",
    "NJD": "New Jersey Devils",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Mammoth",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}


def nhl_team_display_name(abbrev: str | None) -> str:
    """Return full team name for a standard NHL tricode, or the raw abbrev if unknown."""
    if not abbrev:
        return ""
    key = str(abbrev).strip().upper()
    return NHL_TEAM_FULL_NAMES.get(key, key)


# --- Season split (time-based; no random split) ---
# Train on full 2023-24 and 2024-25 regular seasons, plus early 2025-26 (calendar 2025
# portion of the regular season). Test on 2025-26 games from Jan 1, 2026 onward.
# This matches "train through 2025" vs "evaluate on 2025-26" without peeking at future
# games within the same league season.
TRAIN_SEASON_IDS_FULL = ("20232024", "20242025")
TRAIN_PARTIAL_SEASON_ID = "20252026"
TEST_SEASON_ID = "20252026"
# Inclusive end for training rows from the partial season; test starts strictly after.
TRAIN_PARTIAL_CUTOFF_DATE = "2025-12-31"
TEST_START_DATE = "2026-01-01"

# Regular season only (NHL gameType in API == 2)
GAME_TYPE_REGULAR = 2

# --- Modeling ---
RANDOM_STATE = 42
GOAL_LINE_X_FT = 89.0  # NHL standard distance center ice to goal line (~89 ft)
NET_HALF_WIDTH_FT = 6.0  # goal mouth half-width for angle approximation

# Shot event types treated as Fenwick-style attempts (unblocked)
SHOT_EVENT_TYPES = ("shot-on-goal", "goal", "missed-shot")

# --- Database ---
DEFAULT_SQLITE_PATH = DATA_PROCESSED / "nhl_xg.db"


def database_url() -> str:
    """Return SQLAlchemy URL: prefer DATABASE_URL env, else SQLite."""
    import os

    env = os.environ.get("DATABASE_URL")
    if env:
        return env
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{DEFAULT_SQLITE_PATH}"


def ensure_directories() -> None:
    for p in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR):
        p.mkdir(parents=True, exist_ok=True)
