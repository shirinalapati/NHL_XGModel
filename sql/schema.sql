-- NHL xG app — core relational model (SQLite-compatible DDL)
-- Run once before ingestion.

PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS games (
    game_id INTEGER PRIMARY KEY,
    season INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    game_type INTEGER NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    home_abbr TEXT,
    away_abbr TEXT,
    home_defending_side TEXT,
    venue TEXT,
    start_time_utc TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_games_season_date ON games (season, game_date);

CREATE TABLE IF NOT EXISTS events (
    game_id INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    sort_order INTEGER NOT NULL,
    period INTEGER NOT NULL,
    time_in_period TEXT,
    game_seconds INTEGER,
    type_desc_key TEXT NOT NULL,
    team_id INTEGER,
    x_coord REAL,
    y_coord REAL,
    situation_code TEXT,
    PRIMARY KEY (game_id, event_id),
    FOREIGN KEY (game_id) REFERENCES games (game_id)
);

CREATE INDEX IF NOT EXISTS idx_events_game_sort ON events (game_id, sort_order);

CREATE TABLE IF NOT EXISTS shots (
    game_id INTEGER NOT NULL,
    event_id INTEGER NOT NULL,
    sort_order INTEGER NOT NULL,
    season INTEGER NOT NULL,
    game_date TEXT NOT NULL,
    period INTEGER NOT NULL,
    time_in_period TEXT,
    game_seconds INTEGER,
    team_id INTEGER NOT NULL,
    opponent_id INTEGER NOT NULL,
    shooter_id INTEGER NOT NULL,
    goalie_id INTEGER,
    x_coord REAL NOT NULL,
    y_coord REAL NOT NULL,
    shot_type TEXT,
    event_type TEXT NOT NULL,
    situation_code TEXT,
    home_away TEXT NOT NULL,
    is_goal INTEGER NOT NULL,
    strength_state TEXT,
    away_score INTEGER,
    home_score INTEGER,
    PRIMARY KEY (game_id, event_id),
    FOREIGN KEY (game_id) REFERENCES games (game_id)
);

CREATE INDEX IF NOT EXISTS idx_shots_season ON shots (season);
CREATE INDEX IF NOT EXISTS idx_shots_game ON shots (game_id);
CREATE INDEX IF NOT EXISTS idx_shots_shooter ON shots (shooter_id);
CREATE INDEX IF NOT EXISTS idx_shots_team ON shots (team_id);

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY,
    full_name TEXT NOT NULL,
    updated_at TEXT DEFAULT (datetime('now'))
);
