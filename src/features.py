"""
Shot geometry + modeling features built on top of `shots_with_context`.

Coordinate convention (documented for reproducibility):
- NHL publishes absolute rink coordinates with `homeTeamDefendingSide` (`left`/`right`)
  indicating which side of the *fixed* coordinate frame the home team defends.
- When the home team defends the `right` side, we mirror both x and y so the frame
  matches the `left` convention (home defends negative x).
- After mirroring, the home goal line is near x = -89 ft and the away goal line near
  x = +89 ft. The shooting team always attacks the far positive x net if they are the
  home team, and the far negative x net if they are the away team.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text

from src.config import (
    GOAL_LINE_X_FT,
    NET_HALF_WIDTH_FT,
    OUTPUTS_DIR,
    TEST_SEASON_ID,
    TEST_START_DATE,
    TRAIN_PARTIAL_CUTOFF_DATE,
    TRAIN_PARTIAL_SEASON_ID,
    TRAIN_SEASON_IDS_FULL,
    ensure_directories,
)
from src.utils import get_engine, setup_logging

logger = logging.getLogger("nhl_xg.features")


@dataclass(frozen=True)
class SplitConfig:
    train_full_seasons: tuple[int, ...] = tuple(int(x) for x in TRAIN_SEASON_IDS_FULL)
    train_partial_season: int = int(TRAIN_PARTIAL_SEASON_ID)
    train_partial_end: str = TRAIN_PARTIAL_CUTOFF_DATE
    test_season: int = int(TEST_SEASON_ID)
    test_start: str = TEST_START_DATE


def mirror_if_home_defends_right(x: np.ndarray, y: np.ndarray, side: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    s = side.fillna("left").str.lower().to_numpy()
    mx = x.copy()
    my = y.copy()
    mask = s == "right"
    mx[mask] *= -1
    my[mask] *= -1
    return mx, my


def attacking_goal_x(shooting_home: np.ndarray) -> np.ndarray:
    """Return x-coordinate of the net being attacked (+89 for home shooters, -89 for away)."""
    gx = np.where(shooting_home, GOAL_LINE_X_FT, -GOAL_LINE_X_FT)
    return gx


def calculate_shot_distance(x: np.ndarray, y: np.ndarray, shooting_home: np.ndarray) -> np.ndarray:
    gx = attacking_goal_x(shooting_home)
    return np.hypot(x - gx, y - 0.0)


def calculate_shot_angle(x: np.ndarray, y: np.ndarray, shooting_home: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (angle_deg_from_centerline, abs_angle_deg).

    We use the angle between the shooter-to-goal vector and the long axis toward the goal:
    theta = atan2(|y|, |goal_x - x|), reported in degrees (0 = straight ahead from slot).
    """
    gx = attacking_goal_x(shooting_home)
    depth = np.abs(gx - x)
    width = np.abs(y)
    theta = np.arctan2(width, np.maximum(depth, 0.5))  # avoid divide-by-zero at goal mouth
    deg = np.degrees(theta)
    return deg, np.abs(deg)


def empty_net_flag(situation_code: pd.Series, shooting_home: np.ndarray) -> np.ndarray:
    """Digits 3/4 are away/home goalie flags (0 = empty net)."""
    out = np.zeros(len(situation_code), dtype=int)
    codes = situation_code.fillna("").astype(str).str.strip()
    for i, code in enumerate(codes):
        if len(code) != 4 or not code.isdigit():
            continue
        away_g, home_g = int(code[2]), int(code[3])
        if shooting_home[i]:
            if home_g == 0:
                out[i] = 1
        else:
            if away_g == 0:
                out[i] = 1
    return out


def assign_split_vectorized(df: pd.DataFrame, cfg: SplitConfig) -> pd.Series:
    season = df["season"].astype(int)
    gd = pd.to_datetime(df["game_date"]).dt.date
    train_end = date.fromisoformat(cfg.train_partial_end)
    test_start = date.fromisoformat(cfg.test_start)

    is_train_full = season.isin(cfg.train_full_seasons)
    is_train_partial = (season == cfg.train_partial_season) & (gd <= train_end)
    is_test = (season == cfg.test_season) & (gd >= test_start)

    split = pd.Series("drop", index=df.index, dtype="object")
    split[is_train_full | is_train_partial] = "train"
    split[is_test] = "test"
    # If a row satisfies both (shouldn't), prefer test label for strictness
    split[is_test] = "test"
    return split


def load_shots_with_context(engine) -> pd.DataFrame:
    q = text(
        """
        SELECT
            swc.*,
            g.home_abbr AS game_home_abbr,
            g.away_abbr AS game_away_abbr
        FROM shots_with_context swc
        JOIN games g ON g.game_id = swc.game_id
        """
    )
    return pd.read_sql_query(q, engine)


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return a dataframe with engineered numeric/categorical columns + metadata columns."""
    x = df["x_coord"].to_numpy(dtype=float)
    y = df["y_coord"].to_numpy(dtype=float)
    side = df["game_home_defending_side"] if "game_home_defending_side" in df.columns else df.get("home_defending_side", pd.Series(["left"] * len(df)))
    mx, my = mirror_if_home_defends_right(x, y, side)

    shooting_home = (df["team_id"].to_numpy() == df["home_team_id"].to_numpy()).astype(int)
    dist = calculate_shot_distance(mx, my, shooting_home.astype(bool))
    ang, abs_ang = calculate_shot_angle(mx, my, shooting_home.astype(bool))

    out = df.copy()
    out["x_attack"] = mx
    out["y_attack"] = my
    out["shot_distance"] = dist
    out["shot_angle"] = ang
    out["abs_shot_angle"] = abs_ang
    out["shooting_home"] = shooting_home

    tsp = pd.to_numeric(out.get("time_since_prev_event_seconds"), errors="coerce")
    out["time_since_prev_event_seconds"] = tsp.clip(lower=0, upper=120).fillna(120)

    out["rebound_flag"] = pd.to_numeric(out.get("rebound_flag"), errors="coerce").fillna(0).astype(int)
    out["rush_flag"] = pd.to_numeric(out.get("rush_flag_heuristic"), errors="coerce").fillna(0).astype(int)
    out["same_team_prev_shot_flag"] = (
        pd.to_numeric(out.get("same_team_prev_shot_flag"), errors="coerce").fillna(0).astype(int)
    )

    st = out["strength_state"].fillna("UNK")
    out["is_pp"] = (st == "PP").astype(int)
    out["is_sh"] = (st == "SH").astype(int)
    out["is_ev"] = (st == "EV").astype(int)
    out["is_3v3"] = (st == "3v3").astype(int)

    out["empty_net"] = empty_net_flag(out["situation_code"].astype(str), shooting_home.astype(bool))

    out["home_away_home"] = (out["home_away"].str.upper() == "HOME").astype(int)

    out["period"] = pd.to_numeric(out["period"], errors="coerce").fillna(1).astype(int)

    hs = pd.to_numeric(out.get("home_score"), errors="coerce").fillna(0)
    aw = pd.to_numeric(out.get("away_score"), errors="coerce").fillna(0)
    out["score_diff_shooter"] = np.where(out["team_id"].to_numpy() == out["home_team_id"].to_numpy(), hs - aw, aw - hs)

    out["shot_type"] = out["shot_type"].fillna("unknown").astype(str).str.lower()

    if {"game_home_abbr", "game_away_abbr"}.issubset(out.columns):
        out["team_abbr"] = np.where(
            out["team_id"].to_numpy() == out["home_team_id"].to_numpy(),
            out["game_home_abbr"].astype(str),
            out["game_away_abbr"].astype(str),
        )

    cfg = SplitConfig()
    out["dataset_split"] = assign_split_vectorized(out, cfg)
    return out


def export_modeling_table(df: pd.DataFrame, path: str) -> None:
    ensure_directories()
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    logger.info("Wrote modeling table: %s (%s rows)", path, len(df))


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineer features from SQL shot context table.")
    parser.add_argument(
        "--out",
        default=str(OUTPUTS_DIR / "modeling_dataset.parquet"),
        help="Output parquet path",
    )
    args = parser.parse_args()

    setup_logging()
    engine = get_engine()
    base = load_shots_with_context(engine)
    if base.empty:
        raise SystemExit("shots_with_context is empty. Run ingest + `python -m src.clean --context`.")
    feats = build_model_features(base)
    export_modeling_table(feats, args.out)


if __name__ == "__main__":
    main()
