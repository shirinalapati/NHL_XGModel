"""
Download NHL regular-season schedules and play-by-play, persist raw JSON,
and load normalized `games`, `events`, and `shots` tables.

Data source: NHL Web API (`api-web.nhle.com`) + schedule discovery by date.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterator

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import (
    DATA_RAW,
    GAME_TYPE_REGULAR,
    NHL_PBP_URL,
    NHL_SCHEDULE_URL,
    SHOT_EVENT_TYPES,
    ensure_directories,
)
from src.utils import get_engine, http_get_json, parse_game_date, period_clock_to_seconds, setup_logging

logger = logging.getLogger("nhl_xg.ingest")


@dataclass
class GameMeta:
    game_id: int
    season: int
    game_date: date
    home_team_id: int
    away_team_id: int
    home_abbr: str | None
    away_abbr: str | None
    home_defending_side: str | None = None


def _decode_scores_from_situation_or_details(
    play: dict[str, Any],
) -> tuple[int | None, int | None]:
    d = play.get("details") or {}
    a = d.get("awayScore")
    h = d.get("homeScore")
    if a is not None and h is not None:
        return int(a), int(h)
    return None, None


def _situation_skaters(code: str | None) -> tuple[int | None, int | None, int | None, int | None]:
    """
    NHL `situationCode` is four digits: away skaters, home skaters, away goalie flag, home goalie flag.
    Goalie flag 1 = goalie in net, 0 = empty net.
    """
    if not code or len(code) != 4 or not code.isdigit():
        return None, None, None, None
    return int(code[0]), int(code[1]), int(code[2]), int(code[3])


def _strength_label(
    situation_code: str | None,
    shooting_team_id: int,
    home_team_id: int,
) -> str:
    away_s, home_s, away_g, home_g = _situation_skaters(situation_code)
    if away_s is None or home_s is None:
        return "UNK"
    shooting_home = shooting_team_id == home_team_id
    if away_s == home_s:
        if away_s == 3:
            return "3v3"
        if away_s == 4:
            return "4v4"
        return "EV"
    if shooting_home:
        if home_s > away_s:
            return "PP"
        if home_s < away_s:
            return "SH"
    else:
        if away_s > home_s:
            return "PP"
        if away_s < home_s:
            return "SH"
    return "EV"


def iter_schedule_games(start: date, end: date, allowed_seasons: set[int]) -> Iterator[dict[str, Any]]:
    """Walk weekly anchors to reduce duplicate schedule calls."""
    seen: set[int] = set()
    d = start
    while d <= end:
        url = NHL_SCHEDULE_URL.format(date=d.isoformat())
        try:
            payload = http_get_json(url)
        except Exception:
            d += timedelta(days=7)
            continue
        for week in payload.get("gameWeek", []) or []:
            for g in week.get("games", []) or []:
                gid = int(g["id"])
                if gid in seen:
                    continue
                if int(g.get("gameType", 0)) != GAME_TYPE_REGULAR:
                    continue
                season = int(g.get("season", 0))
                if season not in allowed_seasons:
                    continue
                seen.add(gid)
                yield g
        d += timedelta(days=7)
        time.sleep(0.05)


def parse_pbp_to_rows(
    pbp: dict[str, Any],
    game_meta: GameMeta,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (events_rows, shots_rows) from a play-by-play payload."""
    home = pbp.get("homeTeam") or {}
    away = pbp.get("awayTeam") or {}
    home_id = int(home.get("id"))
    away_id = int(away.get("id"))
    home_def = pbp.get("homeTeamDefendingSide")  # may appear on each play instead

    events: list[dict[str, Any]] = []
    shots: list[dict[str, Any]] = []

    away_score = 0
    home_score = 0

    plays = pbp.get("plays") or []
    for play in plays:
        sort_order = int(play.get("sortOrder", 0))
        etype = play.get("typeDescKey") or ""
        period = int((play.get("periodDescriptor") or {}).get("number", 0))
        clock = play.get("timeInPeriod")
        gs = period_clock_to_seconds(period, clock)

        det = play.get("details") or {}
        team_id = det.get("eventOwnerTeamId")
        team_id_int = int(team_id) if team_id is not None else None

        x = det.get("xCoord")
        y = det.get("yCoord")
        situation = play.get("situationCode")

        axs, hxs = _decode_scores_from_situation_or_details(play)
        if axs is not None and hxs is not None:
            away_score, home_score = axs, hxs

        play_home_def = play.get("homeTeamDefendingSide") or home_def

        events.append(
            {
                "game_id": game_meta.game_id,
                "event_id": int(play.get("eventId", sort_order)),
                "sort_order": sort_order,
                "period": period,
                "time_in_period": clock,
                "game_seconds": gs,
                "type_desc_key": etype,
                "team_id": team_id_int,
                "x_coord": float(x) if x is not None else None,
                "y_coord": float(y) if y is not None else None,
                "situation_code": situation,
            }
        )

        if etype not in SHOT_EVENT_TYPES:
            continue
        if team_id_int is None or x is None or y is None:
            continue

        shooter = det.get("shootingPlayerId") or det.get("scoringPlayerId")
        if shooter is None:
            continue
        shooter_id = int(shooter)
        goalie_id = det.get("goalieInNetId")
        goalie_id_int = int(goalie_id) if goalie_id is not None else None

        is_goal = 1 if etype == "goal" else 0
        opponent_id = away_id if team_id_int == home_id else home_id
        home_away = "HOME" if team_id_int == home_id else "AWAY"
        strength = _strength_label(situation, team_id_int, home_id)

        shots.append(
            {
                "game_id": game_meta.game_id,
                "event_id": int(play.get("eventId", sort_order)),
                "sort_order": sort_order,
                "season": game_meta.season,
                "game_date": game_meta.game_date.isoformat(),
                "period": period,
                "time_in_period": clock,
                "game_seconds": gs,
                "team_id": team_id_int,
                "opponent_id": opponent_id,
                "shooter_id": shooter_id,
                "goalie_id": goalie_id_int,
                "x_coord": float(x),
                "y_coord": float(y),
                "shot_type": (det.get("shotType") or "unknown").lower(),
                "event_type": etype,
                "situation_code": situation,
                "home_away": home_away,
                "is_goal": is_goal,
                "strength_state": strength,
                "away_score": away_score,
                "home_score": home_score,
                "_home_team_id": home_id,
                "_home_def": (play_home_def or "").lower() if play_home_def else None,
            }
        )

    return events, shots


def upsert_game_bundle(
    engine: Engine,
    game: GameMeta,
    events: list[dict[str, Any]],
    shots: list[dict[str, Any]],
) -> None:
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO games (game_id, season, game_date, game_type, home_team_id, away_team_id, home_abbr, away_abbr, home_defending_side)
                VALUES (:game_id, :season, :game_date, :game_type, :home_team_id, :away_team_id, :home_abbr, :away_abbr, :home_defending_side)
                ON CONFLICT(game_id) DO UPDATE SET
                    season=excluded.season,
                    game_date=excluded.game_date,
                    home_team_id=excluded.home_team_id,
                    away_team_id=excluded.away_team_id,
                    home_abbr=excluded.home_abbr,
                    away_abbr=excluded.away_abbr,
                    home_defending_side=excluded.home_defending_side
                """
            ),
            {
                "game_id": game.game_id,
                "season": game.season,
                "game_date": game.game_date.isoformat(),
                "game_type": GAME_TYPE_REGULAR,
                "home_team_id": game.home_team_id,
                "away_team_id": game.away_team_id,
                "home_abbr": game.home_abbr,
                "away_abbr": game.away_abbr,
                "home_defending_side": game.home_defending_side,
            },
        )
        conn.execute(text("DELETE FROM events WHERE game_id = :gid"), {"gid": game.game_id})
        conn.execute(text("DELETE FROM shots WHERE game_id = :gid"), {"gid": game.game_id})
        ev_stmt = text(
            """
            INSERT INTO events (game_id, event_id, sort_order, period, time_in_period, game_seconds, type_desc_key, team_id, x_coord, y_coord, situation_code)
            VALUES (:game_id, :event_id, :sort_order, :period, :time_in_period, :game_seconds, :type_desc_key, :team_id, :x_coord, :y_coord, :situation_code)
            """
        )
        for e in events:
            conn.execute(ev_stmt, e)
        if shots:
            # strip helper keys
            shot_params = []
            for s in shots:
                shot_params.append(
                    {
                        "game_id": s["game_id"],
                        "event_id": s["event_id"],
                        "sort_order": s["sort_order"],
                        "season": s["season"],
                        "game_date": s["game_date"],
                        "period": s["period"],
                        "time_in_period": s["time_in_period"],
                        "game_seconds": s["game_seconds"],
                        "team_id": s["team_id"],
                        "opponent_id": s["opponent_id"],
                        "shooter_id": s["shooter_id"],
                        "goalie_id": s["goalie_id"],
                        "x_coord": s["x_coord"],
                        "y_coord": s["y_coord"],
                        "shot_type": s["shot_type"],
                        "event_type": s["event_type"],
                        "situation_code": s["situation_code"],
                        "home_away": s["home_away"],
                        "is_goal": s["is_goal"],
                        "strength_state": s["strength_state"],
                        "away_score": s["away_score"],
                        "home_score": s["home_score"],
                    }
                )
            sh_stmt = text(
                """
                INSERT INTO shots (
                    game_id, event_id, sort_order, season, game_date, period, time_in_period, game_seconds,
                    team_id, opponent_id, shooter_id, goalie_id, x_coord, y_coord, shot_type, event_type,
                    situation_code, home_away, is_goal, strength_state, away_score, home_score
                ) VALUES (
                    :game_id, :event_id, :sort_order, :season, :game_date, :period, :time_in_period, :game_seconds,
                    :team_id, :opponent_id, :shooter_id, :goalie_id, :x_coord, :y_coord, :shot_type, :event_type,
                    :situation_code, :home_away, :is_goal, :strength_state, :away_score, :home_score
                )
                """
            )
            for row in shot_params:
                conn.execute(sh_stmt, row)


def ingest_season_window(
    engine: Engine,
    *,
    start: date,
    end: date,
    allowed_seasons: set[int],
    save_raw: bool,
    max_games: int | None,
) -> int:
    ensure_directories()
    raw_dir = DATA_RAW / "pbp"
    raw_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    for g in iter_schedule_games(start, end, allowed_seasons):
        gid = int(g["id"])
        # Prefer the NHL-reported local game date from the schedule payload; fall back to UTC.
        gdate = parse_game_date(g.get("gameDate")) or parse_game_date(g.get("startTimeUTC"))
        meta = GameMeta(
            game_id=gid,
            season=int(g["season"]),
            game_date=gdate or date.today(),
            home_team_id=int((g.get("homeTeam") or {}).get("id")),
            away_team_id=int((g.get("awayTeam") or {}).get("id")),
            home_abbr=(g.get("homeTeam") or {}).get("abbrev"),
            away_abbr=(g.get("awayTeam") or {}).get("abbrev"),
        )
        url = NHL_PBP_URL.format(game_id=gid)
        try:
            pbp = http_get_json(url)
        except Exception as e:  # noqa: BLE001
            logger.error("Skip game %s: %s", gid, e)
            continue

        if save_raw:
            (raw_dir / f"{gid}.json").write_text(json.dumps(pbp, indent=2), encoding="utf-8")

        meta.home_defending_side = pbp.get("homeTeamDefendingSide")
        pbp_date = parse_game_date(pbp.get("gameDate")) or parse_game_date(pbp.get("startTimeUTC"))
        if pbp_date:
            meta.game_date = pbp_date
        events, shots = parse_pbp_to_rows(pbp, meta)
        upsert_game_bundle(engine, meta, events, shots)
        processed += 1
        if processed % 50 == 0:
            logger.info("Ingested %s games (last %s)", processed, gid)
        if max_games and processed >= max_games:
            break
        time.sleep(0.07)

    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NHL schedule + play-by-play into SQL.")
    parser.add_argument("--start", default="2023-10-01", help="Schedule crawl start (YYYY-MM-DD)")
    parser.add_argument("--end", default="2026-04-20", help="Schedule crawl end (YYYY-MM-DD)")
    parser.add_argument(
        "--seasons",
        default="20232024,20242025,20252026",
        help="Comma-separated season ids to retain (e.g. 20232024,20242025,20252026)",
    )
    parser.add_argument("--no-raw", action="store_true", help="Do not write per-game raw JSON files")
    parser.add_argument("--max-games", type=int, default=None, help="Stop after N games (debug)")
    args = parser.parse_args()

    setup_logging()
    ensure_directories()

    allowed = {int(x.strip()) for x in args.seasons.split(",") if x.strip()}
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)

    engine = get_engine()
    n = ingest_season_window(
        engine,
        start=start,
        end=end,
        allowed_seasons=allowed,
        save_raw=not args.no_raw,
        max_games=args.max_games,
    )
    logger.info("Done. Ingested/updated %s games.", n)


if __name__ == "__main__":
    main()
