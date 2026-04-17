"""
Initialize / reset database objects and build SQL-derived shot context tables.

Typical usage after ingestion:
    python -m src.clean --schema --context
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine

from src.config import SQL_DIR, ensure_directories
from src.utils import fetch_player_names, get_engine, setup_logging

logger = logging.getLogger("nhl_xg.clean")


def init_schema(engine: Engine) -> None:
    """Apply base DDL from `sql/schema.sql`."""
    schema_path = Path(SQL_DIR) / "schema.sql"
    sql = schema_path.read_text(encoding="utf-8")
    if engine.dialect.name == "sqlite":
        raw = engine.raw_connection()
        try:
            raw.executescript(sql)
        finally:
            raw.close()
    else:
        # PostgreSQL / others: strip SQLite PRAGMA and execute statement-by-statement.
        lines = [ln for ln in sql.splitlines() if not ln.strip().upper().startswith("PRAGMA")]
        cleaned = "\n".join(lines)
        with engine.begin() as conn:
            for stmt in cleaned.split(";"):
                st = stmt.strip()
                if st:
                    conn.execute(text(st))
    logger.info("Applied schema from %s", schema_path)


def rebuild_shot_context(engine: Engine) -> None:
    """Rebuild `shots_with_context` using window-function SQL."""
    path = Path(SQL_DIR) / "shot_context_features.sql"
    sql = path.read_text(encoding="utf-8")
    if engine.dialect.name == "sqlite":
        raw = engine.raw_connection()
        try:
            raw.executescript(sql)
        finally:
            raw.close()
    else:
        with engine.begin() as conn:
            for stmt in sql.split(";"):
                st = stmt.strip()
                if st:
                    conn.execute(text(st))
    logger.info("Rebuilt shot context table from %s", path)


def refresh_player_names(engine: Engine) -> None:
    """Populate `players` table for all shooter_ids seen in `shots`."""
    with engine.connect() as conn:
        ids = [r[0] for r in conn.execute(text("SELECT DISTINCT shooter_id FROM shots")).fetchall()]
    if not ids:
        logger.warning("No shooter ids found; skipping player refresh.")
        return
    logger.info("Resolving %s unique player ids (batched statsapi calls)...", len(ids))
    names = fetch_player_names(ids)
    rows = [{"player_id": pid, "full_name": names.get(pid, str(pid))} for pid in ids]
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM players"))
        ins = text("INSERT INTO players (player_id, full_name) VALUES (:player_id, :full_name)")
        for row in rows:
            conn.execute(ins, row)
    logger.info("Player dimension updated (%s rows).", len(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description="DB maintenance: schema + SQL context features.")
    parser.add_argument("--schema", action="store_true", help="Apply sql/schema.sql")
    parser.add_argument("--context", action="store_true", help="Rebuild shots_with_context SQL table")
    parser.add_argument("--players", action="store_true", help="Refresh players table from shooter ids")
    args = parser.parse_args()

    setup_logging()
    ensure_directories()
    engine = get_engine()

    if not (args.schema or args.context or args.players):
        parser.error("Select at least one of --schema --context --players")

    if args.schema:
        init_schema(engine)
    if args.context:
        rebuild_shot_context(engine)
    if args.players:
        refresh_player_names(engine)


if __name__ == "__main__":
    main()
