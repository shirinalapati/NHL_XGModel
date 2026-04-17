"""Shared helpers: HTTP, database, time parsing, logging."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime
from typing import Any, Iterable

import requests
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from src.config import NHL_PEOPLE_URL, NHL_PLAYER_LANDING_URL, ensure_directories

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NHL-xG-research/1.0; +https://github.com/)",
    "Accept": "application/json",
}

logger = logging.getLogger("nhl_xg")


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def get_engine(url: str | None = None) -> Engine:
    from src.config import database_url

    ensure_directories()
    return create_engine(url or database_url(), future=True)


def http_get_json(url: str, *, retries: int = 3, backoff: float = 0.4) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=60, headers=DEFAULT_HEADERS)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            last_err = e
            sleep = backoff * (2**attempt)
            logger.warning("GET %s failed (%s); retry in %.1fs", url, e, sleep)
            time.sleep(sleep)
    raise RuntimeError(f"Failed to fetch {url}") from last_err


def parse_game_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def period_clock_to_seconds(period: int, time_in_period: str | None) -> int | None:
    """
    Convert period + clock MM:SS (counting down) to absolute game seconds (0-based).
    Regulation assumed 3x20min; OT periods add 20 min each (approximation for features).
    """
    if not time_in_period or period < 1:
        return None
    parts = time_in_period.strip().split(":")
    if len(parts) != 2:
        return None
    try:
        mm, ss = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    elapsed_in_period = 20 * 60 - (mm * 60 + ss)
    if period <= 3:
        return (period - 1) * 20 * 60 + elapsed_in_period
    # OT / extras: add full regulation
    return 3 * 20 * 60 + (period - 4) * 20 * 60 + elapsed_in_period


def _name_from_landing(payload: dict[str, Any]) -> str:
    fn = payload.get("firstName") or {}
    ln = payload.get("lastName") or {}
    if isinstance(fn, dict):
        first = (fn.get("default") or "").strip()
    else:
        first = str(fn).strip()
    if isinstance(ln, dict):
        last = (ln.get("default") or "").strip()
    else:
        last = str(ln).strip()
    return f"{first} {last}".strip()


def _fetch_one_landing(pid: int) -> tuple[int, str]:
    url = NHL_PLAYER_LANDING_URL.format(player_id=pid)
    try:
        data = http_get_json(url)
        name = _name_from_landing(data)
        return pid, (name or str(pid))
    except Exception:  # noqa: BLE001
        return pid, str(pid)


def fetch_player_names(player_ids: Iterable[int], *, batch_size: int = 50) -> dict[int, str]:
    """
    Resolve NHL player IDs to display names.

    Uses `api-web.nhle.com` player landing (same family as schedule/PBP). The legacy
    statsapi batch endpoint is kept only as a fallback when landing fails for an id.
    """
    ids = sorted({int(x) for x in player_ids if x})
    out: dict[int, str] = {}

    workers = min(4, max(1, len(ids)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_fetch_one_landing, pid) for pid in ids]
        for fut in as_completed(futures):
            pid, name = fut.result()
            out[pid] = name
    time.sleep(0.2)

    # Fill any gaps via statsapi batch (best-effort; may be unavailable).
    missing = [pid for pid in ids if out.get(pid) == str(pid)]
    if not missing:
        return out
    for i in range(0, len(missing), batch_size):
        chunk = missing[i : i + batch_size]
        url = NHL_PEOPLE_URL + "?personIds=" + ",".join(str(x) for x in chunk)
        try:
            data = http_get_json(url)
            for p in data.get("people", []):
                pid = int(p["id"])
                first = (p.get("firstName") or "").strip()
                last = (p.get("lastName") or "").strip()
                nm = f"{first} {last}".strip()
                if nm:
                    out[pid] = nm
        except Exception:  # noqa: BLE001
            break
        time.sleep(0.12)
    return out


def run_sql_file(engine: Engine, path: str) -> None:
    from pathlib import Path

    sql_text = Path(path).read_text(encoding="utf-8")
    # Execute as a script (SQLite supports multiple statements via executescript in raw)
    with engine.begin() as conn:
        conn.exec_driver_sql(sql_text)
