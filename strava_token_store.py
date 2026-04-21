from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TOKENS_FILE = DATA_DIR / "strava_tokens.json"


def _load() -> Dict[str, Any]:
    if not TOKENS_FILE.exists():
        return {"active_athlete_id": None, "athletes": {}}
    try:
        payload = json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {"active_athlete_id": None, "athletes": {}}
        payload.setdefault("active_athlete_id", None)
        payload.setdefault("athletes", {})
        if not isinstance(payload["athletes"], dict):
            payload["athletes"] = {}
        return payload
    except Exception:
        return {"active_athlete_id": None, "athletes": {}}


def _save(payload: Dict[str, Any]) -> None:
    TOKENS_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def set_active_athlete(athlete_id: str | int) -> None:
    store = _load()
    store["active_athlete_id"] = str(athlete_id)
    _save(store)


def upsert_tokens(
    athlete_id: str | int,
    *,
    access_token: str,
    refresh_token: str,
    expires_at: int,
) -> None:
    store = _load()
    aid = str(athlete_id)
    athletes = store.setdefault("athletes", {})
    athletes[aid] = {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_at": int(expires_at),
    }
    store["active_athlete_id"] = aid
    _save(store)


def get_active_tokens() -> Optional[Dict[str, Any]]:
    store = _load()
    aid = store.get("active_athlete_id")
    if not aid:
        return None
    athletes = store.get("athletes") or {}
    data = athletes.get(str(aid))
    return data if isinstance(data, dict) else None

