from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Set

import requests
from dotenv import find_dotenv, load_dotenv, set_key


# =========================
# Config / Paths
# =========================

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEEN_IDS_FILE = DATA_DIR / "seen_ids.json"
RAW_FILE = DATA_DIR / "activities_raw.json"

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)


def _hydrate_strava_env_from_streamlit_secrets() -> None:
    """
    Streamlit Community Cloud nie ma pliku .env — sekrety ustawia się w UI (Secrets).
    Kopiuje je do os.environ przed odczytem zmiennych Stravy.
    Obsługa: płaskie klucze STRAVA_* albo sekcja [strava] w TOML.
    """
    try:
        import streamlit as st  # type: ignore
    except ImportError:
        return
    try:
        sec = getattr(st, "secrets", None)
        if sec is None:
            return
    except Exception:
        return

    def _set_from_flat(key: str) -> None:
        try:
            if key in sec and sec[key] not in (None, ""):
                os.environ[key] = str(sec[key]).strip()
        except Exception:
            pass

    for k in (
        "STRAVA_CLIENT_ID",
        "STRAVA_CLIENT_SECRET",
        "STRAVA_ACCESS_TOKEN",
        "STRAVA_REFRESH_TOKEN",
        "STRAVA_EXPIRES_AT",
    ):
        _set_from_flat(k)

    try:
        if "strava" in sec:
            sub = sec["strava"]
            mapping = (
                ("STRAVA_CLIENT_ID", "client_id"),
                ("STRAVA_CLIENT_SECRET", "client_secret"),
                ("STRAVA_ACCESS_TOKEN", "access_token"),
                ("STRAVA_REFRESH_TOKEN", "refresh_token"),
                ("STRAVA_EXPIRES_AT", "expires_at"),
            )
            for env_key, sub_key in mapping:
                try:
                    val = sub.get(sub_key) if hasattr(sub, "get") else sub[sub_key]
                except Exception:
                    val = None
                if val not in (None, ""):
                    os.environ[env_key] = str(val).strip()
    except Exception:
        pass


_hydrate_strava_env_from_streamlit_secrets()

CLIENT_ID = os.getenv("STRAVA_CLIENT_ID")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET")

ACCESS_TOKEN = os.getenv("STRAVA_ACCESS_TOKEN")
REFRESH_TOKEN = os.getenv("STRAVA_REFRESH_TOKEN")
_raw_exp = os.getenv("STRAVA_EXPIRES_AT", "0") or "0"
try:
    EXPIRES_AT = int(_raw_exp)
except ValueError:
    EXPIRES_AT = 0


def _hydrate_tokens_from_local_store_if_missing() -> None:
    """
    Dla Streamlit: po OAuth zapisujemy refresh token do `data/strava_tokens.json` (gitignored).
    Jeśli env nie ma tokenów (typowe na Cloud), wczytaj je z tego pliku.
    """
    global ACCESS_TOKEN, REFRESH_TOKEN, EXPIRES_AT
    if REFRESH_TOKEN and ACCESS_TOKEN:
        return
    try:
        from strava_token_store import get_active_tokens

        t = get_active_tokens()
        if not t:
            return
        if not ACCESS_TOKEN:
            ACCESS_TOKEN = str(t.get("access_token") or "").strip() or None
        if not REFRESH_TOKEN:
            REFRESH_TOKEN = str(t.get("refresh_token") or "").strip() or None
        if not EXPIRES_AT:
            try:
                EXPIRES_AT = int(t.get("expires_at") or 0)
            except Exception:
                EXPIRES_AT = 0
    except Exception:
        return


_hydrate_tokens_from_local_store_if_missing()

TOKEN_URL = "https://www.strava.com/oauth/token"
ACTIVITIES_URL = "https://www.strava.com/api/v3/athlete/activities"
ACTIVITY_DETAIL_URL = "https://www.strava.com/api/v3/activities/{id}"


# =========================
# Auth / Token management
# =========================

def refresh_tokens() -> None:
    """Refresh Strava access token using refresh_token and persist to .env."""
    global ACCESS_TOKEN, REFRESH_TOKEN, EXPIRES_AT

    # Streamlit: jeśli tokeny pojawiły się po imporcie modułu, dociągnij je ze store.
    _hydrate_tokens_from_local_store_if_missing()

    if not (CLIENT_ID and CLIENT_SECRET and REFRESH_TOKEN):
        raise RuntimeError(
            "Brakuje STRAVA_CLIENT_ID / STRAVA_CLIENT_SECRET / STRAVA_REFRESH_TOKEN. "
            "Lokalnie: ustaw w pliku .env. Streamlit Cloud: Settings → Secrets — "
            "dodaj te same klucze (np. STRAVA_CLIENT_ID = \"...\") albo sekcję [strava] "
            "z client_id, client_secret, refresh_token."
        )

    data = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": REFRESH_TOKEN,
    }
    r = requests.post(TOKEN_URL, data=data, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"Refresh failed: {r.status_code} {r.text}")

    payload = r.json()
    ACCESS_TOKEN = payload["access_token"]
    REFRESH_TOKEN = payload["refresh_token"]
    EXPIRES_AT = int(payload["expires_at"])

    # Persist updated tokens (lokalnie w .env; na Streamlit Cloud często brak pliku — tylko RAM)
    if dotenv_path and str(dotenv_path).strip() and Path(dotenv_path).is_file():
        try:
            set_key(dotenv_path, "STRAVA_ACCESS_TOKEN", ACCESS_TOKEN)
            set_key(dotenv_path, "STRAVA_REFRESH_TOKEN", REFRESH_TOKEN)
            set_key(dotenv_path, "STRAVA_EXPIRES_AT", str(EXPIRES_AT))
        except OSError:
            pass

    # Persist do lokalnego store (Streamlit UX)
    try:
        from strava_token_store import upsert_tokens

        # Jeśli nie znamy athlete_id, zapisujemy pod "default".
        # (Dokładne athlete_id ustawiamy podczas OAuth exchange w UI.)
        upsert_tokens(
            "default",
            access_token=str(ACCESS_TOKEN or ""),
            refresh_token=str(REFRESH_TOKEN or ""),
            expires_at=int(EXPIRES_AT),
        )
    except Exception:
        pass


def ensure_valid_token(buffer_seconds: int = 60) -> None:
    """Ensure access token is present and not expiring soon."""
    _hydrate_tokens_from_local_store_if_missing()
    if (not ACCESS_TOKEN) or time.time() >= (EXPIRES_AT - buffer_seconds):
        refresh_tokens()


def _auth_headers() -> Dict[str, str]:
    ensure_valid_token()
    return {"Authorization": f"Bearer {ACCESS_TOKEN}"}


# =========================
# Fetching from Strava
# =========================

def fetch_last_days(days: int = 7, per_page: int = 50) -> List[Dict[str, Any]]:
    """Fetch activity list for last N days (no details)."""
    after = int(time.time() - days * 24 * 60 * 60)

    params = {"after": after, "per_page": per_page, "page": 1}
    acts: List[Dict[str, Any]] = []

    while True:
        r = requests.get(ACTIVITIES_URL, headers=_auth_headers(), params=params, timeout=30)

        if r.status_code == 401:
            # one retry after refresh (token rotation / expiry edge)
            refresh_tokens()
            r = requests.get(ACTIVITIES_URL, headers=_auth_headers(), params=params, timeout=30)

        if r.status_code != 200:
            raise RuntimeError(f"Fetch failed: {r.status_code} {r.text}")

        batch = r.json()
        if not batch:
            break

        acts.extend(batch)

        if len(batch) < per_page:
            break

        params["page"] += 1

    return acts


def get_activity_detail(activity_id: int) -> Dict[str, Any]:
    """Fetch full detail for a single activity."""
    url = ACTIVITY_DETAIL_URL.format(id=activity_id)
    r = requests.get(url, headers=_auth_headers(), timeout=30)

    if r.status_code == 401:
        refresh_tokens()
        r = requests.get(url, headers=_auth_headers(), timeout=30)

    if r.status_code != 200:
        raise RuntimeError(f"Detail fetch failed for {activity_id}: {r.status_code} {r.text}")

    return r.json()


# =========================
# Local storage / dedupe
# =========================

def load_seen_ids() -> Set[int]:
    if SEEN_IDS_FILE.exists():
        with open(SEEN_IDS_FILE, "r", encoding="utf-8") as f:
            return set(int(x) for x in json.load(f))
    return set()


def save_seen_ids(seen_ids: Set[int]) -> None:
    with open(SEEN_IDS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(seen_ids)), f, indent=2)


def append_raw_activities(new_activities: List[Dict[str, Any]]) -> None:
    """Append new raw activity detail payloads to RAW_FILE."""
    if RAW_FILE.exists():
        with open(RAW_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
    else:
        existing = []

    existing.extend(new_activities)

    with open(RAW_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def sync_last_days(days: int = 7, per_page: int = 50) -> Dict[str, int]:
    """
    High-level sync:
    - fetch list last N days
    - dedupe with seen_ids
    - fetch details for new ids
    - append to raw
    - update seen_ids
    """
    acts = fetch_last_days(days=days, per_page=per_page)
    ids = [int(a["id"]) for a in acts if a.get("id") is not None]

    seen = load_seen_ids()
    new_ids = [aid for aid in ids if aid not in seen]

    details: List[Dict[str, Any]] = []
    for aid in new_ids:
        details.append(get_activity_detail(aid))

    if details:
        append_raw_activities(details)

    for aid in new_ids:
        seen.add(aid)
    save_seen_ids(seen)

    return {"fetched": len(ids), "new": len(new_ids), "saved": len(details)}


# =========================
# Manual run (optional)
# =========================

if __name__ == "__main__":
    result = sync_last_days(days=7)
    print("OK.", result)