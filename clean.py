from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path("data")
RAW_FILE = DATA_DIR / "activities_raw.json"
CLEAN_FILE = DATA_DIR / "activities_clean.json"


def clean_raw_to_clean() -> List[Dict[str, Any]]:
    DATA_DIR.mkdir(exist_ok=True)

    if not RAW_FILE.exists():
        # nic do zrobienia
        with open(CLEAN_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return []

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)

    clean: List[Dict[str, Any]] = []
    for a in raw:
        clean.append(
            {
                "id": a.get("id"),
                "type": a.get("type"),
                "sport_type": a.get("sport_type"),
                "name": a.get("name"),
                "start_date_local": a.get("start_date_local") or a.get("start_date"),
                "timezone": a.get("timezone"),
                "distance_m": a.get("distance"),
                "moving_time_s": a.get("moving_time"),
                "elapsed_time_s": a.get("elapsed_time"),
                "total_elevation_gain_m": a.get("total_elevation_gain"),
                "average_heartrate": a.get("average_heartrate"),
                "max_heartrate": a.get("max_heartrate"),
            }
        )

    with open(CLEAN_FILE, "w", encoding="utf-8") as f:
        json.dump(clean, f, ensure_ascii=False, indent=2)

    return clean


if __name__ == "__main__":
    out = clean_raw_to_clean()
    print("Cleaned:", len(out))