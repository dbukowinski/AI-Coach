from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

CLEAN_FILE = DATA_DIR / "activities_clean.json"
WEEKLY_SUMMARY_FILE = DATA_DIR / "weekly_summary.json"
FOUR_WEEK_SUMMARY_FILE = DATA_DIR / "four_week_summary.json"
FLAGS_FILE = DATA_DIR / "flags.json"
HR_ZONES_SUMMARY_FILE = DATA_DIR / "hr_zones_summary.json"

HR_MAX = 178


def _parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
    if not dt_str:
        return None
    if dt_str.endswith("Z"):
        dt_str = dt_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def _load_clean() -> List[Dict[str, Any]]:
    if not CLEAN_FILE.exists():
        return []
    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def get_hr_zone(avg_hr: Optional[float], hr_max: int = HR_MAX) -> str:
    if avg_hr is None:
        return "unknown"

    pct = avg_hr / hr_max

    if pct < 0.50:
        return "below_z1"
    if pct < 0.60:
        return "z1"
    if pct < 0.70:
        return "z2"
    if pct < 0.80:
        return "z3"
    if pct < 0.90:
        return "z4"
    return "z5"


def get_intensity_factor(zone: str) -> float:
    mapping = {
        "below_z1": 0.8,
        "z1": 1.0,
        "z2": 1.2,
        "z3": 1.5,
        "z4": 2.0,
        "z5": 2.5,
        "unknown": 1.1,
    }
    return mapping.get(zone, 1.1)


def compute_activity_load(activity: Dict[str, Any], hr_max: int = HR_MAX) -> Dict[str, Any]:
    avg_hr = activity.get("average_heartrate")
    moving_time_s = int(activity.get("moving_time_s") or 0)
    moving_time_h = moving_time_s / 3600.0

    zone = get_hr_zone(avg_hr, hr_max=hr_max)
    intensity_factor = get_intensity_factor(zone)
    load_score = round(moving_time_h * intensity_factor, 3)

    return {
        "id": activity.get("id"),
        "name": activity.get("name"),
        "type": activity.get("type"),
        "start_date_local": activity.get("start_date_local"),
        "average_heartrate": avg_hr,
        "hr_zone": zone,
        "moving_time_h": round(moving_time_h, 3),
        "load_score": load_score,
    }


def summarize_last_days(days: int = 7, hr_max: int = HR_MAX) -> Dict[str, Any]:
    acts = _load_clean()
    cutoff_ts = datetime.now(timezone.utc).timestamp() - days * 24 * 60 * 60

    filtered: List[Dict[str, Any]] = []
    for a in acts:
        dt = _parse_iso(a.get("start_date_local"))
        if dt and dt.timestamp() >= cutoff_ts:
            filtered.append(a)

    zone_counts = {
        "below_z1": 0,
        "z1": 0,
        "z2": 0,
        "z3": 0,
        "z4": 0,
        "z5": 0,
        "unknown": 0,
    }

    total_load = 0.0
    details = []

    for a in filtered:
        item = compute_activity_load(a, hr_max=hr_max)
        details.append(item)
        zone_counts[item["hr_zone"]] += 1
        total_load += item["load_score"]

    summary = {
        "days": days,
        "hr_max": hr_max,
        "activity_count": len(filtered),
        "weekly_load": round(total_load, 3),
        "zone_counts": zone_counts,
        "activities": details,
    }

    _save_json(WEEKLY_SUMMARY_FILE, summary)
    _save_json(HR_ZONES_SUMMARY_FILE, {"days": days, "hr_max": hr_max, "zone_counts": zone_counts})
    return summary


def summarize_last_4_weeks(hr_max: int = HR_MAX) -> Dict[str, Any]:
    acts = _load_clean()
    now_ts = datetime.now(timezone.utc).timestamp()

    weekly_buckets = []

    for week_idx in range(4):
        end_ts = now_ts - (week_idx * 7 * 24 * 60 * 60)
        start_ts = end_ts - (7 * 24 * 60 * 60)

        week_acts = []
        for a in acts:
            dt = _parse_iso(a.get("start_date_local"))
            if dt and start_ts <= dt.timestamp() < end_ts:
                week_acts.append(a)

        zone_counts = {
            "below_z1": 0,
            "z1": 0,
            "z2": 0,
            "z3": 0,
            "z4": 0,
            "z5": 0,
            "unknown": 0,
        }

        weekly_load = 0.0

        for a in week_acts:
            item = compute_activity_load(a, hr_max=hr_max)
            zone_counts[item["hr_zone"]] += 1
            weekly_load += item["load_score"]

        weekly_buckets.append(
            {
                "week_index": week_idx,
                "activity_count": len(week_acts),
                "weekly_load": round(weekly_load, 3),
                "zone_counts": zone_counts,
            }
        )

    current_week_load = weekly_buckets[0]["weekly_load"] if weekly_buckets else 0.0
    avg_4w_load = round(
        sum(w["weekly_load"] for w in weekly_buckets) / len(weekly_buckets),
        3
    ) if weekly_buckets else 0.0

    payload = {
        "hr_max": hr_max,
        "weeks": weekly_buckets,
        "current_week_load": current_week_load,
        "avg_4w_load": avg_4w_load,
    }

    _save_json(FOUR_WEEK_SUMMARY_FILE, payload)
    return payload


def detect_flags(weekly_summary: Dict[str, Any], four_week_summary: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []

    current_week_load = weekly_summary.get("weekly_load", 0.0)
    avg_4w_load = four_week_summary.get("avg_4w_load", 0.0)
    zone_counts = weekly_summary.get("zone_counts", {})

    z4_count = zone_counts.get("z4", 0)
    z5_count = zone_counts.get("z5", 0)
    high_intensity_count = z4_count + z5_count

    if avg_4w_load > 0 and current_week_load > avg_4w_load * 1.30:
        flags.append("current_week_load_above_130pct_of_4w_avg")

    if high_intensity_count >= 3:
        flags.append("high_intensity_frequency_high")

    if weekly_summary.get("activity_count", 0) >= 5 and zone_counts.get("z1", 0) == 0:
        flags.append("low_recovery_volume")

    payload = {
        "weekly_load": current_week_load,
        "avg_4w_load": avg_4w_load,
        "flags": flags,
    }

    _save_json(FLAGS_FILE, payload)
    return payload


def run_analysis(days: int = 7, hr_max: int = HR_MAX) -> Dict[str, Any]:
    weekly = summarize_last_days(days=days, hr_max=hr_max)
    four_weeks = summarize_last_4_weeks(hr_max=hr_max)
    flags_payload = detect_flags(weekly, four_weeks)

    hr_zones_summary = {
        "days": days,
        "hr_max": hr_max,
        "zone_counts": weekly.get("zone_counts", {}),
    }

    return {
        "hr_zones_summary": hr_zones_summary,
        "weekly_summary": weekly,
        "four_week_summary": four_weeks,
        "flags": flags_payload.get("flags", []),
        "flags_payload": flags_payload,
    }


if __name__ == "__main__":
    result = run_analysis(days=7, hr_max=178)
    print("Analysis done.")
    print("Weekly load:", result["weekly_summary"]["weekly_load"])
    print("4w avg load:", result["four_week_summary"]["avg_4w_load"])
    print("Flags:", result["flags"])