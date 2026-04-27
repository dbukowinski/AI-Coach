from __future__ import annotations

from langchain_core.tools import tool

HR_MAX = 178


def _get_hr_zone(avg_hr: float | None, hr_max: int = HR_MAX) -> str:
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


def _get_intensity_factor(zone: str) -> float:
    return {
        "below_z1": 0.8,
        "z1": 1.0,
        "z2": 1.2,
        "z3": 1.5,
        "z4": 2.0,
        "z5": 2.5,
        "unknown": 1.1,
    }.get(zone, 1.1)


@tool
def compute_weekly_load(activities: list[dict]) -> dict:
    """Oblicza tygodniowe obciążenie treningowe z listy aktywności Strava."""
    zone_counts = {k: 0 for k in ("below_z1", "z1", "z2", "z3", "z4", "z5", "unknown")}
    total_load = 0.0
    details = []

    for a in activities:
        avg_hr = a.get("average_heartrate")
        moving_time_s = int(a.get("moving_time_s") or 0)
        moving_time_h = moving_time_s / 3600.0
        zone = _get_hr_zone(avg_hr)
        load_score = round(moving_time_h * _get_intensity_factor(zone), 3)

        details.append({
            "id": a.get("id"),
            "name": a.get("name"),
            "type": a.get("type"),
            "start_date_local": a.get("start_date_local"),
            "average_heartrate": avg_hr,
            "hr_zone": zone,
            "moving_time_h": round(moving_time_h, 3),
            "load_score": load_score,
        })
        zone_counts[zone] += 1
        total_load += load_score

    return {
        "activity_count": len(activities),
        "weekly_load": round(total_load, 3),
        "zone_counts": zone_counts,
        "activities": details,
    }


@tool
def detect_training_flags(
    weekly_load: float,
    avg_4w_load: float,
    high_intensity_count: int,
    z1_count: int,
    activity_count: int,
) -> list[str]:
    """Wykrywa flagi ryzyka treningowego (przetrenowanie, brak recovery)."""
    flags: list[str] = []

    if avg_4w_load > 0 and weekly_load > avg_4w_load * 1.30:
        flags.append("current_week_load_above_130pct_of_4w_avg")

    if high_intensity_count >= 3:
        flags.append("high_intensity_frequency_high")

    if activity_count >= 5 and z1_count == 0:
        flags.append("low_recovery_volume")

    return flags
