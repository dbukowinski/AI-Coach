from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

DATA_DIR = Path("data")
CLEAN_FILE = DATA_DIR / "activities_clean.json"
SUMMARY_JSON = DATA_DIR / "summary_7d.json"
SUMMARY_MD = DATA_DIR / "summary_7d.md"


def _parse_iso(dt_str: str) -> datetime | None:
    if not dt_str:
        return None
    # handle "...Z"
    if dt_str.endswith("Z"):
        dt_str = dt_str.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(dt_str)
    except Exception:
        return None


def generate_report(days: int = 7) -> Dict[str, Any]:
    DATA_DIR.mkdir(exist_ok=True)

    if not CLEAN_FILE.exists():
        summary = {"days": days, "count": 0, "by_type": {}, "totals": {}}
        _write(summary)
        return summary

    with open(CLEAN_FILE, "r", encoding="utf-8") as f:
        acts: List[Dict[str, Any]] = json.load(f)

    cutoff = datetime.now(timezone.utc).timestamp() - days * 24 * 60 * 60

    filtered = []
    for a in acts:
        dt = _parse_iso(a.get("start_date_local") or a.get("start_date"))
        if dt and dt.timestamp() >= cutoff:
            filtered.append(a)

    totals = {"distance_m": 0.0, "moving_time_s": 0, "elevation_m": 0.0, "hr_count": 0}
    by_type: Dict[str, int] = {}

    for a in filtered:
        t = a.get("type") or "Unknown"
        by_type[t] = by_type.get(t, 0) + 1

        totals["distance_m"] += float(a.get("distance_m") or 0.0)
        totals["moving_time_s"] += int(a.get("moving_time_s") or 0)
        totals["elevation_m"] += float(a.get("total_elevation_gain_m") or 0.0)
        if a.get("average_heartrate") is not None:
            totals["hr_count"] += 1

    summary = {
        "days": days,
        "count": len(filtered),
        "by_type": by_type,
        "totals": totals,
        "hr_coverage_pct": (totals["hr_count"] / len(filtered) * 100.0) if filtered else 0.0,
    }

    _write(summary)
    return summary


def _write(summary: Dict[str, Any]) -> None:
    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # md dla człowieka
    lines = []
    lines.append(f"SUMMARY LAST {summary['days']} DAYS")
    lines.append("")
    lines.append(f"Activities: {summary['count']}")
    lines.append(f"HR coverage: {summary['hr_coverage_pct']:.1f}%")
    lines.append("")
    lines.append("By type:")
    for k, v in sorted(summary["by_type"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"- {k}: {v}")
    lines.append("")
    dist_km = summary["totals"]["distance_m"] / 1000.0
    time_h = summary["totals"]["moving_time_s"] / 3600.0
    elev = summary["totals"]["elevation_m"]
    lines.append("Totals:")
    lines.append(f"- Distance: {dist_km:.2f} km")
    lines.append(f"- Moving time: {time_h:.2f} h")
    lines.append(f"- Elevation gain: {elev:.0f} m")

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    s = generate_report(days=7)
    print("Report generated. Activities:", s["count"])