from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

WEEKLY_SUMMARY_FILE = DATA_DIR / "weekly_summary.json"
FOUR_WEEK_SUMMARY_FILE = DATA_DIR / "four_week_summary.json"
FLAGS_FILE = DATA_DIR / "flags.json"
PLAN_CURRENT_FILE = DATA_DIR / "training_plan_current.json"
PLAN_HISTORY_FILE = DATA_DIR / "training_plan_history.json"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def _save_json(path: Path, payload: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _append_plan_history(plan: Dict[str, Any]) -> None:
    if PLAN_HISTORY_FILE.exists():
        with open(PLAN_HISTORY_FILE, "r", encoding="utf-8") as f:
            existing = json.load(f)
            if not isinstance(existing, list):
                existing = []
    else:
        existing = []

    existing.append(plan)
    _save_json(PLAN_HISTORY_FILE, existing)


def _has_flag(flags: Dict[str, Any], flag_name: str) -> bool:
    return flag_name in flags.get("flags", [])


def _choose_plan_template(
    weekly_summary: Dict[str, Any],
    four_week_summary: Dict[str, Any],
    flags: Dict[str, Any],
) -> str:
    current_week_load = weekly_summary.get("weekly_load", 0.0)
    avg_4w_load = four_week_summary.get("avg_4w_load", 0.0)

    if _has_flag(flags, "current_week_load_above_130pct_of_4w_avg"):
        return "recovery"

    if _has_flag(flags, "high_intensity_frequency_high"):
        return "recovery"

    if _has_flag(flags, "low_recovery_volume"):
        return "balanced"

    if avg_4w_load > 0 and current_week_load < avg_4w_load * 0.8:
        return "build"

    return "balanced"


def _build_day(
    date_str: str,
    category: str,
    session_type: str,
    duration_min: int,
    notes: str,
    intensity: str,
    slot: Optional[str] = None,
) -> Dict[str, Any]:
    if duration_min == 0:
        duration_text = "0 min"
    else:
        duration_text = f"{duration_min} min"

    title = session_type
    if slot:
        title = f"{session_type} ({slot})"

    return {
        "date": date_str,
        "category": category,
        "session_type": session_type,
        "title": title,
        "duration_min": duration_min,
        "duration": duration_text,
        "intensity": intensity,
        "notes": notes,
        "slot": slot,
    }


def expand_sessions_to_multi_per_day(
    sessions: List[Dict[str, Any]],
    max_sessions_per_day: int,
) -> List[Dict[str, Any]]:
    """
    Rozbija dni z jedną sesją na 2 lub 3 sesje dziennie przy zachowaniu zbilansowanego RPE:
    - krótsze bloki, głównie easy/moderate, max jedna mocna jednostka na dzień.
    """
    if max_sessions_per_day < 2:
        return sessions

    from collections import defaultdict
    by_date: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in sessions:
        by_date[s.get("date", "")].append(s)

    result: List[Dict[str, Any]] = []
    for date_str in sorted(by_date.keys()):
        day_sessions = by_date[date_str]
        if len(day_sessions) >= max_sessions_per_day:
            result.extend(day_sessions)
            continue

        single = day_sessions[0]
        intensity = str(single.get("intensity", "")).lower()
        is_rest = intensity == "rest" or str(single.get("session_type", "")).lower() == "rest day"
        total_min = int(single.get("duration_min") or 0)

        if is_rest or total_min == 0:
            result.append(single)
            continue

        # Jedna aktywna sesja – rozbij na 2 lub 3 przy zbilansowanym RPE
        is_hard = intensity in ("hard", "long", "moderate")
        category = single.get("category", "run")
        base_notes = single.get("notes", "")

        if max_sessions_per_day == 2:
            # 2 sesje: np. rano + wieczór; jeśli była mocna – jedna główna, druga easy
            if is_hard:
                main_min = max(25, total_min // 2)
                easy_min = total_min - main_min
                result.append(_build_day(
                    date_str, category, single.get("session_type", "Run"),
                    main_min, base_notes, intensity, "rano",
                ))
                result.append(_build_day(
                    date_str, "run", "Easy run",
                    easy_min, "Easy effort, balanced RPE for the day.", "easy", "wieczór",
                ))
            else:
                a, b = total_min // 2, total_min - total_min // 2
                result.append(_build_day(
                    date_str, category, single.get("session_type", "Run"),
                    a, base_notes, "easy", "rano",
                ))
                result.append(_build_day(
                    date_str, category, "Easy / mobility",
                    b, "Light session, balanced daily load.", "easy", "wieczór",
                ))

        else:
            # 3 sesje: rano / w ciągu dnia / wieczór; jedna ewentualnie mocniejsza, reszta easy
            third = total_min // 3
            remainder = total_min - 2 * third
            a, b, c = third, third, remainder
            if a < 15:
                a, b, c = max(15, total_min - 30), 15, 15
            if is_hard:
                # jedna główna (środek dnia), dwie easy
                main_min = min(40, total_min - 30)
                easy_min = (total_min - main_min) // 2
                result.append(_build_day(date_str, "run", "Easy run", easy_min, "Warm-up / easy.", "easy", "rano"))
                result.append(_build_day(
                    date_str, category, single.get("session_type", "Run"),
                    main_min, base_notes, intensity, "środek dnia",
                ))
                result.append(_build_day(date_str, "run", "Easy run", total_min - main_min - easy_min, "Cool-down, easy.", "easy", "wieczór"))
            else:
                result.append(_build_day(date_str, category, single.get("session_type", "Run"), a, base_notes, "easy", "rano"))
                result.append(_build_day(date_str, "mobility", "Mobility / core", b, "Short support, balanced RPE.", "easy", "w ciągu dnia"))
                result.append(_build_day(date_str, "run", "Easy run", c, "Light finish.", "easy", "wieczór"))

    return result


def _generate_recovery_week(start_date: datetime) -> List[Dict[str, Any]]:
    return [
        _build_day((start_date + timedelta(days=0)).date().isoformat(), "rest", "Rest day", 0, "Full recovery or easy walk.", "rest"),
        _build_day((start_date + timedelta(days=1)).date().isoformat(), "run", "Easy aerobic run", 45, "Keep effort relaxed, conversational pace.", "easy"),
        _build_day((start_date + timedelta(days=2)).date().isoformat(), "mobility", "Mobility / strength", 30, "Mobility, core, light strength only.", "easy"),
        _build_day((start_date + timedelta(days=3)).date().isoformat(), "run", "Recovery run", 35, "Very easy effort, no quality.", "recovery"),
        _build_day((start_date + timedelta(days=4)).date().isoformat(), "rest", "Rest day", 0, "Prioritize sleep and recovery.", "rest"),
        _build_day((start_date + timedelta(days=5)).date().isoformat(), "run", "Easy long run", 75, "Steady easy effort, do not push.", "easy"),
        _build_day((start_date + timedelta(days=6)).date().isoformat(), "bike_or_walk", "Optional easy cross-training", 40, "Optional: easy spin or brisk walk.", "easy"),
    ]


def _generate_balanced_week(start_date: datetime) -> List[Dict[str, Any]]:
    return [
        _build_day((start_date + timedelta(days=0)).date().isoformat(), "run", "Easy aerobic run", 45, "Conversational effort.", "easy"),
        _build_day((start_date + timedelta(days=1)).date().isoformat(), "mobility", "Mobility / strength", 30, "Core, mobility, hips, glutes.", "easy"),
        _build_day((start_date + timedelta(days=2)).date().isoformat(), "run", "Moderate workout", 50, "Controlled moderate effort, not all-out.", "moderate"),
        _build_day((start_date + timedelta(days=3)).date().isoformat(), "rest", "Rest day", 0, "Recovery focus.", "rest"),
        _build_day((start_date + timedelta(days=4)).date().isoformat(), "run", "Easy run", 40, "Keep it light.", "easy"),
        _build_day((start_date + timedelta(days=5)).date().isoformat(), "run", "Long run", 90, "Easy-to-steady, no racing.", "long"),
        _build_day((start_date + timedelta(days=6)).date().isoformat(), "mobility", "Mobility / strength", 25, "Short support session.", "easy"),
    ]


def _generate_build_week(start_date: datetime) -> List[Dict[str, Any]]:
    return [
        _build_day((start_date + timedelta(days=0)).date().isoformat(), "run", "Easy aerobic run", 50, "Controlled, easy effort.", "easy"),
        _build_day((start_date + timedelta(days=1)).date().isoformat(), "strength", "Strength / mobility", 35, "General strength and mobility.", "easy"),
        _build_day((start_date + timedelta(days=2)).date().isoformat(), "run", "Quality session", 60, "Moderate-to-hard intervals, controlled.", "hard"),
        _build_day((start_date + timedelta(days=3)).date().isoformat(), "rest", "Rest day", 0, "No intensity.", "rest"),
        _build_day((start_date + timedelta(days=4)).date().isoformat(), "run", "Easy run", 45, "Relaxed pace.", "easy"),
        _build_day((start_date + timedelta(days=5)).date().isoformat(), "run", "Long run", 100, "Primary endurance session of the week.", "long"),
        _build_day((start_date + timedelta(days=6)).date().isoformat(), "bike_or_walk", "Optional easy cross-training", 45, "Optional easy session only.", "easy"),
    ]


def _generate_explanation(
    template_name: str,
    weekly_summary: Dict[str, Any],
    four_week_summary: Dict[str, Any],
    flags: Dict[str, Any],
    max_weekly_minutes: Optional[int],
    days_off: List[str],
    preferred_quality_days: List[str],
) -> str:
    lines = []
    lines.append(f"Selected template: {template_name}.")
    lines.append(f"Current weekly load: {weekly_summary.get('weekly_load', 0.0)}.")
    lines.append(f"Average 4-week load: {four_week_summary.get('avg_4w_load', 0.0)}.")

    current_flags = flags.get("flags", [])
    if current_flags:
        lines.append("Flags influencing the plan:")
        for flag in current_flags:
            lines.append(f"- {flag}")
    else:
        lines.append("No major flags detected.")

    if template_name == "recovery":
        lines.append("Plan bias: reduce intensity and restore freshness.")
    elif template_name == "build":
        lines.append("Plan bias: gradual progression with one quality session and one long run.")
    else:
        lines.append("Plan bias: balanced maintenance week.")

    # User preferences commentary
    if max_weekly_minutes:
        lines.append(f"User preference: keep total planned volume around {max_weekly_minutes} minutes.")
    if days_off:
        lines.append(f"User preferred days off: {', '.join(days_off)}.")
    if preferred_quality_days:
        lines.append(f"User preferred quality days: {', '.join(preferred_quality_days)}.")

    return "\n".join(lines)


def _generate_explanation_multi_session(max_sessions_per_day: int) -> str:
    if max_sessions_per_day <= 1:
        return ""
    return f"Up to {max_sessions_per_day} sessions per day, with balanced RPE (no overload)."


def _apply_days_off(
    sessions: List[Dict[str, Any]],
    days_off: List[str],
) -> None:
    if not days_off:
        return

    normalized_days_off = {d.lower() for d in days_off}
    for s in sessions:
        date_str = s.get("date")
        if not date_str:
            continue
        try:
            dt = datetime.fromisoformat(date_str)
        except Exception:
            continue
        weekday_idx = dt.weekday()  # 0=Monday
        weekday_name = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ][weekday_idx]
        if weekday_name in normalized_days_off:
            s["category"] = "rest"
            s["session_type"] = "Rest day"
            s["title"] = "Rest day"
            s["duration_min"] = 0
            s["duration"] = "0 min"
            s["intensity"] = "rest"
            s["notes"] = (s.get("notes") or "") + " (Adjusted to user day off)"


def _apply_max_weekly_minutes(
    sessions: List[Dict[str, Any]],
    max_weekly_minutes: Optional[int],
) -> None:
    if not max_weekly_minutes or max_weekly_minutes <= 0:
        return

    total = sum(int(s.get("duration_min") or 0) for s in sessions)
    if total <= max_weekly_minutes:
        return

    # Redukujemy w pierwszej kolejności sesje easy / recovery
    # w małych krokach, aż zejdziemy poniżej limitu.
    def is_easy_like(s: Dict[str, Any]) -> bool:
        return str(s.get("intensity", "")).lower() in {"easy", "recovery"}

    while total > max_weekly_minutes:
        changed = False
        for s in sessions:
            if not is_easy_like(s):
                continue
            cur = int(s.get("duration_min") or 0)
            if cur <= 20:
                continue
            s["duration_min"] = cur - 10
            s["duration"] = f"{s['duration_min']} min"
            s["notes"] = (s.get("notes") or "") + " (Slightly shortened to fit weekly time preference)"
            total -= 10
            changed = True
            if total <= max_weekly_minutes:
                break
        if not changed:
            break  # nie chcemy nieskończonej pętli


def _apply_preferred_quality_days(
    sessions: List[Dict[str, Any]],
    preferred_quality_days: List[str],
) -> None:
    if not preferred_quality_days:
        return

    preferred = {d.lower() for d in preferred_quality_days}
    weekday_names = [
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    ]

    def weekday_of(s: Dict[str, Any]) -> str | None:
        date_str = s.get("date")
        if not date_str:
            return None
        try:
            dt = datetime.fromisoformat(date_str)
        except Exception:
            return None
        return weekday_names[dt.weekday()]

    # Chcemy, żeby "środa jakościowa" faktycznie oznaczała, że najmocniejsza jednostka
    # (hard/moderate) wyląduje w preferowanym dniu. Nie przenosimy "long" (zwykle weekend).
    quality_candidates = [
        s for s in sessions
        if str(s.get("intensity", "")).lower() in {"hard", "moderate"}
    ]
    if not quality_candidates:
        return

    # znajdź pierwszą mocną jednostkę
    hard_session = quality_candidates[0]
    hard_wd = weekday_of(hard_session)
    if hard_wd in preferred:
        return

    # znajdź sesję w preferowanym dniu, którą możemy zamienić (najlepiej easy)
    swap_target = None
    for s in sessions:
        wd = weekday_of(s)
        if wd in preferred and str(s.get("intensity", "")).lower() in {"easy", "recovery"}:
            swap_target = s
            break
    if swap_target is None:
        # fallback: dowolna sesja w preferowanym dniu, byle nie long/hard
        for s in sessions:
            wd = weekday_of(s)
            if wd in preferred and str(s.get("intensity", "")).lower() not in {"long", "hard"}:
                swap_target = s
                break

    if swap_target is None:
        return

    # swap dates (and keep everything else)
    hard_date = hard_session.get("date")
    hard_session["date"] = swap_target.get("date")
    swap_target["date"] = hard_date

    hard_session["notes"] = (hard_session.get("notes") or "") + " (Moved to preferred quality day)"
    swap_target["notes"] = (swap_target.get("notes") or "") + " (Swapped to keep quality day preference)"


def generate_training_plan(
    days: int = 7,
    max_weekly_minutes: Optional[int] = None,
    days_off: Optional[List[str]] = None,
    preferred_quality_days: Optional[List[str]] = None,
    max_sessions_per_day: int = 1,
    weekend_focus: bool = False,
) -> Dict[str, Any]:
    days_off = days_off or []
    preferred_quality_days = preferred_quality_days or []
    max_sessions_per_day = max(1, min(3, int(max_sessions_per_day)))

    weekly_summary = _load_json(WEEKLY_SUMMARY_FILE)
    four_week_summary = _load_json(FOUR_WEEK_SUMMARY_FILE)
    flags = _load_json(FLAGS_FILE)

    if not weekly_summary:
        raise RuntimeError("Missing weekly_summary.json. Run analysis first.")

    template_name = _choose_plan_template(weekly_summary, four_week_summary, flags)

    start_date = datetime.now()
    if template_name == "recovery":
        sessions = _generate_recovery_week(start_date)
    elif template_name == "build":
        sessions = _generate_build_week(start_date)
    else:
        sessions = _generate_balanced_week(start_date)

    if days < len(sessions):
        sessions = sessions[:days]

    # Zastosuj preferencje użytkownika.
    _apply_days_off(sessions, days_off)
    _apply_max_weekly_minutes(sessions, max_weekly_minutes)
    _apply_preferred_quality_days(sessions, preferred_quality_days)

    # Weekend focus: jeśli user mówi, że ma czas w weekend, delikatnie podbij weekendowe sesje,
    # ale bez łamania max_weekly_minutes (na końcu re-apply constraint).
    if weekend_focus:
        weekday_names = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]

        def weekday_of(s: Dict[str, Any]) -> str | None:
            date_str = s.get("date")
            if not date_str:
                return None
            try:
                dt = datetime.fromisoformat(date_str)
            except Exception:
                return None
            return weekday_names[dt.weekday()]

        # 1) Ensure the longest session lands on Saturday if possible
        non_rest = [s for s in sessions if int(s.get("duration_min") or 0) > 0]
        if non_rest:
            longest = max(non_rest, key=lambda s: int(s.get("duration_min") or 0))
            # find a saturday session to swap dates with
            sat = next((s for s in sessions if weekday_of(s) == "saturday"), None)
            if sat and weekday_of(longest) != "saturday":
                longest_date = longest.get("date")
                longest["date"] = sat.get("date")
                sat["date"] = longest_date
                longest["notes"] = (longest.get("notes") or "") + " (Weekend focus: moved to Saturday)"
                sat["notes"] = (sat.get("notes") or "") + " (Weekend focus: swapped)"

        # 2) If weekend has a rest day, turn it into easy optional session
        for s in sessions:
            wd = weekday_of(s)
            if wd in {"saturday", "sunday"} and str(s.get("intensity", "")).lower() == "rest":
                s["category"] = "bike_or_walk"
                s["session_type"] = "Optional easy cross-training"
                s["title"] = s["session_type"]
                s["duration_min"] = 45
                s["duration"] = "45 min"
                s["intensity"] = "easy"
                s["notes"] = (s.get("notes") or "") + " (Weekend focus: use free time lightly)"

        # 3) Slightly extend weekend long/hard sessions
        for s in sessions:
            wd = weekday_of(s)
            if wd not in {"saturday", "sunday"}:
                continue
            if str(s.get("intensity", "")).lower() in {"long", "hard"}:
                cur = int(s.get("duration_min") or 0)
                if cur and cur < 180:
                    s["duration_min"] = min(180, cur + 20)
                    s["duration"] = f"{s['duration_min']} min"
                    s["notes"] = (s.get("notes") or "") + " (Weekend focus: slightly longer)"

        # Re-apply time constraint after weekend tweaks
        _apply_max_weekly_minutes(sessions, max_weekly_minutes)

    # Opcjonalnie: 2 lub 3 treningi dziennie przy zbilansowanym RPE
    sessions = expand_sessions_to_multi_per_day(sessions, max_sessions_per_day)

    explanation = _generate_explanation(
        template_name,
        weekly_summary,
        four_week_summary,
        flags,
        max_weekly_minutes=max_weekly_minutes,
        days_off=days_off,
        preferred_quality_days=preferred_quality_days,
    )
    if max_sessions_per_day > 1:
        explanation += "\n\n" + _generate_explanation_multi_session(max_sessions_per_day)

    plan = {
        "generated_at": datetime.now().isoformat(),
        "days": days,
        "template": template_name,
        "weekly_load": weekly_summary.get("weekly_load", 0.0),
        "avg_4w_load": four_week_summary.get("avg_4w_load", 0.0),
        "flags": flags.get("flags", []),
        "explanation": explanation,
        "sessions": sessions,
        "status": "draft",
    }

    _save_json(PLAN_CURRENT_FILE, plan)
    _append_plan_history(plan)

    return plan


if __name__ == "__main__":
    result = generate_training_plan(days=7)
    print("Plan generated.")
    print("Template:", result["template"])
    print("Sessions:", len(result["sessions"]))