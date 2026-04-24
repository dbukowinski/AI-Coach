from __future__ import annotations

import copy
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from agent_state import AgentState

# Tkinter (hitl_dialog) — import leniwy w ask_user_node, żeby Streamlit Cloud / headless
# mogły importować ten moduł bez _tkinter.

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# =========================
# Helpers
# =========================

def _safe_call(module_name: str, candidate_functions: List[str], *args, **kwargs):
    """
    Próbuje wywołać jedną z kilku możliwych funkcji z modułu.
    Dzięki temu nie musisz od razu refaktorować całego starego kodu.
    """
    module = __import__(module_name)

    for fn_name in candidate_functions:
        fn = getattr(module, fn_name, None)
        if callable(fn):
            return fn(*args, **kwargs)

    raise AttributeError(
        f"Nie znaleziono żadnej funkcji {candidate_functions} w module {module_name}."
    )


def _get_bedrock_client():
    """
    Tworzy klienta Bedrock Runtime.
    Region i model są pobierane z env (opcjonalnie).
    """
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
    return boto3.client("bedrock-runtime", region_name=region)


def _call_bedrock(prompt: str, max_tokens: int = 512, system_prompt: str = "") -> str:
    """
    Wywołanie Claude przez Amazon Bedrock (Messages API).
    Opcjonalny system_prompt jest przekazywany jako pole 'system'.
    """
    model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v1:0")
    client = _get_bedrock_client()

    body: Dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "top_p": 0.9,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        body["system"] = system_prompt

    try:
        response = client.invoke_model(
            body=json.dumps(body),
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        return payload["content"][0]["text"]
    except (BotoCoreError, ClientError, KeyError, json.JSONDecodeError, AttributeError) as e:
        print(f"[Agent] Bedrock call failed, falling back to deterministic logic: {e}")
        raise


def _format_plan_for_console(plan: Dict[str, Any]) -> str:
    if not plan:
        return "Brak planu."

    template = plan.get("template", "unknown")
    explanation = plan.get("explanation", "")
    sessions = plan.get("sessions", [])

    lines: List[str] = []
    lines.append("=== TRAINING PLAN ===")
    lines.append(f"Template: {template}")

    if explanation:
        lines.append(f"Explanation: {explanation}")

    for session in sessions:
        date = session.get("date", "N/A")
        session_type = session.get("session_type", "N/A")
        duration = session.get("duration", "N/A")
        intensity = session.get("intensity", "N/A")
        notes = session.get("notes", "")

        lines.append(f"{date}: {session_type}")
        lines.append(f"  {duration} | {intensity}")
        if notes:
            lines.append(f"  {notes}")

    return "\n".join(lines)


def _save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _export_plan_to_csv(plan: Dict[str, Any]) -> Path:
    """
    Eksportuje aktualny plan treningowy do pliku CSV, który można wygodnie
    otworzyć w Excelu / Google Sheets.
    Każdy wiersz to jedna sesja.
    """
    sessions = plan.get("sessions") or []
    target_dir = DATA_DIR
    target_dir.mkdir(exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = target_dir / f"training_plan_{stamp}.csv"

    fieldnames = [
        "date",
        "session_type",
        "category",
        "duration_min",
        "duration",
        "intensity",
        "notes",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in sessions:
            writer.writerow(
                {
                    "date": s.get("date", ""),
                    "session_type": s.get("session_type", ""),
                    "category": s.get("category", ""),
                    "duration_min": s.get("duration_min", ""),
                    "duration": s.get("duration", ""),
                    "intensity": s.get("intensity", ""),
                    "notes": s.get("notes", ""),
                }
            )

    return csv_path


def _build_coach_analysis_deterministic(
    weekly_summary: Dict[str, Any],
    four_week_summary: Dict[str, Any],
    flags: List[str],
    hr_zones_summary: Dict[str, Any],
) -> str:
    """
    Deterministyczna interpretacja – fallback, gdy LLM jest niedostępny.
    """
    weekly_load = weekly_summary.get("weekly_load", "N/A")
    avg_4w_load = four_week_summary.get("avg_4w_load", "N/A")

    parts = [
        "Training analysis summary:",
        f"- weekly_load: {weekly_load}",
        f"- avg_4w_load: {avg_4w_load}",
    ]

    if flags:
        parts.append(f"- flags: {', '.join(flags)}")
    else:
        parts.append("- flags: none")

    if hr_zones_summary:
        parts.append("- hr_zones_summary available")

    if "current_week_load_above_130pct_of_4w_avg" in flags:
        parts.append("- recommendation: bias toward reduced load / recovery")
    elif "low_recovery_volume" in flags:
        parts.append("- recommendation: add more recovery / easy volume")
    elif "high_intensity_frequency_high" in flags:
        parts.append("- recommendation: reduce hard sessions frequency")
    else:
        parts.append("- recommendation: balanced progression is acceptable")

    return "\n".join(parts)


def _build_coach_analysis(
    weekly_summary: Dict[str, Any],
    four_week_summary: Dict[str, Any],
    flags: List[str],
    hr_zones_summary: Dict[str, Any],
) -> str:
    """
    Wersja LLM: prosi model w Bedrocku o krótką, konkretną interpretację.
    W razie problemów wraca do wersji deterministycznej.
    """
    try:
        system_prompt = (
            "Jesteś doświadczonym trenerem biegowym. Analizujesz dane treningowe i piszesz "
            "krótką, konkretną interpretację po polsku. Styl: bezpośredni, coachingowy — "
            "nie streszczaj surowych liczb, interpretuj ich znaczenie dla sportowca."
        )
        user_prompt = (
            "Na podstawie poniższych metryk napisz analizę treningową zawierającą:\n"
            "• 3–5 punktorów (każdy to konkretna obserwacja lub wniosek — nie powtarzaj liczb bez interpretacji)\n"
            "• 1–2 zalecenia na kolejny tydzień\n\n"
            "Dane tygodniowe:\n"
            f"{json.dumps(weekly_summary, ensure_ascii=False, indent=2)}\n\n"
            "Dane z ostatnich 4 tygodni:\n"
            f"{json.dumps(four_week_summary, ensure_ascii=False, indent=2)}\n\n"
            "Flagi ryzyka:\n"
            f"{json.dumps(flags, ensure_ascii=False)}\n\n"
            "Strefy tętna:\n"
            f"{json.dumps(hr_zones_summary, ensure_ascii=False, indent=2)}\n\n"
            "Odpowiedź to czysty tekst gotowy do pokazania sportowcowi."
        )
        return _call_bedrock(user_prompt, max_tokens=512, system_prompt=system_prompt)
    except Exception:
        return _build_coach_analysis_deterministic(
            weekly_summary=weekly_summary,
            four_week_summary=four_week_summary,
            flags=flags,
            hr_zones_summary=hr_zones_summary,
        )


def _apply_feedback_rules_deterministic(plan: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    """
    Prosty, deterministyczny reviser – fallback gdy LLM jest niedostępny.
    Obsługuje m.in.: brak dnia wolnego, więcej treningu, konkretny dzień off, max min, lżej.
    """
    import re

    revised = copy.deepcopy(plan)
    sessions = revised.get("sessions", [])
    text = feedback.lower().strip()

    if not sessions or not text:
        return revised

    def _is_rest(s: Dict[str, Any]) -> bool:
        return (
            str(s.get("session_type", "")).lower() == "rest day"
            or str(s.get("intensity", "")).lower() == "rest"
        )

    def _get_duration_min(s: Dict[str, Any]) -> int:
        d = s.get("duration_min")
        if d is not None:
            return int(d)
        raw = str(s.get("duration", ""))
        m = re.search(r"(\d+)", raw)
        return int(m.group(1)) if m else 0

    def _set_duration(s: Dict[str, Any], minutes: int) -> None:
        s["duration_min"] = minutes
        s["duration"] = f"{minutes} min"

    # =========================
    # Extract dynamic "flags" from free-text feedback
    # =========================
    # Cel: sprawić, żeby np. "max weekly minutes 450" zachowywało się jak flaga:
    # --max-weekly-minutes 450, nawet podczas rewizji (bez restartu).
    weekday_map = {
        # PL (z polskimi znakami)
        "poniedziałek": "monday",
        "wtorek": "tuesday",
        "środa": "wednesday",
        "czwartek": "thursday",
        "piątek": "friday",
        "sobota": "saturday",
        "niedziela": "sunday",
        # PL (bez polskich znaków / odmiany)
        "poniedzialek": "monday",
        "sroda": "wednesday",
        "srode": "wednesday",
        "srodę": "wednesday",
        "piatek": "friday",
        "patek": "friday",
        # EN
        "monday": "monday",
        "tuesday": "tuesday",
        "wednesday": "wednesday",
        "thursday": "thursday",
        "friday": "friday",
        "saturday": "saturday",
        "sunday": "sunday",
    }

    extracted_max_weekly_minutes: Optional[int] = None
    try:
        # minutes / week
        m = re.search(r"(?:max|maks|limit)[^\\d]{0,20}(\\d+)\\s*(?:min|minutes|minut)[^\\n]{0,80}(?:week|tydzie)", text)
        if m:
            extracted_max_weekly_minutes = int(m.group(1))
        else:
            # hours / week (e.g. 8h per week)
            m = re.search(r"(?:max|maks|limit)[^\\d]{0,20}(\\d+)\\s*(?:h|godzin|hours)[^\\n]{0,80}(?:week|tydzie)", text)
            if m:
                extracted_max_weekly_minutes = int(m.group(1)) * 60
            else:
                # fallback: any "450 minutes ... week" even without explicit "max"
                m = re.search(r"(\\d+)\\s*(?:min|minutes|minut)[^\\n]{0,60}(?:week|tydzie)", text)
                if m:
                    extracted_max_weekly_minutes = int(m.group(1))
    except Exception:
        extracted_max_weekly_minutes = None

    extracted_days_off: List[str] = []
    # e.g. "dni wolne: poniedziałek i piątek" / "days off monday, friday" / "chcę wolną środę"
    if any(
        x in text
        for x in [
            "dni wolne",
            "day off",
            "days off",
            "wolne dni",
            "dzień wolny",
            "dzien wolny",
            "bez treningu",
            "no training",
            "wolne",
            "wolna",
            "wolny",
            "free day",
        ]
    ):
        for raw_day, slug in weekday_map.items():
            if raw_day in text:
                extracted_days_off.append(slug)

    # e.g. "wtorek odpada"
    if "odpada" in text or "off" in text:
        for raw_day, slug in weekday_map.items():
            if raw_day in text and any(x in text for x in ["odpada", "bez treningu", "off"]):
                extracted_days_off.append(slug)

    extracted_days_off = sorted(set(extracted_days_off))

    extracted_preferred_quality_days: List[str] = []
    # e.g. "środa jakościowa" / "jakosciowe dni: wtorek, sobota" / "quality day wednesday"
    if any(
        x in text
        for x in [
            "quality",
            "quality day",
            "jakości",
            "jakosci",
            "jakościowe",
            "jakosciowe",
            "jakościowy",
            "jakosciowy",
            "jakościowa",
            "jakosciowa",
            "mocne dni",
            "hard days",
            "trening jakościowy",
            "trening jakosciowy",
            "interwały",
            "interwaly",
        ]
    ):
        for raw_day, slug in weekday_map.items():
            if raw_day in text:
                extracted_preferred_quality_days.append(slug)
    extracted_preferred_quality_days = sorted(set(extracted_preferred_quality_days))

    extracted_max_sessions_per_day: Optional[int] = None
    extracted_max_training_days: Optional[int] = None
    # przykłady: "maks 4 dni na trening", "maksymalnie 4 dni treningowe", "max 4 training days"
    m = re.search(r"(?:maksymalnie|maks|max)\\s*(\\d+)\\s*(?:dni|days)[^\\n]{0,40}(?:trening|training)", text)
    if not m:
        m = re.search(r"(\\d+)\\s*(?:dni|days)[^\\n]{0,40}(?:maksymalnie|maks|max)[^\\n]{0,40}(?:trening|training)", text)
    if m:
        try:
            extracted_max_training_days = int(m.group(1))
        except ValueError:
            extracted_max_training_days = None

    extracted_weekend_focus = any(x in text for x in ["weekend mam czas", "więcej w weekend", "mam czas w weekend", "weekend", "sobota", "niedziela"])

    if "trzy" in text and "dzien" in text and "trening" in text:
        extracted_max_sessions_per_day = 3
    elif "dwa" in text and "dzien" in text and "trening" in text:
        extracted_max_sessions_per_day = 2
    else:
        if "3" in text and "trening" in text and "dzien" in text:
            extracted_max_sessions_per_day = 3
        elif "2" in text and "trening" in text and "dzien" in text:
            extracted_max_sessions_per_day = 2

    # Persist extracted preferences into plan metadata (so node logic can update AgentState)
    if extracted_max_weekly_minutes:
        revised["__extracted_max_weekly_minutes"] = extracted_max_weekly_minutes
    if extracted_days_off:
        revised["__extracted_days_off"] = extracted_days_off
    if extracted_preferred_quality_days:
        revised["__extracted_preferred_quality_days"] = extracted_preferred_quality_days
    if extracted_max_sessions_per_day:
        revised["__extracted_max_sessions_per_day"] = extracted_max_sessions_per_day

    # Apply extracted days off / quality preference early
    if extracted_days_off:
        normalized_days_off = {d.lower() for d in extracted_days_off}
        for s in sessions:
            date_str = s.get("date")
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(date_str)
            except Exception:
                continue
            weekday_idx = dt.weekday()
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
                s["session_type"] = "Rest day"
                s["category"] = "rest"
                s["title"] = "Rest day"
                s["duration_min"] = 0
                s["duration"] = "0 min"
                s["intensity"] = "rest"
                s["notes"] = (s.get("notes") or "") + " (Adjusted: user day off)".strip()

    if extracted_preferred_quality_days:
        preferred = {d.lower() for d in extracted_preferred_quality_days}
        for s in sessions:
            intensity = str(s.get("intensity", "")).lower()
            if intensity not in {"hard", "moderate-hard", "high", "moderate", "long"}:
                continue
            date_str = s.get("date")
            if not date_str:
                continue
            try:
                dt = datetime.fromisoformat(date_str)
            except Exception:
                continue
            weekday_idx = dt.weekday()
            weekday_name = [
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ][weekday_idx]
            if weekday_name not in preferred:
                # złagodź zamiast usuwać, żeby utrzymać sensowną objętość
                s["intensity"] = "easy"
                s["session_type"] = str(s.get("session_type", "Run")) + " (easier)"
                s["title"] = s["session_type"]
                s["notes"] = (s.get("notes") or "") + " (Adjusted: preferred quality days)".strip()

    # Jeśli user mówi "max 4 dni treningu", ale nie podał konkretnie dni wolnych,
    # to NIE zgadujemy – zamieniamy to w pytanie do użytkownika (dialog trenerski).
    if extracted_max_training_days is not None and not extracted_days_off:
        # Zostawiamy sessions bez agresywnych zmian; pytanie będzie ustawione wyżej w logice node.
        revised["__extracted_max_training_days"] = extracted_max_training_days

    if extracted_weekend_focus:
        revised["__extracted_weekend_focus"] = True

    # 0) "nie chce rest day" / "trenowac codziennie" / "wszystkie dni treningowe"
    no_rest_phrases = [
        "nie chce rest", "no rest", "bez dnia wolnego", "trenowac codziennie",
        "ćwiczyć codziennie", "wszystkie dni treningowe", "codziennie trening",
        "all days training", "every day", "train every day", "no rest day",
        "wszystkie dni treningowe",
    ]
    if any(p in text for p in no_rest_phrases):
        for s in sessions:
            if _is_rest(s):
                s["session_type"] = "Easy aerobic run"
                s["category"] = "run"
                _set_duration(s, 45)
                s["intensity"] = "easy"
                s["notes"] = "Dzień treningowy zamiast odpoczynku (na życzenie)."
                if "title" in s:
                    s["title"] = s["session_type"]

    # 0b) "więcej treningu" / "po 1h" / "weekend 2-4h" – zwiększ czas sesji
    more_volume_phrases = [
        "więcej treningu", "dorzuc", "po 1h", "1h ", " 1h", "2-4h", "2h", "4h",
        "weekend 2", "weekend 4", "przynajmniej po 1h", "2-4 h",
    ]
    if any(p in text for p in more_volume_phrases):
        weekday_min_target = 60
        weekend_long_min = 120
        if "4h" in text or "4 h" in text or "240" in text:
            weekend_long_min = 240
        elif "2h" in text or "2 h" in text or "120" in text:
            weekend_long_min = max(weekend_long_min, 120)

        for s in sessions:
            if _is_rest(s):
                continue
            dur = _get_duration_min(s)
            date_str = s.get("date", "")
            is_weekend = False
            if date_str:
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    is_weekend = dt.weekday() in (5, 6)  # sobota, niedziela
                except Exception:
                    pass

            if is_weekend and str(s.get("intensity", "")).lower() in ("long", "hard"):
                if dur < weekend_long_min:
                    _set_duration(s, min(weekend_long_min, 240))
                    s["notes"] = f"{s.get('notes', '')} Wydłużone na życzenie (weekend).".strip()
            elif not is_weekend and dur < weekday_min_target and dur > 0:
                new_dur = min(weekday_min_target, dur + 20)
                _set_duration(s, new_dur)
                s["notes"] = f"{s.get('notes', '')} Wydłużone do ~1h (na życzenie).".strip()

    weekday_map = {
        "poniedziałek": "monday",
        "wtorek": "tuesday",
        "środa": "wednesday",
        "czwartek": "thursday",
        "piątek": "friday",
        "sobota": "saturday",
        "niedziela": "sunday",
        "monday": "monday",
        "tuesday": "tuesday",
        "wednesday": "wednesday",
        "thursday": "thursday",
        "friday": "friday",
        "saturday": "saturday",
        "sunday": "sunday",
    }

    # 1) "wtorek odpada" / "remove tuesday" – tylko jeśli user NIE mówi "wszystkie dni treningowe"
    if not any(p in text for p in no_rest_phrases):
        for raw_day, norm_day in weekday_map.items():
            if raw_day in text and any(x in text for x in ["odpada", "nie mogę", "remove", "off", "bez treningu"]):
                for s in sessions:
                    date_val = str(s.get("date", "")).lower()
                    session_type = str(s.get("session_type", "")).lower()
                    if norm_day in date_val or raw_day in date_val or raw_day in session_type:
                        s["session_type"] = "Rest day"
                        s["duration"] = "0 min"
                        s["duration_min"] = 0
                        s["intensity"] = "rest"
                        s["notes"] = f"Adjusted due to user feedback: {feedback}"

    # 2) "max 45 min" / "tylko 45 min"
    match = re.search(r"(\d+)\s*min", text)
    if match and any(x in text for x in ["max", "maks", "tylko", "only"]):
        max_minutes = int(match.group(1))
        for s in sessions:
            if _get_duration_min(s) > max_minutes:
                _set_duration(s, max_minutes)
                s["notes"] = f"{s.get('notes', '')} Adjusted to max {max_minutes} min.".strip()

    # 3) "lżej", "more easy", "mniej intensywnie"
    if any(x in text for x in ["lżej", "łatwiej", "more easy", "easier", "mniej intensywnie", "less intensity"]):
        for s in sessions:
            if str(s.get("intensity", "")).lower() in {"hard", "moderate-hard", "high"}:
                s["intensity"] = "easy"
                s["notes"] = f"{s.get('notes', '')} Intensity reduced after feedback.".strip()

    # 4) "2/3 treningi dziennie" – rozbij dni na 2 lub 3 sesje przy zbilansowanym RPE
    multi_3 = ["3 treningi dziennie", "trzy treningi", "3 sesje dziennie", "więcej niż 2 treningi", "po 3 treningi"]
    multi_2 = ["2 treningi dziennie", "dwa treningi", "2 sesje dziennie", "po 2 treningi"]
    if any(p in text for p in multi_3):
        from plan_generator import expand_sessions_to_multi_per_day
        revised["sessions"] = expand_sessions_to_multi_per_day(revised["sessions"], 3)
    elif any(p in text for p in multi_2):
        from plan_generator import expand_sessions_to_multi_per_day
        revised["sessions"] = expand_sessions_to_multi_per_day(revised["sessions"], 2)

    # Apply extracted weekly minutes as a final constraint (after other changes)
    if extracted_max_weekly_minutes:
        try:
            from plan_generator import _apply_max_weekly_minutes

            _apply_max_weekly_minutes(revised["sessions"], extracted_max_weekly_minutes)
        except Exception:
            # jeśli nie uda się zastosować algorytmu, zachowujemy resztę zmian
            pass

    # Apply extracted max sessions/day if it wasn't triggered by existing multi rules
    if extracted_max_sessions_per_day in (2, 3):
        try:
            from plan_generator import expand_sessions_to_multi_per_day

            revised["sessions"] = expand_sessions_to_multi_per_day(revised["sessions"], extracted_max_sessions_per_day)
        except Exception:
            pass

    revised["status"] = "revised"
    revised["explanation"] = (
        f"{plan.get('explanation', '')}\n\n"
        f"Revised after user feedback: {feedback}"
    ).strip()

    return revised


def _apply_feedback_rules(plan: Dict[str, Any], feedback: str) -> Dict[str, Any]:
    """
    Wersja LLM: prosi model o przepisanie planu JSON zgodnie z feedbackiem.
    W razie problemów wraca do prostych reguł deterministycznych.
    """
    if not feedback.strip():
        return plan

    try:
        system_prompt = (
            "Jesteś asystentem trenera biegowego. Modyfikujesz plan treningowy w formacie JSON "
            "zgodnie z feedbackiem sportowca. Zwracasz WYŁĄCZNIE poprawny JSON — zero komentarzy poza nim.\n\n"
            "Zasady:\n"
            "• Zachowaj oryginalną strukturę i wszystkie klucze JSON.\n"
            "• Uwzględnij feedback rozsądnie — nie dopuść do niebezpiecznego przeciążenia.\n"
            "• Jeśli prośba jest niejasna, wprowadź minimalną, bezpieczną zmianę."
        )
        user_prompt = (
            "Zmodyfikuj plan treningowy zgodnie z poniższym feedbackiem.\n\n"
            "Oryginalny plan (JSON):\n"
            f"{json.dumps(plan, ensure_ascii=False, indent=2)}\n\n"
            "Feedback sportowca:\n"
            f"{feedback}\n\n"
            "Zwróć poprawiony plan jako JSON:"
        )
        raw = _call_bedrock(user_prompt, max_tokens=1024, system_prompt=system_prompt)
        # Model może owinąć JSON w markdown – spróbujmy to oczyścić.
        text = raw.strip()
        if text.startswith("```"):
            # Usuń ewentualny prefix ```json / ``` i suffix ```.
            text = text.strip("`")
            if "\n" in text:
                text = "\n".join(text.split("\n")[1:])
        revised = json.loads(text)
        if not isinstance(revised, dict):
            raise ValueError("Revised plan is not a JSON object.")
        return revised
    except Exception as e:
        print(f"[Agent] Bedrock feedback reviser failed, falling back to deterministic rules: {e}")
        return _apply_feedback_rules_deterministic(plan, feedback)


# =========================
# Deterministic pipeline nodes
# =========================

def ensure_token_node(state: AgentState) -> AgentState:
    try:
        state.log("ensure_token")
        _safe_call("sync_strava", ["ensure_token", "ensure_valid_token", "refresh_if_needed"])
    except Exception as e:
        state.add_error(f"ensure_token failed: {e}")
        state.done = True
    return state


def fetch_activities_node(state: AgentState) -> AgentState:
    try:
        state.log(f"fetch_activities days={state.days}")
        result = _safe_call(
            "sync_strava",
            [
                "sync_last_days",          # high-level sync: list + details + save_raw
                "fetch_last_days",         # fallback: only list, no details
                "fetch_activities",
                "fetch_recent_activities",
                "get_recent_activities",
            ],
            days=state.days,
        )

        # sync_last_days / fetch_last_days zwracają zazwyczaj słownik ze statystykami,
        # ale na potrzeby dalszych kroków nie jest to krytyczne – ważne, że dane są w plikach.
        if isinstance(result, list):
            state.fetched_activity_ids = [
                x.get("id") if isinstance(x, dict) else x for x in result
            ]
        elif isinstance(result, dict):
            # np. {"fetched": 20, "new": 3, "saved": 3}
            state.fetched_activity_ids = result.get("fetched_activity_ids", [])
            state.new_activity_ids = result.get("new_activity_ids", state.new_activity_ids)
        else:
            state.fetched_activity_ids = []
    except Exception as e:
        state.add_error(f"fetch_activities failed: {e}")
        state.done = True
    return state


def fetch_new_details_node(state: AgentState) -> AgentState:
    # Przy obecnej implementacji sync_strava.sync_last_days() od razu pobiera szczegóły
    # i zapisuje je do pliku activities_raw.json, więc ten krok jest no-opem.
    state.log("fetch_new_details (noop – handled by sync_last_days)")
    return state


def save_raw_node(state: AgentState) -> AgentState:
    # sync_last_days zapisuje już dane do activities_raw.json; ten krok zostaje
    # jako miejsce na ewentualne dodatkowe zapisy w przyszłości.
    state.log("save_raw (noop – raw saved by sync_last_days)")
    return state


def clean_data_node(state: AgentState) -> AgentState:
    try:
        state.log("clean_data")
        result = _safe_call(
            "clean",
            [
                "clean_data",
                "run_clean",
                "build_clean_dataset",
                "clean_raw_to_clean",  # faktyczna funkcja w clean.py
            ],
        )

        if isinstance(result, list):
            state.clean_activities = result
        elif isinstance(result, dict):
            state.clean_activities = result.get("clean_activities", [])
        else:
            state.clean_activities = []
    except Exception as e:
        state.add_error(f"clean_data failed: {e}")
        state.done = True
    return state


def report_node(state: AgentState) -> AgentState:
    try:
        state.log("report")
        result = _safe_call("report", ["generate_report", "report", "build_summary_report"], days=state.days)

        if isinstance(result, dict):
            state.summary = result
        else:
            state.summary = {}
    except Exception as e:
        state.add_error(f"report failed: {e}")
        state.done = True
    return state


# =========================
# Reasoning / planning nodes
# =========================

def analyze_training_node(state: AgentState) -> AgentState:
    try:
        state.log("analyze_training")

        result = _safe_call(
            "analysis",
            ["analyze_training", "run_analysis", "analyze"],
            days=state.days,
        )

        if not isinstance(result, dict):
            result = {}

        state.hr_zones_summary = result.get("hr_zones_summary", {})
        state.weekly_summary = result.get("weekly_summary", {})
        state.four_week_summary = result.get("four_week_summary", {})
        state.flags = result.get("flags", [])

        state.coach_analysis = _build_coach_analysis(
            weekly_summary=state.weekly_summary,
            four_week_summary=state.four_week_summary,
            flags=state.flags,
            hr_zones_summary=state.hr_zones_summary,
        )

        _save_json(DATA_DIR / "coach_analysis.json", {"coach_analysis": state.coach_analysis})
    except Exception as e:
        state.add_error(f"analyze_training failed: {e}")
        state.done = True
    return state


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _build_training_summary(state: AgentState) -> Dict[str, Any]:
    return {
        "weekly_summary": state.weekly_summary,
        "four_week_summary": state.four_week_summary,
        "flags": state.flags,
        "hr_zones_summary": state.hr_zones_summary,
    }


def _coaching_brief_deterministic(state: AgentState) -> None:
    w = state.weekly_summary or {}
    fw = state.four_week_summary or {}
    fl = state.flags or []
    obs: List[str] = []
    wl = w.get("weekly_load")
    av = fw.get("avg_4w_load")
    if wl is not None and av not in (None, 0, 0.0):
        try:
            wlf, avf = float(wl), float(av)
            if avf > 0 and wlf < avf * 0.75:
                obs.append(
                    f"Ostatni tydzień: obciążenie wyraźnie poniżej 4-tygodniowej średniej ({wlf} vs śr. {avf})."
                )
            elif avf > 0 and wlf > avf * 1.25:
                obs.append(
                    f"Ostatni tydzień: obciążenie powyżej średniej 4-tygodniowej ({wlf} vs śr. {avf})."
                )
        except (TypeError, ValueError):
            pass
    if fl:
        obs.append("Sygnały z analizy: " + ", ".join(str(x) for x in fl[:6]))
    if not obs:
        obs.append("Wzorce są jeszcze mało czytelne — zaczniemy od formy i celu na najbliższe dni.")
    state.patterns = obs[:5]
    state.coaching_hypothesis = (
        "Być może tydzień był celowo lżejszy albo życie codzienne zajęło priorytet — warto to potwierdzić."
    )
    state.opening_message = (
        "Zanim ułożymy plan: jak oceniasz ostatni tydzień treningowy "
        "względem tego, co chciałeś zrobić — poszło po Twojej myśli, czy coś go rozbiło?"
    )


def _coaching_brief_llm(state: AgentState) -> None:
    payload = _build_training_summary(state)
    system_prompt = (
        "Jesteś doświadczonym trenerem biegowym przygotowującym się do rozmowy z zawodnikiem "
        "przed ułożeniem planu tygodnia. Odpowiadasz wyłącznie w formacie JSON, po polsku."
    )
    user_prompt = (
        "Przeanalizuj poniższe dane treningowe i przygotuj briefing.\n"
        "Nie kopiuj surowych liczb — wyciągaj wnioski i interpretuj wzorce.\n\n"
        "Zwróć WYŁĄCZNIE poprawny JSON bez markdown, dokładnie w tej strukturze:\n"
        "{\n"
        '  "observations": ["obserwacja 1", "obserwacja 2"],\n'
        '  "opening_question": "pierwsze zdanie do sportowca",\n'
        '  "hypothesis": "jedno zdanie o tym, co prawdopodobnie się dzieje"\n'
        "}\n\n"
        "Wymagania:\n"
        '• "observations": 2–3 konkretne obserwacje w tonie coachingowym\n'
        '• "opening_question": zacznij od wzorca z danych, nie od pytania ogólnego (np. "ile dni trenujesz")\n'
        '• "hypothesis": krótka hipoteza o sytuacji sportowca (zmęczenie, lżejszy tydzień, dobra forma itp.)\n\n'
        f"Dane treningowe:\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    raw = _call_bedrock(user_prompt, max_tokens=700, system_prompt=system_prompt)
    data = _parse_json_object(raw)
    if not data:
        raise ValueError("brief JSON parse failed")
    obs = data.get("observations")
    if not isinstance(obs, list) or not obs:
        raise ValueError("brief missing observations")
    oq = data.get("opening_question")
    if not isinstance(oq, str) or not oq.strip():
        raise ValueError("brief missing opening_question")
    hyp = data.get("hypothesis", "")
    state.patterns = [str(x).strip() for x in obs[:5] if str(x).strip()]
    state.opening_message = oq.strip()
    state.coaching_hypothesis = str(hyp).strip() if isinstance(hyp, str) else ""


def coaching_brief_node(state: AgentState) -> AgentState:
    """
    Przygotowuje briefing coachingowy przed dialogiem (obserwacje + otwarcie).
    """
    try:
        state.log("coaching_brief")
        state.training_summary = _build_training_summary(state)
        try:
            _coaching_brief_llm(state)
        except Exception as e:
            state.log(f"coaching_brief LLM fallback: {e}")
            _coaching_brief_deterministic(state)
        state.coaching_brief_ready = True
        _save_json(
            DATA_DIR / "coaching_brief.json",
            {
                "training_summary": state.training_summary,
                "patterns": state.patterns,
                "opening_message": state.opening_message,
                "hypothesis": state.coaching_hypothesis,
            },
        )
    except Exception as e:
        state.add_error(f"coaching_brief failed: {e}")
        state.done = True
    return state


def _format_messages_for_prompt(messages: List[Dict[str, str]]) -> str:
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        label = "Trener" if role == "assistant" else "Sportowiec"
        lines.append(f"{label}: {content}")
    return "\n".join(lines)


def _strip_plan_ready_marker(text: str) -> Tuple[str, bool]:
    marker = "PLAN_READY"
    if marker in text:
        cleaned = text.replace(marker, "").strip()
        return cleaned, True
    return text.strip(), False


def _coach_turn_fallback_deterministic(state: AgentState) -> Tuple[str, bool]:
    """
    Fallback gdy LLM (Bedrock) jest niedostępny (np. brak AWS credentials).
    Zwraca krótką wypowiedź + jedno pytanie i od razu przechodzi do planu.
    """
    w = state.weekly_summary or {}
    fw = state.four_week_summary or {}
    wl = w.get("weekly_load")
    av = fw.get("avg_4w_load")
    bits: List[str] = []
    if wl is not None and av is not None:
        try:
            wlf = float(wl)
            avf = float(av)
            if avf > 0:
                if wlf < avf * 0.8:
                    bits.append("Widzę, że ostatni tydzień był wyraźnie lżejszy niż Twoja średnia z 4 tygodni.")
                elif wlf > avf * 1.25:
                    bits.append("Widzę, że ostatni tydzień był mocniejszy niż Twoja średnia z 4 tygodni.")
        except Exception:
            pass
    if not bits and state.patterns:
        bits.append(f"Z danych: {state.patterns[0]}")
    if not bits:
        bits.append("OK — mam ogólny obraz z danych z ostatnich tygodni.")

    question = (
        "Zanim ułożę plan: jaki jest najbliższy cel (start / dystans / data) i ile dni realnie możesz biegać w tym tygodniu?"
    )
    return (" ".join(bits) + " " + question).strip(), True


def _coach_turn_llm(state: AgentState) -> Tuple[str, bool]:
    summary_json = json.dumps(state.training_summary, ensure_ascii=False, indent=2)
    patterns_txt = "\n".join(f"• {p}" for p in state.patterns) or "• (brak)"
    flags_txt = "\n".join(f"• {f}" for f in (state.flags or [])) or "• (brak)"
    history = _format_messages_for_prompt(state.messages)

    system_prompt = (
        "Jesteś doświadczonym trenerem biegowym prowadzącym krótką rozmowę przed ułożeniem "
        "planu tygodniowego dla zawodnika.\n\n"
        "Zasady:\n"
        "• Cytuj konkretne dane, gdy wzmacniają Twoją obserwację.\n"
        "• Zadawaj jedno pytanie na raz — poprzedź je krótką obserwacją lub opinią.\n"
        "• Miej zdanie; nie zbieraj odpowiedzi jak formularz.\n"
        "• Gdy wiesz już wystarczająco dużo (cel sportowca, dostępność dni w tygodniu, "
        "ewentualne zmęczenie lub kontuzja) — zakończ wypowiedź tokenem PLAN_READY "
        "(samodzielnie, bez żadnego tekstu po nim).\n"
        "• Pisz po polsku, naturalnie. Maksymalnie kilka zdań + jedno pytanie lub zamknięcie."
    )
    user_prompt = (
        f"Dane treningowe (JSON):\n{summary_json}\n\n"
        f"Wzorce z analizy:\n{patterns_txt}\n\n"
        f"Flagi ryzyka:\n{flags_txt}\n\n"
        f"Hipoteza (wewnętrzna): {state.coaching_hypothesis or '(brak)'}\n\n"
        f"Rozmowa do tej pory:\n{history or '(brak — to pierwsze słowo trenera)'}\n\n"
        "Napisz kolejną wypowiedź trenera. Tylko treść — bez nagłówków ani etykiet."
    )
    try:
        raw = _call_bedrock(user_prompt, max_tokens=600, system_prompt=system_prompt)
        visible, done = _strip_plan_ready_marker(raw)
        return visible, done
    except Exception as e:
        state.log(f"coach_turn LLM unavailable, fallback to deterministic: {e}")
        return _coach_turn_fallback_deterministic(state)


def _extract_context_llm(state: AgentState) -> Dict[str, Any]:
    conv = _format_messages_for_prompt(state.messages)
    system_prompt = (
        "Jesteś ekstraherem preferencji treningowych. Na podstawie rozmowy trener–sportowiec "
        "wyciągasz dane do planowania tygodnia i zwracasz wyłącznie JSON — zero komentarzy poza nim."
    )
    user_prompt = (
        "Z poniższej rozmowy wyciągnij preferencje treningowe sportowca.\n\n"
        "Zwróć WYŁĄCZNIE poprawny JSON bez markdown, w tej strukturze:\n"
        "{\n"
        '  "max_weekly_minutes": number | null,\n'
        '  "days_off": ["monday", "tuesday", ...] | [],\n'
        '  "preferred_quality_days": ["wednesday", ...] | [],\n'
        '  "weekend_focus": true | false,\n'
        '  "max_training_days": number | null,\n'
        '  "max_sessions_per_day": 1 | 2 | 3 | null,\n'
        '  "goal_summary": "krótki opis celu" | ""\n'
        "}\n\n"
        "Zasady:\n"
        "• Nazwy dni ZAWSZE po angielsku: monday, tuesday, wednesday, thursday, friday, saturday, sunday.\n"
        "• Jeśli pole nie padło w rozmowie — wstaw null lub [].\n"
        "• Przykład: sportowiec mówi 'wolna środa' → days_off: [\"wednesday\"].\n"
        "• Przykład: 'cel: półmaraton w maju' → goal_summary: \"półmaraton w maju\".\n\n"
        f"Rozmowa:\n{conv}"
    )
    try:
        raw = _call_bedrock(user_prompt, max_tokens=500, system_prompt=system_prompt)
        data = _parse_json_object(raw)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _apply_extracted_context(state: AgentState, ctx: Dict[str, Any]) -> None:
    state.extracted_context = ctx
    mwm = ctx.get("max_weekly_minutes")
    if isinstance(mwm, int) and mwm >= 60:
        state.max_weekly_minutes = mwm

    days_off = ctx.get("days_off")
    if isinstance(days_off, list) and days_off:
        state.days_off = sorted({str(d).lower() for d in days_off if d})

    qd = ctx.get("preferred_quality_days")
    if isinstance(qd, list) and qd:
        state.preferred_quality_days = sorted({str(d).lower() for d in qd if d})

    if ctx.get("weekend_focus") is True:
        state.weekend_focus = True

    mtd = ctx.get("max_training_days")
    if isinstance(mtd, int) and 1 <= mtd <= 7:
        state.max_training_days = mtd

    mspd = ctx.get("max_sessions_per_day")
    if mspd in (1, 2, 3):
        state.max_sessions_per_day = int(mspd)


def _enrich_plan_explanation_with_conversation(state: AgentState, plan: Dict[str, Any]) -> Dict[str, Any]:
    conv = _format_messages_for_prompt(state.messages)
    summary_json = json.dumps(state.training_summary, ensure_ascii=False, indent=2)
    system_prompt = (
        "Jesteś doświadczonym trenerem biegowym. Piszesz krótkie, konkretne uzasadnienia "
        "planów treningowych po polsku. Styl: bezpośredni, motywujący — bez ogólników."
    )
    user_prompt = (
        "Na podstawie danych treningowych i rozmowy z zawodnikiem dopisz krótki komentarz do planu.\n"
        "Komentarz (2–5 zdań po polsku): co jest najważniejsze w tym tygodniu i jak rozmowa to uzasadnia.\n"
        "Odpowiedź WYŁĄCZNIE plain text — bez JSON, nagłówków ani list.\n\n"
        f"Dane treningowe:\n{summary_json}\n\n"
        f"Rozmowa z zawodnikiem:\n{conv}\n\n"
        f"Obecne uzasadnienie planu:\n{plan.get('explanation', '(brak)')}"
    )
    try:
        addon = _call_bedrock(user_prompt, max_tokens=400, system_prompt=system_prompt).strip()
        if addon:
            base = (plan.get("explanation") or "").strip()
            plan["explanation"] = (base + "\n\n— Komentarz trenera —\n" + addon).strip()
    except Exception as e:
        state.log(f"plan explanation enrich skipped: {e}")
    return plan


def coach_dialog_node(state: AgentState) -> AgentState:
    """
    Prowadzi dialog coachingowy przed planem. W CLI — pętla z input().
    W GUI — pomijamy (produkt działa); pełny dialog w terminalu (--no-gui).
    """
    try:
        state.log("coach_dialog")

        if state.hitl_mode != "cli":
            state.log("coach_dialog: tryb GUI — pomijam interaktywny dialog (użyj --no-gui).")
            state.dialog_complete = True
            return state

        if not state.opening_message:
            _coaching_brief_deterministic(state)
            state.opening_message = state.opening_message or "Jak oceniasz ostatni tydzień treningowy?"

        if not state.messages:
            state.messages.append({"role": "assistant", "content": state.opening_message})

        max_turns = 14
        turn = 0
        while not state.dialog_complete and turn < max_turns:
            turn += 1
            last = state.messages[-1]
            if last.get("role") == "assistant":
                print()
                print(f"Coach: {last.get('content', '')}")
                print()
                try:
                    user_line = input("Ty (Enter = zakończ rozmowę i przejdź do planu): ").strip()
                except EOFError:
                    user_line = ""
                if not user_line:
                    state.dialog_complete = True
                    state.log("coach_dialog: pusty input — kończę dialog.")
                    break
                state.messages.append({"role": "user", "content": user_line})

            reply, done = _coach_turn_llm(state)
            if not reply and not done:
                reply = "Dzięki za kontekst. PLAN_READY"
                done = True
            state.messages.append({"role": "assistant", "content": reply})
            state.dialog_complete = bool(done)

        if not state.dialog_complete:
            state.log("coach_dialog: limit tur — wymuszam zakończenie dialogu.")
            state.dialog_complete = True

        ctx = _extract_context_llm(state)
        _apply_extracted_context(state, ctx)
        _save_json(DATA_DIR / "coaching_transcript.json", {"messages": state.messages, "extracted": ctx})
    except Exception as e:
        state.add_error(f"coach_dialog failed: {e}")
        state.dialog_complete = True
    return state


def streamlit_seed_coaching_messages(state: AgentState) -> AgentState:
    """Pierwsza wiadomość coacha w UI (po coaching_brief_node)."""
    if not (state.opening_message or "").strip():
        _coaching_brief_deterministic(state)
    if not state.messages:
        msg = (state.opening_message or "").strip() or (
            "Opowiedz proszę, jak minął ostatni tydzień treningowy — co poszło dobrze, a co Cię zaskoczyło?"
        )
        state.messages.append({"role": "assistant", "content": msg})
    return state


def streamlit_coach_after_user(state: AgentState, user_text: str) -> Tuple[AgentState, bool]:
    """
    Jedna tura: wiadomość użytkownika + odpowiedź modelu.
    Zwraca (stan, plan_ready).
    """
    text = (user_text or "").strip()
    if not text:
        return state, False
    state.messages.append({"role": "user", "content": text})
    reply, done = _coach_turn_llm(state)
    if not reply and not done:
        reply = "Dzięki za rozmowę — przechodzę do ułożenia planu."
        done = True
    state.messages.append({"role": "assistant", "content": reply})
    return state, done


def streamlit_finalize_coaching_and_plan(state: AgentState) -> AgentState:
    """Ekstrakcja preferencji z rozmowy + generate_plan_node (deterministyczny plan + ewentualne LLM w explanation)."""
    state.dialog_complete = True
    ctx = _extract_context_llm(state)
    _apply_extracted_context(state, ctx)
    _save_json(
        DATA_DIR / "coaching_transcript.json",
        {"messages": state.messages, "extracted": ctx},
    )
    return generate_plan_node(state)


def streamlit_revise_plan_after_user(state: AgentState, user_text: str) -> AgentState:
    """
    Kontynuacja dialogu PO wygenerowaniu planu: traktujemy wiadomość jako feedback
    i przepuszczamy przez `revise_plan_node`, a następnie dopisujemy odpowiedź coacha
    do `state.messages` (żeby UI miało ciągłą historię).
    """
    text = (user_text or "").strip()
    if not text:
        return state

    # user mówi w czacie
    state.messages.append({"role": "user", "content": text})

    # rewizja planu (deterministycznie lub przez Bedrock, zależnie od dostępności)
    state.user_feedback = text
    state = revise_plan_node(state)

    # odpowiedź coacha do UI
    reply = (state.coach_question or "").strip() or "Zaktualizowałem plan. Co jeszcze dopracować?"
    state.messages.append({"role": "assistant", "content": reply})

    return state

def build_streamlit_demo_agent_state() -> AgentState:
    """
    Stan demonstracyjny bez Stravy — do podglądu UI i dialogu.
    """
    state = AgentState(mode="full", hitl_mode="streamlit", days=28)
    state.coach_analysis = "demo"
    state.weekly_summary = {
        "activity_count": 16,
        "weekly_load": 36.4,
        "zone_counts": {"z1": 6, "z2": 5, "z3": 2, "z4": 1, "z5": 0, "unknown": 2},
    }
    state.four_week_summary = {
        "avg_4w_load": 44.0,
        "current_week_load": 22.0,
        "weeks": [],
    }
    state.flags = []
    state.hr_zones_summary = {"days": 28, "hr_max": 178}
    state.training_summary = _build_training_summary(state)
    # Zapis do plików — generate_training_plan czyta weekly/four_week/flags z dysku.
    _save_json(DATA_DIR / "weekly_summary.json", state.weekly_summary)
    _save_json(DATA_DIR / "four_week_summary.json", state.four_week_summary)
    _save_json(DATA_DIR / "flags.json", {"flags": state.flags})
    _save_json(DATA_DIR / "hr_zones_summary.json", state.hr_zones_summary)
    try:
        _coaching_brief_llm(state)
    except Exception:
        _coaching_brief_deterministic(state)
    state.coaching_brief_ready = True
    return streamlit_seed_coaching_messages(state)


def generate_plan_node(state: AgentState) -> AgentState:
    try:
        state.log("generate_plan")

        # Formularz preferencji CLI tylko gdy nie było coachingowego dialogu przed planem.
        skip_prefs_form = (
            state.dialog_complete
            or state.preferences_collected
            or bool(state.messages)
        )
        if state.hitl_mode == "cli" and not state.preferences_collected and not skip_prefs_form:
            print()
            print("Before I create your plan, a few quick questions.")
            print("(Press ENTER to keep the suggested defaults.)")
            try:
                val = input(
                    f"- Roughly how many minutes per week can you train? "
                    f"[current: {state.max_weekly_minutes}]: "
                ).strip()
            except EOFError:
                val = ""
            if val:
                try:
                    state.max_weekly_minutes = max(60, int(val))
                except ValueError:
                    state.log("Could not parse max_weekly_minutes, keeping previous value.")

            try:
                days_off_raw = input(
                    "- Days you prefer to keep free from training "
                    "(comma separated, e.g. monday, friday) [leave empty for none]: "
                ).strip()
            except EOFError:
                days_off_raw = ""
            if days_off_raw:
                state.days_off = [
                    d.strip().lower()
                    for d in days_off_raw.split(",")
                    if d.strip()
                ]

            try:
                qdays_raw = input(
                    "- Preferred days for harder sessions "
                    "(comma separated, e.g. tuesday, saturday) [leave empty for flexible]: "
                ).strip()
            except EOFError:
                qdays_raw = ""
            if qdays_raw:
                state.preferred_quality_days = [
                    d.strip().lower()
                    for d in qdays_raw.split(",")
                    if d.strip()
                ]

            try:
                sessions_raw = input(
                    "- Max sessions per day: 1, 2, or 3 (balanced RPE) [default 1]: "
                ).strip()
            except EOFError:
                sessions_raw = ""
            if sessions_raw in ("2", "3"):
                state.max_sessions_per_day = int(sessions_raw)

            state.preferences_collected = True
            state.log(
                f"Preferences: max_weekly_minutes={state.max_weekly_minutes}, "
                f"days_off={state.days_off}, "
                f"preferred_quality_days={state.preferred_quality_days}, "
                f"max_sessions_per_day={state.max_sessions_per_day}"
            )

        result = _safe_call(
            "plan_generator",
            ["generate_training_plan", "generate_plan"],
            days=state.days,
            max_weekly_minutes=state.max_weekly_minutes,
            days_off=state.days_off,
            preferred_quality_days=state.preferred_quality_days,
            max_sessions_per_day=state.max_sessions_per_day,
            weekend_focus=state.weekend_focus,
        )

        if not isinstance(result, dict):
            raise ValueError("plan_generator returned non-dict result")

        # ważne: sessions, nie days
        if "sessions" not in result:
            raise ValueError("plan result does not contain 'sessions'")

        state.plan_draft = result
        state.explanation = result.get("explanation", state.coach_analysis)
        state.plan_accepted = False

        if state.messages:
            state.plan_draft = _enrich_plan_explanation_with_conversation(state, state.plan_draft)
            state.explanation = state.plan_draft.get("explanation", state.explanation)

        _save_json(DATA_DIR / "training_plan_current.json", state.plan_draft)
    except Exception as e:
        state.add_error(f"generate_plan failed: {e}")
        state.done = True
    return state


def display_plan_node(state: AgentState) -> AgentState:
    state.log("display_plan")

    if not state.plan_draft:
        state.add_error("display_plan called without plan_draft")
        state.done = True
        return state

    print()
    print(_format_plan_for_console(state.plan_draft))
    print()

    return state


def ask_user_node(state: AgentState) -> AgentState:
    state.log("ask_user")

    if not state.plan_draft:
        state.add_error("ask_user called without plan_draft")
        state.done = True
        return state
    # Tryb interaktywny w terminalu (bez GUI)
    if state.hitl_mode == "cli":
        print()
        print("=== TRAINING PLAN (CLI REVIEW) ===")
        print(_format_plan_for_console(state.plan_draft))
        print()
        try:
            answer = input("Accept this plan? [y/n] (ENTER = y): ").strip().lower()
        except EOFError:
            answer = "y"

        if answer in {"", "y", "yes"}:
            state.plan_accepted = True
            state.user_feedback = ""
            state.log("User accepted the plan (CLI).")
            return state

        try:
            feedback = input(
                "What should be changed in this plan? "
                "(leave empty to finish without acceptance):\n> "
            )
        except EOFError:
            feedback = ""

        if feedback.strip():
            state.plan_accepted = False
            state.user_feedback = feedback.strip()
            state.log(f"User feedback received (CLI): {state.user_feedback}")
            return state

        state.log("CLI: no accept / no feedback provided. Finishing.")
        state.done = True
        return state

    # Domyślnie – pełne okno Tkinter w formie dialogu z trenerem
    # jeśli nie ma historii (pierwsze odpalenie), pokaż inicjalne pytanie trenera
    if not state.coach_question and not state.dialog_history:
        state.coach_question = (
            "Co chcesz zmienić w tym planie? Napisz normalnie, np. "
            "'wolne we wtorek', 'środa jakościowa', 'w weekend mam czas', "
            "'maks 4 dni treningowe', 'max 2–3 treningi dziennie'."
        )

    try:
        from hitl_dialog import show_hitl_dialog
    except ImportError as e:
        state.add_error(
            "GUI Tkinter nie jest dostępne (brak tkinter). "
            f"Użyj --no-gui albo aplikacji Streamlit. Szczegóły: {e}"
        )
        state.done = True
        return state

    accepted, feedback = show_hitl_dialog(
        state.plan_draft,
        coach_question=state.coach_question,
        history=state.dialog_history,
    )

    if accepted:
        state.plan_accepted = True
        state.user_feedback = ""
        state.log("User accepted the plan.")
        return state

    if feedback.strip():
        state.plan_accepted = False
        state.user_feedback = feedback.strip()
        state.log(f"User feedback received: {state.user_feedback}")
        return state

    # Zamknięcie okna / brak decyzji => kończymy elegancko, bez pętli
    state.log("Dialog closed without accept/feedback. Finishing without acceptance.")
    state.done = True
    return state


def revise_plan_node(state: AgentState) -> AgentState:
    state.log("revise_plan")

    if not state.plan_draft:
        state.add_error("revise_plan called without plan_draft")
        state.done = True
        return state

    feedback = state.user_feedback.strip()
    if not feedback:
        state.log("No feedback present, skipping revision.")
        return state

    try:
        # Dopisz do historii dialogu
        state.dialog_history.append(("user", feedback))

        before_state = (
            state.max_weekly_minutes,
            tuple(state.days_off),
            tuple(state.preferred_quality_days),
            state.max_sessions_per_day,
            state.max_training_days,
            state.weekend_focus,
        )

        # 1) Spróbuj wywnioskować preferencje z naturalnego języka (bez "komend")
        # 2) Jeśli czegoś brakuje => dopytaj (coach_question)
        # 3) Jeśli preferencje się zmieniły => zregenruj plan z plan_generator
        import re

        text = feedback.lower()

        # max training days
        m = re.search(r"(?:maksymalnie|maks|max)\s*(\d+)\s*(?:dni|days)[^\n]{0,40}(?:trening|training)", text)
        if not m:
            m = re.search(r"(?:(\d+)\s*(?:dni|days)[^\n]{0,40}(?:na\s*trening|trening|training))", text)
        if m:
            try:
                state.max_training_days = int(m.group(1))
            except Exception:
                pass

        # max weekly minutes / hours (natural)
        m = re.search(r"(\d+)\s*(?:min|minut|minutes)\b", text)
        if m and any(x in text for x in ["tydzie", "week"]):
            try:
                state.max_weekly_minutes = int(m.group(1))
            except Exception:
                pass
        m = re.search(r"(\d+)\s*(?:h|godzin|hours)\b", text)
        if m and any(x in text for x in ["tydzie", "week"]):
            try:
                state.max_weekly_minutes = int(m.group(1)) * 60
            except Exception:
                pass

        # sessions per day
        if any(x in text for x in ["3 treningi dziennie", "trzy treningi dziennie", "3 sesje dziennie", "więcej niż 2 treningi"]):
            state.max_sessions_per_day = 3
        elif any(x in text for x in ["2 treningi dziennie", "dwa treningi dziennie", "2 sesje dziennie"]):
            state.max_sessions_per_day = 2

        # weekend focus (natural): "w weekend mam trochę czasu", "weekend mam czas", etc.
        if "weekend" in text and any(x in text for x in ["czas", "czasu", "woln", "luz"]):
            state.weekend_focus = True
        elif any(x in text for x in ["weekend mam czas", "mam czas w weekend", "więcej w weekend", "w weekend mam czas"]):
            state.weekend_focus = True

        # days off + quality days (Polish/English variants, including no-diacritics)
        day_aliases = {
            "monday": ["monday", "poniedziałek", "poniedzialek", "pon", "pn"],
            "tuesday": ["tuesday", "wtorek", "wto", "wt"],
            "wednesday": ["wednesday", "środa", "sroda", "srode", "środę", "srodę", "sr"],
            "thursday": ["thursday", "czwartek", "czw"],
            "friday": ["friday", "piątek", "piatek", "patek", "pt"],
            "saturday": ["saturday", "sobota", "sob"],
            "sunday": ["sunday", "niedziela", "ndz", "nie"],
        }

        def _mentioned_days() -> List[str]:
            out: List[str] = []
            for slug, aliases in day_aliases.items():
                if any(a in text for a in aliases):
                    out.append(slug)
            return out

        mentioned = _mentioned_days()
        if mentioned:
            # day off intent
            if any(x in text for x in ["wolne", "wolna", "wolny", "dzień wolny", "dzien wolny", "bez treningu", "odpoczynek", "off"]):
                state.days_off = sorted(set(state.days_off).union(set(mentioned)))
            # quality intent
            if any(x in text for x in ["jakości", "jakosci", "jakościow", "jakosciow", "quality", "interwa", "tempo", "threshold"]):
                state.preferred_quality_days = sorted(set(state.preferred_quality_days).union(set(mentioned)))

        # If user says "max X training days", we must know which days off
        if state.max_training_days is not None:
            need = 7 - state.max_training_days
            if need > 0 and len(state.days_off) < need:
                state.coach_question = (
                    f"OK. Chcesz trenować maksymalnie {state.max_training_days} dni w tygodniu. "
                    f"Potrzebuję {need} dni wolnych: które dni wybierasz?"
                )
                state.dialog_history.append(("coach", state.coach_question))
                state.user_feedback = ""
                state.plan_accepted = False
                return state

        after_state = (
            state.max_weekly_minutes,
            tuple(state.days_off),
            tuple(state.preferred_quality_days),
            state.max_sessions_per_day,
            state.max_training_days,
            state.weekend_focus,
        )

        prefs_changed = after_state != before_state

        # Jeśli preferencje się zmieniły, regeneruj plan (stabilniej niż "patchowanie")
        if prefs_changed:
            from plan_generator import generate_training_plan

            state.plan_draft = generate_training_plan(
                days=state.days,
                max_weekly_minutes=state.max_weekly_minutes,
                days_off=state.days_off,
                preferred_quality_days=state.preferred_quality_days,
                max_sessions_per_day=state.max_sessions_per_day,
                weekend_focus=state.weekend_focus,
            )
            state.explanation = state.plan_draft.get("explanation", state.explanation)
            _save_json(DATA_DIR / "training_plan_current.json", state.plan_draft)
            state.coach_question = "OK — ułożyłem plan pod Twoje preferencje. Co jeszcze dopracować?"
            state.dialog_history.append(("coach", state.coach_question))
        else:
            # fallback: revise plan via rules/LLM (for fine-grained edits)
            before = copy.deepcopy(state.plan_draft)
            revised = _apply_feedback_rules(state.plan_draft, feedback)

            state.plan_draft = revised
            state.explanation = revised.get("explanation", state.explanation)
            _save_json(DATA_DIR / "training_plan_current.json", state.plan_draft)

            changed = True
            try:
                changed = json.dumps(before, sort_keys=True, ensure_ascii=False) != json.dumps(
                    state.plan_draft, sort_keys=True, ensure_ascii=False
                )
            except Exception:
                changed = True

            if not changed:
                state.coach_question = (
                    "OK — powiedz proszę konkretniej, np.:\n"
                    "- które dni mają być wolne,\n"
                    "- który dzień ma być jakościowy,\n"
                    "- ile dni w tygodniu chcesz trenować,\n"
                    "- ile czasu masz w weekend."
                )
            else:
                state.coach_question = "Zaktualizowałem plan. Co jeszcze dopracować?"

            state.dialog_history.append(("coach", state.coach_question))
    except Exception as e:
        state.add_error(f"revise_plan failed: {e}")
        state.done = True
        return state

    # KLUCZOWE: czyścimy feedback, żeby nie wejść w pętlę revise -> think -> revise -> ...
    state.user_feedback = ""
    state.plan_accepted = False
    state.log("Plan revised and user_feedback cleared.")

    return state


def finish_node(state: AgentState) -> AgentState:
    # spróbujmy zapisać plan do CSV (Excel-friendly), jeśli istnieje
    if state.plan_draft:
        try:
            csv_path = _export_plan_to_csv(state.plan_draft)
            state.log(f"Plan exported to CSV: {csv_path}")
        except Exception as e:
            state.add_error(f"Failed to export plan to CSV: {e}")

    state.done = True
    state.next_action = "finish"
    state.log("finish")
    return state


def think_node(state: AgentState) -> AgentState:
    state.loop_count += 1
    state.log(f"think loop_count={state.loop_count}")

    if state.done:
        state.next_action = "finish"
        return state

    if state.errors:
        state.next_action = "finish"
        return state

    if state.loop_count > state.max_loops:
        state.add_error(f"Max loops reached ({state.max_loops}).")
        state.next_action = "finish"
        return state

    if not state.coach_analysis:
        state.next_action = "analyze_training"
        return state

    if state.mode == "full" and not state.coaching_brief_ready:
        state.next_action = "coaching_brief"
        return state

    if (
        state.mode == "full"
        and state.coaching_brief_ready
        and not state.dialog_complete
        and not state.plan_draft
    ):
        state.next_action = "coach_dialog"
        return state

    if not state.plan_draft:
        state.next_action = "generate_plan"
        return state

    if state.plan_accepted:
        state.next_action = "finish"
        return state

    if state.user_feedback.strip():
        state.next_action = "revise_plan"
        return state

    # Plan istnieje, nie jest accepted i nie ma feedbacku => pytamy usera
    state.next_action = "ask_user"
    return state


def route_after_think(state: AgentState) -> str:
    return state.next_action or "finish"