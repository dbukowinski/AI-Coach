from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from groq import Groq

from agent_state import AgentState


# ─── Validation maps ──────────────────────────────────────────────────────────

VALID_INTENTS = {
    "TRAINING_PLAN", "QUESTION", "ROUTE", "NUTRITION",
    "RACE", "CONTEXT_INFO", "GEAR",
}

VALID_SUBTYPES: Dict[str, set] = {
    "TRAINING_PLAN": {"PLAN_CREATE", "PLAN_MODIFY", "PLAN_REDUCE", "PLAN_EXTEND", "PLAN_EXPLAIN"},
    "QUESTION": {
        "Q_PACE", "Q_ZONES", "Q_LONG_RUN", "Q_INTERVALS", "Q_RECOVERY",
        "Q_ELEVATION", "Q_TERRAIN", "Q_VERTICAL", "Q_PACE_TRAIL", "Q_GEAR_TRAIL",
    },
    "ROUTE": {"ROUTE_REQUEST", "ROUTE_LOCATION", "ROUTE_SURFACE", "ROUTE_SAFETY"},
    "NUTRITION": {
        "NUTRITION_PRE", "NUTRITION_DURING", "NUTRITION_POST",
        "NUTRITION_RACE", "NUTRITION_WEIGHT",
    },
    "RACE": {"RACE_SELECT", "RACE_PREP", "RACE_STRATEGY", "RACE_POST", "RACE_TAPER"},
    "CONTEXT_INFO": {
        "INJURY", "ILLNESS", "FATIGUE", "LIFE_EVENT", "WEATHER", "PERSONAL_RECORD",
    },
    "GEAR": {
        "GEAR_SHOES_ROAD", "GEAR_SHOES_TRAIL", "GEAR_WATCH", "GEAR_PACK", "GEAR_POLES",
        "GEAR_CLOTHING_WEATHER", "GEAR_CLOTHING_SUMMER", "GEAR_LAYERING_TRAIL", "GEAR_RAIN_GEAR",
    },
}

FALLBACK: Dict[str, Any] = {
    "intent": "QUESTION",
    "subtype": "Q_LONG_RUN",
    "confidence": 0.5,
    "discipline": "both",
}

# ─── Classifier system prompt ─────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """Classify the user message into exactly one intent and one subtype.
Return ONLY valid JSON, no explanation, no markdown, no backticks.

Valid intents and subtypes:

TRAINING_PLAN: PLAN_CREATE, PLAN_MODIFY, PLAN_REDUCE, PLAN_EXTEND, PLAN_EXPLAIN
QUESTION: Q_PACE, Q_ZONES, Q_LONG_RUN, Q_INTERVALS, Q_RECOVERY, Q_ELEVATION, Q_TERRAIN, Q_VERTICAL, Q_PACE_TRAIL, Q_GEAR_TRAIL
ROUTE: ROUTE_REQUEST, ROUTE_LOCATION, ROUTE_SURFACE, ROUTE_SAFETY
NUTRITION: NUTRITION_PRE, NUTRITION_DURING, NUTRITION_POST, NUTRITION_RACE, NUTRITION_WEIGHT
RACE: RACE_SELECT, RACE_PREP, RACE_STRATEGY, RACE_POST, RACE_TAPER
CONTEXT_INFO: INJURY, ILLNESS, FATIGUE, LIFE_EVENT, WEATHER, PERSONAL_RECORD
GEAR: GEAR_SHOES_ROAD, GEAR_SHOES_TRAIL, GEAR_WATCH, GEAR_PACK, GEAR_POLES, GEAR_CLOTHING_WEATHER, GEAR_CLOTHING_SUMMER, GEAR_LAYERING_TRAIL, GEAR_RAIN_GEAR

Response format:
{
  "intent": "INTENT_NAME",
  "subtype": "SUBTYPE_NAME",
  "confidence": 0.0-1.0,
  "discipline": "road|trail|both"
}

Rules:
- discipline: "road" for road running, "trail" for mountains/trails, "both" if unclear
- If unsure about intent, default to QUESTION
- Never return anything outside the valid intents and subtypes listed above

--- FEW-SHOT EXAMPLES ---

User: "jak zrobić long run?"
{"intent": "QUESTION", "subtype": "Q_LONG_RUN", "confidence": 0.97, "discipline": "both"}

User: "co to jest tempo progowe?"
{"intent": "QUESTION", "subtype": "Q_PACE", "confidence": 0.95, "discipline": "both"}

User: "jak mierzyć strefy tętna?"
{"intent": "QUESTION", "subtype": "Q_ZONES", "confidence": 0.94, "discipline": "both"}

User: "ile przewyższenia powinienem robić tygodniowo?"
{"intent": "QUESTION", "subtype": "Q_ELEVATION", "confidence": 0.93, "discipline": "trail"}

User: "jak przeliczać tempo trailowe na płaskie?"
{"intent": "QUESTION", "subtype": "Q_PACE_TRAIL", "confidence": 0.92, "discipline": "trail"}

User: "zrób mi plan na maraton za 12 tygodni"
{"intent": "TRAINING_PLAN", "subtype": "PLAN_CREATE", "confidence": 0.98, "discipline": "road"}

User: "chcę w weekend góry, zmień plan"
{"intent": "TRAINING_PLAN", "subtype": "PLAN_MODIFY", "confidence": 0.96, "discipline": "trail"}

User: "za dużo kilometrów, ogranicz trochę"
{"intent": "TRAINING_PLAN", "subtype": "PLAN_REDUCE", "confidence": 0.95, "discipline": "both"}

User: "dodaj więcej podejść do planu"
{"intent": "TRAINING_PLAN", "subtype": "PLAN_EXTEND", "confidence": 0.93, "discipline": "trail"}

User: "dlaczego mam dziś interwały?"
{"intent": "TRAINING_PLAN", "subtype": "PLAN_EXPLAIN", "confidence": 0.91, "discipline": "both"}

User: "zaplanuj mi trasę 20 km z podejściami"
{"intent": "ROUTE", "subtype": "ROUTE_REQUEST", "confidence": 0.97, "discipline": "trail"}

User: "gdzie biegać w Zakopanem?"
{"intent": "ROUTE", "subtype": "ROUTE_LOCATION", "confidence": 0.96, "discipline": "trail"}

User: "czy bezpiecznie biegać solo po Tatrach?"
{"intent": "ROUTE", "subtype": "ROUTE_SAFETY", "confidence": 0.94, "discipline": "trail"}

User: "co jeść przed startem?"
{"intent": "NUTRITION", "subtype": "NUTRITION_PRE", "confidence": 0.97, "discipline": "both"}

User: "co zabrać do jedzenia na 4-godzinny bieg w górach?"
{"intent": "NUTRITION", "subtype": "NUTRITION_DURING", "confidence": 0.95, "discipline": "trail"}

User: "co jeść po długim biegu?"
{"intent": "NUTRITION", "subtype": "NUTRITION_POST", "confidence": 0.96, "discipline": "both"}

User: "jaka strategia żywieniowa na 50 km?"
{"intent": "NUTRITION", "subtype": "NUTRITION_RACE", "confidence": 0.94, "discipline": "trail"}

User: "mam maraton za 6 tygodni, co teraz?"
{"intent": "RACE", "subtype": "RACE_PREP", "confidence": 0.97, "discipline": "road"}

User: "jaki wyścig pasuje do mojego poziomu?"
{"intent": "RACE", "subtype": "RACE_SELECT", "confidence": 0.93, "discipline": "both"}

User: "kiedy zacząć tapering przed startem?"
{"intent": "RACE", "subtype": "RACE_TAPER", "confidence": 0.95, "discipline": "both"}

User: "co robić po maratonie?"
{"intent": "RACE", "subtype": "RACE_POST", "confidence": 0.94, "discipline": "road"}

User: "bolą mnie kolana od wczoraj"
{"intent": "CONTEXT_INFO", "subtype": "INJURY", "confidence": 0.97, "discipline": "both"}

User: "byłem chory 3 dni, nie biegałem"
{"intent": "CONTEXT_INFO", "subtype": "ILLNESS", "confidence": 0.96, "discipline": "both"}

User: "jestem bardzo zmęczony, nogi jak z ołowiu"
{"intent": "CONTEXT_INFO", "subtype": "FATIGUE", "confidence": 0.95, "discipline": "both"}

User: "mam delegację w przyszłym tygodniu"
{"intent": "CONTEXT_INFO", "subtype": "LIFE_EVENT", "confidence": 0.93, "discipline": "both"}

User: "zrobiłem PR na 10 km!"
{"intent": "CONTEXT_INFO", "subtype": "PERSONAL_RECORD", "confidence": 0.96, "discipline": "road"}

User: "jakie buty do maratonu polecasz?"
{"intent": "GEAR", "subtype": "GEAR_SHOES_ROAD", "confidence": 0.96, "discipline": "road"}

User: "jakie buty na mokre szlaki?"
{"intent": "GEAR", "subtype": "GEAR_SHOES_TRAIL", "confidence": 0.97, "discipline": "trail"}

User: "w co się ubrać na bieg przy 5°C i deszczu?"
{"intent": "GEAR", "subtype": "GEAR_CLOTHING_WEATHER", "confidence": 0.97, "discipline": "both"}

User: "co zakładać na bieganie w upał?"
{"intent": "GEAR", "subtype": "GEAR_CLOTHING_SUMMER", "confidence": 0.95, "discipline": "both"}

User: "jakie warstwy na bieg górski w zimie?"
{"intent": "GEAR", "subtype": "GEAR_LAYERING_TRAIL", "confidence": 0.94, "discipline": "trail"}

User: "czy warto wziąć kurtkę przeciwdeszczową na trail?"
{"intent": "GEAR", "subtype": "GEAR_RAIN_GEAR", "confidence": 0.93, "discipline": "trail"}

User: "jaki plecak na ultra?"
{"intent": "GEAR", "subtype": "GEAR_PACK", "confidence": 0.95, "discipline": "trail"}

User: "kiedy używać kijków?"
{"intent": "GEAR", "subtype": "GEAR_POLES", "confidence": 0.93, "discipline": "trail"}"""


# ─── Answer-first global rule ────────────────────────────────────────────────

ANSWER_FIRST_RULE = (
    "[ZASADA GLOBALNA]\n"
    "Najpierw odpowiedz na pytanie użytkownika.\n"
    "Jeśli potrzebujesz kontekstu — zadaj JEDNO pytanie, PO odpowiedzi.\n"
    "Nigdy nie pytaj o zmęczenie, cele tygodniowe ani obciążenie treningowe\n"
    "zanim nie odpiszesz na to co użytkownik zapytał.\n"
    "Dane ze Stravy używaj cicho do personalizacji — nigdy jako powód\n"
    "żeby opóźnić lub zastąpić odpowiedź.\n"
)

# ─── LLM helpers ─────────────────────────────────────────────────────────────

def _call_llm(user_prompt: str, system_prompt: str = "", max_tokens: int = 150) -> str:
    model = os.getenv("GROQ_MODEL_ID", "llama-3.3-70b-versatile")
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=max_tokens,
        top_p=0.9,
    )
    return response.choices[0].message.content


def _safe_call_llm(
    user_prompt: str,
    system_prompt: str = "",
    max_tokens: int = 400,
    state: Optional[AgentState] = None,
) -> str:
    try:
        return _call_llm(user_prompt, system_prompt=system_prompt, max_tokens=max_tokens)
    except Exception as e:
        msg = f"[IntentClassifier] handler LLM failed: {e}"
        if state:
            state.log(msg)
        else:
            print(msg)
        return ""


# ─── Core classification ──────────────────────────────────────────────────────

def classify_intent(message: str) -> Dict[str, Any]:
    """Call Groq to classify message. Returns dict with intent/subtype/confidence/discipline."""
    try:
        raw = _call_llm(message, system_prompt=CLASSIFIER_SYSTEM_PROMPT, max_tokens=150)
        text = raw.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(text)

        intent = str(data.get("intent", ""))
        subtype = str(data.get("subtype", ""))
        confidence = float(data.get("confidence", 0.5))
        discipline = str(data.get("discipline", "both"))

        if intent not in VALID_INTENTS:
            print(f"[IntentClassifier] FALLBACK: invalid intent '{intent}' for: {message!r}")
            return FALLBACK.copy()

        if subtype not in VALID_SUBTYPES.get(intent, set()):
            print(f"[IntentClassifier] FALLBACK: invalid subtype '{subtype}' (intent={intent}) for: {message!r}")
            return FALLBACK.copy()

        if confidence < 0.4:
            print(f"[IntentClassifier] FALLBACK: low confidence {confidence:.2f} for: {message!r}")
            return {**FALLBACK, "discipline": discipline}

        return {"intent": intent, "subtype": subtype, "confidence": confidence, "discipline": discipline}

    except json.JSONDecodeError as e:
        print(f"[IntentClassifier] FALLBACK: JSON parse error for: {message!r} — {e}")
        return FALLBACK.copy()
    except Exception as e:
        print(f"[IntentClassifier] FALLBACK: unexpected error for: {message!r} — {e}")
        return FALLBACK.copy()


# ─── Classifier node ──────────────────────────────────────────────────────────

def intent_classifier_node(state: AgentState) -> AgentState:
    message = (state.user_feedback or state.user_message or "").strip()

    if not message:
        state.log("intent_classifier: empty message, defaulting to QUESTION/Q_LONG_RUN")
        state.intent = "QUESTION"
        state.subtype = "Q_LONG_RUN"
        state.discipline = "both"
        state.intent_confidence = 0.5
        return state

    state.user_message = message
    result = classify_intent(message)

    state.intent = result["intent"]
    state.subtype = result["subtype"]
    state.discipline = result.get("discipline", "both")
    state.intent_confidence = result["confidence"]

    state.log(
        f"intent_classifier: {state.intent}/{state.subtype} "
        f"(confidence={state.intent_confidence:.2f}, discipline={state.discipline})"
    )
    return state


def route_after_intent(state: AgentState) -> str:
    intent = state.intent or "QUESTION"
    mapping = {
        "TRAINING_PLAN": "handle_training_plan",
        "QUESTION": "handle_question",
        "ROUTE": "handle_route",
        "NUTRITION": "handle_nutrition",
        "RACE": "handle_race",
        "CONTEXT_INFO": "handle_context_info",
        "GEAR": "handle_gear",
    }
    return mapping.get(intent, "handle_question")


# ─── Shared state helper ──────────────────────────────────────────────────────

def _append_response(state: AgentState, response: str) -> None:
    """Push handler response into dialog history, messages, and coach_question."""
    if not response:
        return
    state.dialog_history.append(("coach", response))
    state.messages.append({"role": "assistant", "content": response})
    state.coach_question = response


# ─── 7 intent handler nodes ───────────────────────────────────────────────────

def handle_question_node(state: AgentState) -> AgentState:
    """Answer factually, max 4 sentences. Do NOT modify the training plan."""
    state.log(f"handle_question subtype={state.subtype}")
    message = state.user_message or state.user_feedback

    system_prompt = (
        ANSWER_FIRST_RULE
        + "Użytkownik zadał pytanie merytoryczne. Jesteś ekspertem w bieganiu.\n"
        "Zasady:\n"
        "- Odpowiedz w max 4 zdaniach. Nie pytaj o nic przed odpowiedzią.\n"
        "- Nie modyfikuj planu. Nie sprawdzaj obciążenia treningowego.\n"
        "- Nie pytaj o cele tygodniowe ani samopoczucie.\n"
        "- Jeśli pytanie dotyczy dyscypliny której nie znasz — odpowiedz ogólnie.\n"
        "- Odpowiedź po polsku.\n"
        f"- Kontekst pytania: {state.subtype}\n"
        f"- Dyscyplina: {state.discipline or 'both'}"
    )

    response = _safe_call_llm(message, system_prompt=system_prompt, max_tokens=300, state=state)
    if not response:
        response = "Przepraszam, nie mogę teraz odpowiedzieć na to pytanie. Spróbuj ponownie."

    _append_response(state, response)
    state.user_feedback = ""  # prevent think → revise_plan from triggering
    return state


def handle_training_plan_node(state: AgentState) -> AgentState:
    """Route plan modification requests. PLAN_EXPLAIN answers inline; others pass to revise_plan."""
    state.log(f"handle_training_plan subtype={state.subtype}")
    message = state.user_message or state.user_feedback

    if state.subtype == "PLAN_EXPLAIN":
        plan_snippet = ""
        if state.plan_draft:
            plan_snippet = "\nFragment planu: " + json.dumps(
                state.plan_draft.get("sessions", [])[:3], ensure_ascii=False
            )[:400]

        system_prompt = (
            ANSWER_FIRST_RULE
            + "Jesteś trenerem biegowym. Wyjaśniasz konkretną sesję lub element planu treningowego.\n"
            "Odpowiedz po polsku, max 4 zdania. Nie modyfikuj planu.\n"
            "Nie pytaj o zmęczenie ani obciążenie przed wyjaśnieniem."
        )
        response = _safe_call_llm(
            message + plan_snippet, system_prompt=system_prompt, max_tokens=300, state=state
        )
        if not response:
            response = "Ta sesja jest częścią progresji — buduje konkretną cechę fizyczną potrzebną do realizacji Twojego celu."
        _append_response(state, response)
        state.user_feedback = ""  # explanation only, no revision
    else:
        # PLAN_CREATE / PLAN_MODIFY / PLAN_REDUCE / PLAN_EXTEND
        # Keep user_feedback intact so think → revise_plan handles the actual change.
        state.dialog_history.append(("user", message))

    return state


def handle_route_node(state: AgentState) -> AgentState:
    state.log(f"handle_route subtype={state.subtype}")
    message = state.user_message or state.user_feedback

    system_prompt = (
        ANSWER_FIRST_RULE
        + "Jesteś ekspertem w planowaniu tras biegowych. Odpowiadasz po polsku.\n"
        f"Typ zapytania: {state.subtype}. Dyscyplina: {state.discipline or 'both'}.\n"
        "Jeśli lokalizacja jest podana — zasugeruj trasy od razu.\n"
        "Jeśli brakuje lokalizacji — zapytaj TYLKO o lokalizację, nic więcej.\n"
        "Nie pytaj o samopoczucie ani stan zdrowia."
    )

    response = _safe_call_llm(message, system_prompt=system_prompt, max_tokens=350, state=state)
    if not response:
        response = "Aby zaplanować trasę, podaj lokalizację i preferowany dystans."

    _append_response(state, response)
    state.user_feedback = ""
    return state


def handle_nutrition_node(state: AgentState) -> AgentState:
    state.log(f"handle_nutrition subtype={state.subtype}")
    message = state.user_message or state.user_feedback

    system_prompt = (
        ANSWER_FIRST_RULE
        + "Jesteś ekspertem w żywieniu sportowym dla biegaczy. Odpowiadasz po polsku.\n"
        f"Kontekst: {state.subtype}. Dyscyplina: {state.discipline or 'both'}.\n"
        "Odpowiedz bezpośrednio na pytanie. Nie wspominaj o obciążeniu treningowym ani zmęczeniu."
    )

    response = _safe_call_llm(message, system_prompt=system_prompt, max_tokens=350, state=state)
    if not response:
        response = "Nie udało się udzielić odpowiedzi na pytanie żywieniowe. Spróbuj ponownie."

    _append_response(state, response)
    state.user_feedback = ""
    return state


def handle_race_node(state: AgentState) -> AgentState:
    state.log(f"handle_race subtype={state.subtype}")
    message = state.user_message or state.user_feedback

    system_prompt = (
        ANSWER_FIRST_RULE
        + "Jesteś ekspertem w przygotowaniach startowych i strategii wyścigowej. Odpowiadasz po polsku.\n"
        f"Kontekst: {state.subtype}. Dyscyplina: {state.discipline or 'both'}.\n"
        "Odpowiedz bezpośrednio. Nie pytaj o zdrowie ani samopoczucie przed odpowiedzią.\n"
        "Jeśli brakuje daty startu — zapytaj TYLKO o datę startu, nic więcej."
    )

    response = _safe_call_llm(message, system_prompt=system_prompt, max_tokens=350, state=state)
    if not response:
        response = "Nie udało się odpowiedzieć na pytanie dotyczące wyścigu. Spróbuj ponownie."

    _append_response(state, response)
    state.user_feedback = ""
    return state


def handle_context_info_node(state: AgentState) -> AgentState:
    """Acknowledge what user shared, ask ONE question about what to do. Never auto-generate plan."""
    state.log(f"handle_context_info subtype={state.subtype}")
    message = state.user_message or state.user_feedback

    key = (state.subtype or "info").lower()
    state.extracted_context[key] = message

    system_prompt = (
        "[ZASADA GLOBALNA]\n"
        "Odpowiedz dokładnie w dwóch zdaniach:\n"
        "1. Zacznij od 'Rozumiem' i sparafrazuj krótko co napisał użytkownik.\n"
        "2. Zadaj JEDNO pytanie: co użytkownik chce zrobić z planem treningowym "
        "(np. przesunąć trening, zmodyfikować plan, dostosować intensywność).\n"
        "Nie generuj planu. Nie sugeruj zmian samodzielnie. Tylko potwierdź i zapytaj o intencję.\n"
        "Odpowiedź po polsku."
    )
    response = _safe_call_llm(message, system_prompt=system_prompt, max_tokens=150, state=state)

    if not response:
        fallback_responses = {
            "INJURY": "Rozumiem — zanotowałem uraz. Co chcesz z tym zrobić — przesunąć trening czy zmodyfikować plan?",
            "ILLNESS": "Rozumiem — byłeś/aś chory/a. Co chcesz z tym zrobić — przesunąć trening czy zmodyfikować plan?",
            "FATIGUE": "Rozumiem — czujesz zmęczenie. Co chcesz z tym zrobić — złagodzić intensywność czy przesunąć trening?",
            "LIFE_EVENT": "Zanotowałem zmianę w Twoich planach. Co chcesz z tym zrobić — przesunąć trening czy dostosować plan?",
            "WEATHER": "Rozumiem — warunki pogodowe wpłyną na trening. Co chcesz z tym zrobić — dostosować plan czy zmienić trasę?",
            "PERSONAL_RECORD": "Rozumiem — osiągnąłeś/aś nowy rekord, gratulacje! Czy chcesz zaktualizować strefy treningowe na tej podstawie?",
        }
        response = fallback_responses.get(
            state.subtype or "", "Rozumiem. Co chcesz teraz zrobić z planem treningowym?"
        )

    _append_response(state, response)
    state.user_feedback = ""  # CRITICAL: do NOT auto-generate plan
    return state


def handle_gear_node(state: AgentState) -> AgentState:
    state.log(f"handle_gear subtype={state.subtype}")
    message = state.user_message or state.user_feedback

    system_prompt = (
        ANSWER_FIRST_RULE
        + "Użytkownik pyta o sprzęt lub ubranie. Jesteś ekspertem w sprzęcie biegowym.\n"
        f"Typ pytania: {state.subtype}. Dyscyplina: {state.discipline or 'both'}.\n"
        "Zasady:\n"
        "- Odpowiedz bezpośrednio. Jeśli temperatura jest podana — użyj jej od razu.\n"
        "- Nie pytaj o cele tygodniowe. Nie analizuj Stravy.\n"
        "- Jeśli potrzebujesz dokładnie jednej brakującej informacji "
        "(np. temperatura, teren) — zapytaj o NIĄ JEDNĄ, po udzieleniu ogólnej odpowiedzi.\n"
        "- Jeśli dyscyplina nieznana — odpowiedz dla trail.\n"
        "- Odpowiedź po polsku."
    )

    response = _safe_call_llm(message, system_prompt=system_prompt, max_tokens=350, state=state)
    if not response:
        response = "Nie udało się odpowiedzieć na pytanie o sprzęt. Spróbuj ponownie."

    _append_response(state, response)
    state.user_feedback = ""
    return state
