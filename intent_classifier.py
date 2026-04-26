from __future__ import annotations

import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional

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
        "Q_OTHER",
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
    "subtype": "Q_OTHER",
    "confidence": 0.5,
    "discipline": "both",
}

# ─── Classifier system prompt ─────────────────────────────────────────────────

CLASSIFIER_SYSTEM_PROMPT = """You receive a conversation (last up to 3 messages) and must classify the LAST user message.
Return ONLY valid JSON, no explanation, no markdown, no backticks.

Valid intents and subtypes:

TRAINING_PLAN: PLAN_CREATE, PLAN_MODIFY, PLAN_REDUCE, PLAN_EXTEND, PLAN_EXPLAIN
QUESTION: Q_PACE, Q_ZONES, Q_LONG_RUN, Q_INTERVALS, Q_RECOVERY, Q_ELEVATION, Q_TERRAIN, Q_VERTICAL, Q_PACE_TRAIL, Q_GEAR_TRAIL, Q_OTHER
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
- If unsure about intent, default to QUESTION / Q_OTHER
- Never return anything outside the valid intents and subtypes listed above

SPECIAL CASES — classify as QUESTION / Q_OTHER:
- User expresses frustration with the agent ("coś mylisz", "nie rozumiesz", "źle odpowiedziałeś", "pomyliłeś")
- User corrects the agent about topic ("o to mi nie chodziło", "zapytałem po prostu o X")

SHORT CONTEXTUAL REPLIES — use conversation history to classify, do NOT treat as frustration:
- "ok", "tak", "nie", "nie zupełnie nie" are normal conversational answers, NOT frustration
- If previous agent message asked about fatigue/injury/illness → short reply maps to that CONTEXT_INFO subtype
  Example: agent asks "Czy czujesz się zmęczony?" → user replies "nie, zupełnie nie"
  → CONTEXT_INFO / FATIGUE (user is answering the fatigue question, denying it)
- If previous agent message asked about plan → "ok" = TRAINING_PLAN / PLAN_MODIFY
- If context is unclear → QUESTION / Q_OTHER

NEVER classify as FATIGUE:
- Expressions of frustration with the AI ("coś mylisz", "nie rozumiesz", "źle odpowiedziałeś")
- Corrections of agent misunderstanding when not about physical state

FATIGUE is ONLY for explicit physical statements OR direct answers to fatigue questions:
- "jestem zmęczony po treningu" / "jestem wyczerpany"
- "bolą mnie nogi" / "mam zakwasy"
- "nie mam siły biegać" / "nogi jak z ołowiu"
- "nie, zupełnie nie" as answer to "Czy czujesz się zmęczony?" → CONTEXT_INFO / FATIGUE

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

Conversation: assistant: "Czy czujesz się zmęczony?" / user: "nie, zupełnie nie"
{"intent": "CONTEXT_INFO", "subtype": "FATIGUE", "confidence": 0.85, "discipline": "both"}

Conversation: assistant: "Czy czujesz się zmęczony?" / user: "ok, trochę"
{"intent": "CONTEXT_INFO", "subtype": "FATIGUE", "confidence": 0.80, "discipline": "both"}

Conversation: unclear / user: "ok"
{"intent": "QUESTION", "subtype": "Q_OTHER", "confidence": 0.60, "discipline": "both"}

User: "znów coś mylisz"
{"intent": "QUESTION", "subtype": "Q_OTHER", "confidence": 0.85, "discipline": "both"}

User: "mówiłem że nie"
{"intent": "QUESTION", "subtype": "Q_OTHER", "confidence": 0.80, "discipline": "both"}

User: "jestem bardzo zmęczony, nogi jak z ołowiu"
{"intent": "CONTEXT_INFO", "subtype": "FATIGUE", "confidence": 0.95, "discipline": "both"}

User: "jestem zmęczony po wczorajszym treningu"
{"intent": "CONTEXT_INFO", "subtype": "FATIGUE", "confidence": 0.95, "discipline": "both"}

User: "zrób mi plan na maraton za 12 tygodni"
{"intent": "TRAINING_PLAN", "subtype": "PLAN_CREATE", "confidence": 0.98, "discipline": "road"}

User: "chcę w weekend góry, zmień plan"
{"intent": "TRAINING_PLAN", "subtype": "PLAN_MODIFY", "confidence": 0.96, "discipline": "trail"}

User: "kontynuujmy wzrost loadu"
{"intent": "TRAINING_PLAN", "subtype": "PLAN_CREATE", "confidence": 0.93, "discipline": "both"}

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

User: "mam delegację w przyszłym tygodniu"
{"intent": "CONTEXT_INFO", "subtype": "LIFE_EVENT", "confidence": 0.93, "discipline": "both"}

User: "zrobiłem PR na 10 km!"
{"intent": "CONTEXT_INFO", "subtype": "PERSONAL_RECORD", "confidence": 0.96, "discipline": "road"}

User: "co ubrać na 9 stopni?"
{"intent": "GEAR", "subtype": "GEAR_CLOTHING_WEATHER", "confidence": 0.97, "discipline": "both"}

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

    last_exc: Exception = RuntimeError("no attempts made")
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.9,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_exc = e
            if attempt < 2:
                delay = 2 ** attempt  # 1s then 2s
                print(f"[RUS-25] Groq retry {attempt + 1}/3 after {delay}s: {type(e).__name__}: {e}")
                time.sleep(delay)
    raise last_exc


def _safe_call_llm(
    user_prompt: str,
    system_prompt: str = "",
    max_tokens: int = 400,
    state: Optional[AgentState] = None,
) -> str:
    try:
        return _call_llm(user_prompt, system_prompt=system_prompt, max_tokens=max_tokens)
    except Exception as e:
        msg = f"[RUS-25] AGENT ERROR: {type(e).__name__}: {e}"
        if state:
            state.log(msg)
        else:
            print(msg)
        traceback.print_exc()
        return ""


# ─── Core classification ──────────────────────────────────────────────────────

def classify_intent(
    message: str,
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Call Groq to classify message. Returns dict with intent/subtype/confidence/discipline.
    Passes last 3 conversation messages as context so short replies are classified correctly.
    """
    try:
        last_3 = (messages or [])[-3:]
        if last_3:
            context = "\n".join(f"{m['role']}: {m['content']}" for m in last_3)
            full_input = (
                f"Conversation so far:\n{context}\n\n"
                f"Classify the LAST user message only: {message!r}"
            )
        else:
            full_input = message

        raw = _call_llm(full_input, system_prompt=CLASSIFIER_SYSTEM_PROMPT, max_tokens=150)
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
            print(f"FALLBACK: invalid intent '{intent}' for: {message!r} — raw: {raw!r}")
            return FALLBACK.copy()

        if subtype not in VALID_SUBTYPES.get(intent, set()):
            print(f"FALLBACK: invalid subtype '{subtype}' (intent={intent}) for: {message!r} — raw: {raw!r}")
            return FALLBACK.copy()

        if confidence < 0.4:
            print(f"FALLBACK: low confidence {confidence:.2f} for: {message!r}")
            return {**FALLBACK, "discipline": discipline}

        return {"intent": intent, "subtype": subtype, "confidence": confidence, "discipline": discipline}

    except json.JSONDecodeError as e:
        print(f"FALLBACK: JSON parse error for: {message!r} — {e} — raw: {raw!r}")
        return FALLBACK.copy()
    except Exception as e:
        print(f"FALLBACK: unexpected error for: {message!r} — {e}")
        return FALLBACK.copy()


# ─── Classifier node ──────────────────────────────────────────────────────────

def intent_classifier_node(state: AgentState) -> AgentState:
    message = (state.user_feedback or state.user_message or "").strip()

    if not message:
        state.log("intent_classifier: empty message, defaulting to QUESTION/Q_OTHER")
        state.intent = "QUESTION"
        state.subtype = "Q_OTHER"
        state.discipline = "both"
        state.intent_confidence = 0.5
        return state

    state.user_message = message
    result = classify_intent(message, messages=state.messages)

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


# ─── Human-readable subtype hints (never leak raw subtype codes to LLM) ─────

# ─── Per-subtype prompt instructions (subtype codes never enter LLM strings) ──

QUESTION_PROMPTS: Dict[str, str] = {
    "Q_LONG_RUN":   "Użytkownik pyta jak wykonać długi bieg. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_PACE":       "Użytkownik pyta o tempo biegu. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_ZONES":      "Użytkownik pyta o strefy tętna. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_INTERVALS":  "Użytkownik pyta o interwały. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_RECOVERY":   "Użytkownik pyta o regenerację. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_ELEVATION":  "Użytkownik pyta o przewyższenia w treningu. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_TERRAIN":    "Użytkownik pyta o technikę biegu w terenie. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_VERTICAL":   "Użytkownik pyta o trening podejść. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_PACE_TRAIL": "Użytkownik pyta o tempo trailowe. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_GEAR_TRAIL": "Użytkownik pyta o sprzęt trailowy. Odpowiedz merytorycznie w max 4 zdaniach.",
    "Q_OTHER":      "Użytkownik zadał pytanie treningowe. Odpowiedz merytorycznie w max 4 zdaniach.",
}

GEAR_PROMPTS: Dict[str, str] = {
    "GEAR_SHOES_ROAD":       "Użytkownik pyta o buty do biegania na drodze. Odpowiedz bezpośrednio.",
    "GEAR_SHOES_TRAIL":      "Użytkownik pyta o buty trailowe. Odpowiedz bezpośrednio.",
    "GEAR_CLOTHING_WEATHER": "Użytkownik pyta co ubrać na bieg przy danej pogodzie. Odpowiedz bezpośrednio.",
    "GEAR_CLOTHING_SUMMER":  "Użytkownik pyta o strój na upał. Odpowiedz bezpośrednio.",
    "GEAR_LAYERING_TRAIL":   "Użytkownik pyta o warstwowanie ubrań na trail. Odpowiedz bezpośrednio.",
    "GEAR_RAIN_GEAR":        "Użytkownik pyta o sprzęt na deszcz. Odpowiedz bezpośrednio.",
    "GEAR_PACK":             "Użytkownik pyta o plecak biegowy. Odpowiedz bezpośrednio.",
    "GEAR_POLES":            "Użytkownik pyta o kijki biegowe. Odpowiedz bezpośrednio.",
    "GEAR_WATCH":            "Użytkownik pyta o zegarek sportowy. Odpowiedz bezpośrednio.",
}

ROUTE_PROMPTS: Dict[str, str] = {
    "ROUTE_REQUEST":  "Użytkownik prosi o zaplanowanie trasy biegowej. Zasugeruj trasę lub zapytaj o lokalizację.",
    "ROUTE_LOCATION": "Użytkownik pyta gdzie biegać w danej lokalizacji. Podaj konkretne miejsca.",
    "ROUTE_SURFACE":  "Użytkownik pyta o nawierzchnię lub typ terenu. Odpowiedz bezpośrednio.",
    "ROUTE_SAFETY":   "Użytkownik pyta o bezpieczeństwo na trasie. Odpowiedz bezpośrednio.",
}

NUTRITION_PROMPTS: Dict[str, str] = {
    "NUTRITION_PRE":    "Użytkownik pyta o żywienie przed treningiem lub startem. Odpowiedz bezpośrednio.",
    "NUTRITION_DURING": "Użytkownik pyta o jedzenie i picie podczas biegu. Odpowiedz bezpośrednio.",
    "NUTRITION_POST":   "Użytkownik pyta o żywienie po treningu i regenerację żywieniową. Odpowiedz bezpośrednio.",
    "NUTRITION_RACE":   "Użytkownik pyta o strategię żywieniową na wyścig. Odpowiedz bezpośrednio.",
    "NUTRITION_WEIGHT": "Użytkownik pyta o wagę lub kontrolę masy ciała. Odpowiedz bezpośrednio.",
}

RACE_PROMPTS: Dict[str, str] = {
    "RACE_SELECT":   "Użytkownik pyta o wybór wyścigu odpowiedniego do jego poziomu. Odpowiedz bezpośrednio.",
    "RACE_PREP":     "Użytkownik pyta o przygotowanie do nadchodzącego wyścigu. Odpowiedz bezpośrednio.",
    "RACE_STRATEGY": "Użytkownik pyta o strategię wyścigową. Odpowiedz bezpośrednio.",
    "RACE_POST":     "Użytkownik pyta o regenerację i analizę po wyścigu. Odpowiedz bezpośrednio.",
    "RACE_TAPER":    "Użytkownik pyta o tapering przed startem. Odpowiedz bezpośrednio.",
}

# ─── 7 intent handler nodes ───────────────────────────────────────────────────

def handle_question_node(state: AgentState) -> AgentState:
    """Answer factually, max 4 sentences. Do NOT modify the training plan."""
    state.log(f"handle_question subtype={state.subtype}")
    message = state.user_message or state.user_feedback

    subtype_instruction = QUESTION_PROMPTS.get(state.subtype or "", QUESTION_PROMPTS["Q_OTHER"])
    system_prompt = (
        ANSWER_FIRST_RULE
        + subtype_instruction + "\n"
        "Zasady:\n"
        "- Nie pytaj o nic przed odpowiedzią.\n"
        "- Nie modyfikuj planu. Nie sprawdzaj obciążenia treningowego.\n"
        "- Nie pytaj o cele tygodniowe ani samopoczucie.\n"
        "- Jeśli pytanie dotyczy dyscypliny której nie znasz — odpowiedz ogólnie.\n"
        "- Odpowiedz po polsku."
    )

    response = _safe_call_llm(message, system_prompt=system_prompt, max_tokens=300, state=state)
    if not response:
        response = "Coś poszło nie tak po stronie AI — napisz ponownie, a odpowiem."

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

    route_instruction = ROUTE_PROMPTS.get(state.subtype or "", "Użytkownik pyta o trasę biegową. Odpowiedz bezpośrednio.")
    system_prompt = (
        ANSWER_FIRST_RULE
        + route_instruction + "\n"
        "Jeśli lokalizacja jest podana — zasugeruj trasy od razu.\n"
        "Jeśli brakuje lokalizacji — zapytaj TYLKO o lokalizację, nic więcej.\n"
        "Nie pytaj o samopoczucie ani stan zdrowia.\n"
        "Odpowiedz po polsku."
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

    nutrition_instruction = NUTRITION_PROMPTS.get(state.subtype or "", "Użytkownik pyta o żywienie sportowe. Odpowiedz bezpośrednio.")
    system_prompt = (
        ANSWER_FIRST_RULE
        + nutrition_instruction + "\n"
        "Jesteś ekspertem w żywieniu sportowym dla biegaczy.\n"
        "Nie wspominaj o obciążeniu treningowym ani zmęczeniu.\n"
        "Odpowiedz po polsku."
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

    race_instruction = RACE_PROMPTS.get(state.subtype or "", "Użytkownik pyta o wyścig lub start. Odpowiedz bezpośrednio.")
    system_prompt = (
        ANSWER_FIRST_RULE
        + race_instruction + "\n"
        "Jesteś ekspertem w przygotowaniach startowych i strategii wyścigowej.\n"
        "Nie pytaj o zdrowie ani samopoczucie przed odpowiedzią.\n"
        "Jeśli brakuje daty startu — zapytaj TYLKO o datę startu, nic więcej.\n"
        "Odpowiedz po polsku."
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
            "FATIGUE": "Rozumiem — zanotowałem Twój aktualny poziom zmęczenia. Co chcesz z tym zrobić — dostosować plan czy kontynuować jak jest?",
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

    gear_instruction = GEAR_PROMPTS.get(state.subtype or "", "Użytkownik pyta o sprzęt biegowy. Odpowiedz bezpośrednio.")
    system_prompt = (
        ANSWER_FIRST_RULE
        + gear_instruction + "\n"
        "Jesteś ekspertem w sprzęcie biegowym.\n"
        "Zasady:\n"
        "- Jeśli temperatura jest podana — użyj jej od razu.\n"
        "- Nie pytaj o cele tygodniowe. Nie analizuj Stravy.\n"
        "- Jeśli potrzebujesz jednej brakującej informacji (np. temperatura, teren) "
        "— zapytaj o NIĄ JEDNĄ, po udzieleniu ogólnej odpowiedzi.\n"
        "- Jeśli dyscyplina nieznana — odpowiedz dla trail.\n"
        "- Odpowiedz po polsku."
    )

    response = _safe_call_llm(message, system_prompt=system_prompt, max_tokens=350, state=state)
    if not response:
        response = "Nie udało się odpowiedzieć na pytanie o sprzęt. Spróbuj ponownie."

    _append_response(state, response)
    state.user_feedback = ""
    return state
