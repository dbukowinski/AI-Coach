"""
Tests for RUS-19: answer-first behavior across all 7 intent handlers.

Verifies that:
- Every handler has the global answer-first rule in its system prompt
- No handler asks about fatigue/load before answering
- CONTEXT_INFO is the only intent that asks before acting
- TRAINING_PLAN PLAN_CREATE passes through without an LLM question first

All tests mock _safe_call_llm — no real API key required.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from agent_state import AgentState
from intent_classifier import (
    ANSWER_FIRST_RULE,
    handle_context_info_node,
    handle_gear_node,
    handle_nutrition_node,
    handle_question_node,
    handle_race_node,
    handle_route_node,
    handle_training_plan_node,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_state(message: str, intent: str, subtype: str, discipline: str = "both") -> AgentState:
    state = AgentState()
    state.user_feedback = message
    state.user_message = message
    state.intent = intent
    state.subtype = subtype
    state.discipline = discipline
    return state


def _captured_system_prompt(mock_llm) -> str:
    """Extract the system_prompt kwarg from the first _safe_call_llm call."""
    assert mock_llm.called, "_safe_call_llm was not called"
    return mock_llm.call_args.kwargs.get("system_prompt", "")


# ─── Test 1: QUESTION — "Jak zrobić long run" ─────────────────────────────────

def test_question_long_run_answers_first():
    """First sentence must be a factual answer, not a health/load question."""
    state = _make_state("Jak zrobić long run", "QUESTION", "Q_LONG_RUN")
    direct_answer = "Long run to bieg w tempie konwersacyjnym trwający 60–90 minut, budujący bazę tlenową."

    with patch("intent_classifier._safe_call_llm", return_value=direct_answer) as mock_llm:
        result = handle_question_node(state)

    sys_prompt = _captured_system_prompt(mock_llm)
    assert "Najpierw odpowiedz" in sys_prompt, "Global answer-first rule missing from QUESTION prompt"
    assert result.coach_question == direct_answer, "Response must be the direct answer"
    assert result.plan_draft == {}, "Plan must not be touched"
    assert result.user_feedback == "", "user_feedback must be cleared"


def test_question_system_prompt_forbids_fatigue_and_load_check():
    """QUESTION system prompt must not instruct the LLM to ask about fatigue or load."""
    state = _make_state("Jak zrobić long run", "QUESTION", "Q_LONG_RUN")

    with patch("intent_classifier._safe_call_llm", return_value="Odpowiedź.") as mock_llm:
        handle_question_node(state)

    sys_prompt = _captured_system_prompt(mock_llm).lower()
    assert "zmęczenie" not in sys_prompt, "QUESTION prompt must not ask about fatigue"
    assert "obciążenie" not in sys_prompt, "QUESTION prompt must not check training load"


# ─── Test 2: NUTRITION — "Co jeść przed startem?" ─────────────────────────────

def test_nutrition_pre_answers_directly():
    """Nutrition handler must answer immediately — no health-check preamble."""
    state = _make_state("Co jeść przed startem?", "NUTRITION", "NUTRITION_PRE")
    answer = "Przed startem zjedz węglowodany o niskim IG 3 godziny wcześniej, unikaj błonnika."

    with patch("intent_classifier._safe_call_llm", return_value=answer) as mock_llm:
        result = handle_nutrition_node(state)

    sys_prompt = _captured_system_prompt(mock_llm)
    assert "Najpierw odpowiedz" in sys_prompt
    assert result.coach_question == answer
    assert result.user_feedback == ""


def test_nutrition_system_prompt_has_no_load_preamble():
    """NUTRITION system prompt must not mention training load or fatigue."""
    state = _make_state("Co jeść przed startem?", "NUTRITION", "NUTRITION_PRE")

    with patch("intent_classifier._safe_call_llm", return_value="Odpowiedź.") as mock_llm:
        handle_nutrition_node(state)

    sys_prompt = _captured_system_prompt(mock_llm).lower()
    assert "obciążenie" not in sys_prompt
    assert "zmęczenie" not in sys_prompt


# ─── Test 3: GEAR — "Jakie buty na mokre szlaki?" ─────────────────────────────

def test_gear_trail_answers_directly():
    """Gear handler must give a direct recommendation without asking about health first."""
    state = _make_state("Jakie buty na mokre szlaki?", "GEAR", "GEAR_SHOES_TRAIL", discipline="trail")
    answer = "Na mokre szlaki polecam buty z agresywną bieżnikiem Vibram, np. Salomon Speedcross."

    with patch("intent_classifier._safe_call_llm", return_value=answer) as mock_llm:
        result = handle_gear_node(state)

    sys_prompt = _captured_system_prompt(mock_llm)
    assert "Najpierw odpowiedz" in sys_prompt
    assert result.coach_question == answer
    assert result.user_feedback == ""


def test_gear_unknown_discipline_defaults_to_trail():
    """When discipline is unknown, GEAR prompt must default to trail, not ask about health."""
    state = _make_state("Jakie buty na mokre szlaki?", "GEAR", "GEAR_SHOES_TRAIL", discipline="both")

    with patch("intent_classifier._safe_call_llm", return_value="Odpowiedź.") as mock_llm:
        handle_gear_node(state)

    sys_prompt = _captured_system_prompt(mock_llm)
    assert "trail" in sys_prompt.lower(), "GEAR prompt must mention trail as default"


# ─── Test 4: CONTEXT_INFO — "Byłem chory 3 dni" ──────────────────────────────

def test_context_info_illness_acknowledges_and_asks_one_question():
    """CONTEXT_INFO must acknowledge what user shared and ask exactly ONE question."""
    state = _make_state("Byłem chory 3 dni", "CONTEXT_INFO", "ILLNESS")
    llm_response = (
        "Rozumiem, byłeś chory przez 3 dni. "
        "Co chcesz z tym zrobić — przesunąć trening czy zmodyfikować plan?"
    )

    with patch("intent_classifier._safe_call_llm", return_value=llm_response):
        result = handle_context_info_node(state)

    assert "Rozumiem" in result.coach_question, "Response must acknowledge user's message"
    assert result.user_feedback == "", "user_feedback must be cleared — no auto plan"
    assert "illness" in result.extracted_context, "Context must be stored under illness key"


def test_context_info_does_not_auto_generate_plan():
    """After CONTEXT_INFO, user_feedback must be cleared so think_node never triggers revise_plan."""
    state = _make_state("Byłem chory 3 dni", "CONTEXT_INFO", "ILLNESS")

    with patch("intent_classifier._safe_call_llm", return_value="Rozumiem. Co chcesz zrobić?"):
        result = handle_context_info_node(state)

    assert result.user_feedback == "", "CRITICAL: user_feedback must be empty after CONTEXT_INFO"
    assert result.plan_accepted is False


def test_context_info_fallback_also_starts_with_rozumiem():
    """Even when LLM fails, the fallback response must start with 'Rozumiem'."""
    state = _make_state("Byłem chory 3 dni", "CONTEXT_INFO", "ILLNESS")

    with patch("intent_classifier._safe_call_llm", return_value=""):  # LLM failure
        result = handle_context_info_node(state)

    assert "Rozumiem" in result.coach_question, "Fallback must also acknowledge with 'Rozumiem'"
    assert result.user_feedback == ""


# ─── Test 5: TRAINING_PLAN — "Zrób mi plan na maraton za 12 tygodni" ──────────

def test_training_plan_create_does_not_call_llm_before_creating_plan():
    """PLAN_CREATE must NOT call LLM asking health questions — it passes through to think_node."""
    state = _make_state("Zrób mi plan na maraton za 12 tygodni", "TRAINING_PLAN", "PLAN_CREATE")

    with patch("intent_classifier._safe_call_llm") as mock_llm:
        result = handle_training_plan_node(state)

    assert not mock_llm.called, "PLAN_CREATE must not call LLM before plan is created"


def test_training_plan_create_preserves_user_feedback_for_think_node():
    """PLAN_CREATE must keep user_feedback intact so think_node triggers plan generation."""
    original = "Zrób mi plan na maraton za 12 tygodni"
    state = _make_state(original, "TRAINING_PLAN", "PLAN_CREATE")

    with patch("intent_classifier._safe_call_llm"):
        result = handle_training_plan_node(state)

    assert result.user_feedback == original, "user_feedback must survive handle_training_plan_node"


# ─── Test 6: ROUTE — "Gdzie biegać w Zakopanem?" ─────────────────────────────

def test_route_with_location_suggests_routes_without_health_check():
    """When location is in the query, route handler must suggest routes directly."""
    state = _make_state("Gdzie biegać w Zakopanem?", "ROUTE", "ROUTE_LOCATION", discipline="trail")
    answer = "W Zakopanem polecam Dolinę Chochołowską i szlak na Gubałówkę jako trasy biegowe."

    with patch("intent_classifier._safe_call_llm", return_value=answer) as mock_llm:
        result = handle_route_node(state)

    sys_prompt = _captured_system_prompt(mock_llm)
    assert "Najpierw odpowiedz" in sys_prompt
    assert result.coach_question == answer
    assert result.user_feedback == ""


def test_route_system_prompt_forbids_health_questions():
    """ROUTE system prompt must explicitly forbid health/wellness questions."""
    state = _make_state("Gdzie biegać w Zakopanem?", "ROUTE", "ROUTE_LOCATION")

    with patch("intent_classifier._safe_call_llm", return_value="Odpowiedź.") as mock_llm:
        handle_route_node(state)

    sys_prompt = _captured_system_prompt(mock_llm)
    assert "samopoczucie" not in sys_prompt.lower() or "nie pytaj" in sys_prompt.lower(), (
        "ROUTE prompt must forbid health questions"
    )


# ─── Test 7: RACE — "Jak rozłożyć siły na 50 km?" ────────────────────────────

def test_race_strategy_answers_directly_without_load_preamble():
    """Race strategy handler must deliver a direct answer — no load check before responding."""
    state = _make_state("Jak rozłożyć siły na 50 km?", "RACE", "RACE_STRATEGY", discipline="trail")
    answer = "Na 50 km zacznij 10–15% wolniej niż docelowe tempo, pierwsze 30 km traktuj jako rozgrzewkę."

    with patch("intent_classifier._safe_call_llm", return_value=answer) as mock_llm:
        result = handle_race_node(state)

    sys_prompt = _captured_system_prompt(mock_llm)
    assert "Najpierw odpowiedz" in sys_prompt
    assert result.coach_question == answer
    assert result.user_feedback == ""


def test_race_system_prompt_has_no_load_check():
    """RACE system prompt must not instruct LLM to check training load before answering."""
    state = _make_state("Jak rozłożyć siły na 50 km?", "RACE", "RACE_STRATEGY")

    with patch("intent_classifier._safe_call_llm", return_value="Odpowiedź.") as mock_llm:
        handle_race_node(state)

    sys_prompt = _captured_system_prompt(mock_llm).lower()
    assert "obciążenie" not in sys_prompt, "RACE prompt must not check training load"
    assert "zmęczenie" not in sys_prompt, "RACE prompt must not ask about fatigue"
