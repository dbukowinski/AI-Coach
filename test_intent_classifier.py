"""
Pytest tests for the intent_classifier module.
All tests mock _call_llm so no real Groq API key is required.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from agent_state import AgentState
from intent_classifier import intent_classifier_node


def _make_state(message: str) -> AgentState:
    state = AgentState()
    state.user_feedback = message
    return state


def _mock_response(intent: str, subtype: str, confidence: float = 0.95, discipline: str = "both") -> str:
    return json.dumps({"intent": intent, "subtype": subtype, "confidence": confidence, "discipline": discipline})


# ─── One test per required intent ────────────────────────────────────────────

@pytest.mark.parametrize("message,expected_intent,expected_subtype", [
    ("jak zrobić long run", "QUESTION", "Q_LONG_RUN"),
    ("chcę w weekend góry", "TRAINING_PLAN", "PLAN_MODIFY"),
    ("bolą mnie kolana", "CONTEXT_INFO", "INJURY"),
    ("co jeść przed startem", "NUTRITION", "NUTRITION_PRE"),
    ("jakie buty na mokre szlaki", "GEAR", "GEAR_SHOES_TRAIL"),
    ("zaplanuj trasę 20km z podejściami", "ROUTE", "ROUTE_REQUEST"),
    ("mam maraton za 6 tygodni", "RACE", "RACE_PREP"),
])
def test_intent_classification(message: str, expected_intent: str, expected_subtype: str) -> None:
    mock_json = _mock_response(expected_intent, expected_subtype)
    with patch("intent_classifier._call_llm", return_value=mock_json):
        state = _make_state(message)
        result = intent_classifier_node(state)

    assert result.intent == expected_intent, (
        f"Expected intent {expected_intent!r}, got {result.intent!r} for: {message!r}"
    )
    assert result.subtype == expected_subtype, (
        f"Expected subtype {expected_subtype!r}, got {result.subtype!r} for: {message!r}"
    )
    assert result.intent_confidence > 0.4


def test_fallback_on_invalid_json() -> None:
    """When Groq returns non-JSON, classifier must fall back to QUESTION / Q_LONG_RUN / 0.5."""
    with patch("intent_classifier._call_llm", return_value="this is not json at all"):
        state = _make_state("some random message")
        result = intent_classifier_node(state)

    assert result.intent == "QUESTION"
    assert result.subtype == "Q_LONG_RUN"
    assert result.intent_confidence == 0.5


# ─── Additional edge-case coverage ───────────────────────────────────────────

def test_fallback_on_low_confidence() -> None:
    """Confidence below 0.4 must fall back to QUESTION regardless of classification."""
    mock_json = _mock_response("TRAINING_PLAN", "PLAN_MODIFY", confidence=0.3)
    with patch("intent_classifier._call_llm", return_value=mock_json):
        state = _make_state("coś")
        result = intent_classifier_node(state)

    assert result.intent == "QUESTION"
    assert result.subtype == "Q_LONG_RUN"


def test_fallback_on_unknown_intent() -> None:
    """An intent not in VALID_INTENTS must fall back."""
    mock_json = json.dumps({"intent": "UNKNOWN", "subtype": "X", "confidence": 0.9, "discipline": "both"})
    with patch("intent_classifier._call_llm", return_value=mock_json):
        state = _make_state("coś")
        result = intent_classifier_node(state)

    assert result.intent == "QUESTION"
    assert result.subtype == "Q_LONG_RUN"


def test_fallback_on_mismatched_subtype() -> None:
    """A subtype that doesn't belong to its intent must fall back."""
    mock_json = json.dumps({"intent": "QUESTION", "subtype": "PLAN_MODIFY", "confidence": 0.9, "discipline": "both"})
    with patch("intent_classifier._call_llm", return_value=mock_json):
        state = _make_state("coś")
        result = intent_classifier_node(state)

    assert result.intent == "QUESTION"
    assert result.subtype == "Q_LONG_RUN"


def test_empty_message_defaults_to_question() -> None:
    """Empty user_feedback must default gracefully without calling Groq."""
    with patch("intent_classifier._call_llm") as mock_llm:
        state = _make_state("")
        result = intent_classifier_node(state)
        mock_llm.assert_not_called()

    assert result.intent == "QUESTION"
    assert result.subtype == "Q_LONG_RUN"


def test_discipline_stored_in_state() -> None:
    """discipline field from classifier response must be persisted to state."""
    mock_json = _mock_response("ROUTE", "ROUTE_REQUEST", confidence=0.95, discipline="trail")
    with patch("intent_classifier._call_llm", return_value=mock_json):
        state = _make_state("zaplanuj trasę 20km z podejściami")
        result = intent_classifier_node(state)

    assert result.discipline == "trail"


def test_user_message_stored_in_state() -> None:
    """user_feedback must be copied to user_message during classification."""
    mock_json = _mock_response("GEAR", "GEAR_SHOES_TRAIL", discipline="trail")
    with patch("intent_classifier._call_llm", return_value=mock_json):
        message = "jakie buty na mokre szlaki"
        state = _make_state(message)
        result = intent_classifier_node(state)

    assert result.user_message == message
